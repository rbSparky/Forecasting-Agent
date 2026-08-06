#!/usr/bin/env python3
"""Validation-only deterministic Tolokers search on all ten official splits.

The script deliberately never reads ``test_masks``.  Every label-dependent
feature used by the final Ridge stack is five-fold cross-fitted inside the
corresponding official training split.  The output is therefore safe for
selecting one global configuration before a single locked test evaluation.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh, solve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "tmp_cf_data"
RESULT_DIR = ROOT / "tmp_cf_results"
RESULT_DIR.mkdir(exist_ok=True)


def emit(**kw):
    print("RESULT " + json.dumps(kw, sort_keys=True), flush=True)


def restore_npz(name: str) -> np.lib.npyio.NpzFile:
    manifest = json.loads((DATA_DIR / "manifest.json").read_text())[name]
    encoded = "".join((DATA_DIR / part).read_text().strip() for part in manifest["chunks"])
    payload = base64.b64decode(encoded)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != manifest["sha256"]:
        raise RuntimeError(f"checksum mismatch for {name}: {digest}")
    return np.load(io.BytesIO(payload), allow_pickle=False)


def orient_masks(raw: np.ndarray, n: int) -> np.ndarray:
    m = np.asarray(raw, dtype=bool)
    if m.shape[0] == n:
        return m
    if m.shape[1] == n:
        return m.T
    raise ValueError((m.shape, n))


def load_graph():
    z = restore_npz("tolokers")
    x = np.asarray(z["node_features"], dtype=np.float32)
    y = np.asarray(z["node_labels"], dtype=np.int64).reshape(-1)
    e = np.asarray(z["edges"], dtype=np.int64)
    if e.shape[0] == 2 and e.shape[1] != 2:
        e = e.T
    n = len(y)
    u, v = e[:, 0].astype(np.int32), e[:, 1].astype(np.int32)
    a = sp.coo_matrix(
        (np.ones(2 * len(e), dtype=np.float32), (np.r_[u, v], np.r_[v, u])),
        shape=(n, n),
    ).tocsr()
    a.sum_duplicates()
    a.data[:] = 1.0
    a.setdiag(0)
    a.eliminate_zeros()
    train_masks = orient_masks(z["train_masks"], n)
    val_masks = orient_masks(z["val_masks"], n)
    # Intentionally do not access test_masks in this development script.
    return x, y, a, u, v, train_masks, val_masks


def rownorm(a: sp.csr_matrix, self_loop: bool = False) -> sp.csr_matrix:
    b = a + sp.eye(a.shape[0], format="csr", dtype=np.float32) if self_loop else a
    d = np.asarray(b.sum(axis=1)).ravel()
    return (sp.diags((1.0 / np.maximum(d, 1.0)).astype(np.float32)) @ b).tocsr()


def symnorm_self(a: sp.csr_matrix) -> sp.csr_matrix:
    b = a + sp.eye(a.shape[0], format="csr", dtype=np.float32)
    d = np.asarray(b.sum(axis=1)).ravel()
    q = (1.0 / np.sqrt(np.maximum(d, 1.0))).astype(np.float32)
    return (sp.diags(q) @ b @ sp.diags(q)).tocsr()


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def standardize_all(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64)
    med = np.median(z, axis=0)
    q25, q75 = np.percentile(z, [25, 75], axis=0)
    scale = np.maximum(q75 - q25, 1e-6)
    z = np.clip((z - med) / scale, -8.0, 8.0)
    return z.astype(np.float32)


def attention_mean(xn: np.ndarray, a: sp.csr_matrix, temp: float = 0.5) -> np.ndarray:
    coo = a.tocoo()
    r, c = coo.row.astype(np.int32), coo.col.astype(np.int32)
    sim = np.einsum("ij,ij->i", xn[r], xn[c]).astype(np.float32)
    w = np.exp(np.clip(sim / temp, -20.0, 20.0)).astype(np.float32)
    ids = np.arange(len(xn), dtype=np.int32)
    r = np.r_[r, ids]
    c = np.r_[c, ids]
    w = np.r_[w, np.full(len(xn), np.exp(1.0 / temp), dtype=np.float32)]
    wm = sp.coo_matrix((w, (r, c)), shape=a.shape).tocsr()
    den = np.asarray(wm.sum(axis=1)).ravel()
    return np.asarray(wm @ xn, dtype=np.float32) / np.maximum(den[:, None], 1e-12)


def engineered_node_features(x: np.ndarray, a: sp.csr_matrix) -> np.ndarray:
    rates = np.clip(x[:, :4].astype(np.float64), 1e-6, 1.0)
    education = x[:, 4:8].astype(np.float64)
    english = x[:, 8:10].astype(np.float64)
    pair = []
    for i in range(4):
        for j in range(i, 4):
            pair.append((rates[:, i] * rates[:, j])[:, None])
    entropy = -(rates * np.log(rates)).sum(axis=1, keepdims=True)
    sorted_rates = np.sort(rates, axis=1)
    margin = (sorted_rates[:, -1] - sorted_rates[:, -2])[:, None]
    quality = (rates[:, 0] - rates[:, 3])[:, None]
    completion = (1.0 - rates[:, 1] - rates[:, 2])[:, None]
    rate_logits = np.log(rates) - np.log(np.maximum(1.0 - rates, 1e-6))
    edu_rate = np.einsum("ni,nj->nij", education, rates).reshape(len(x), -1)
    eng_rate = np.einsum("ni,nj->nij", english, rates).reshape(len(x), -1)

    degree = np.asarray(a.sum(axis=1)).ravel().astype(np.float64)
    r = rownorm(a)
    logd = np.log1p(degree)
    nbr_logd = np.asarray(r @ logd).ravel()
    nbr_logd2 = np.asarray(r @ (logd * logd)).ravel()
    nbr_var = np.maximum(nbr_logd2 - nbr_logd * nbr_logd, 0.0)
    two_logd = np.asarray(r @ nbr_logd).ravel()
    degree_rank = np.argsort(np.argsort(degree, kind="mergesort"), kind="mergesort") / max(len(degree) - 1, 1)
    structural = np.column_stack(
        [
            logd,
            np.sqrt(degree),
            1.0 / np.sqrt(np.maximum(degree, 1.0)),
            degree_rank,
            nbr_logd,
            np.sqrt(nbr_var),
            two_logd,
            logd - nbr_logd,
        ]
    )
    out = np.column_stack(
        [
            x,
            np.sqrt(rates),
            np.log1p(100.0 * rates),
            rate_logits,
            *pair,
            entropy,
            margin,
            quality,
            completion,
            edu_rate,
            eng_rate,
            structural,
        ]
    )
    return standardize_all(out)


def build_h0(x: np.ndarray, a: sp.csr_matrix, variant: str):
    if variant == "basic":
        node = normalize_rows(x).astype(np.float32)
    elif variant == "rich":
        node = normalize_rows(engineered_node_features(x, a)).astype(np.float32)
    else:
        raise ValueError(variant)
    p = symnorm_self(a)
    h1 = np.asarray(p @ node, dtype=np.float32)
    h2 = np.asarray(p @ h1, dtype=np.float32)
    h3 = np.asarray(p @ h2, dtype=np.float32)
    h4 = np.asarray(p @ h3, dtype=np.float32)
    v1 = np.maximum(np.asarray(p @ (node * node)) - h1 * h1, 0.0).astype(np.float32)
    v2 = np.maximum(np.asarray(p @ (p @ (node * node))) - h2 * h2, 0.0).astype(np.float32)
    att = attention_mean(node, a, temp=0.5)
    h0 = np.column_stack(
        [node, h1, h2, h3, h4, v1, v2, node - h1, h1 - h2, h2 - h3, h3 - h4, att]
    ).astype(np.float32)
    return h0, p


def ridge_multi(f: np.ndarray, ids: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    ft = np.asarray(f[ids], dtype=np.float64)
    yt = np.asarray(target[ids], dtype=np.float64)
    g = ft.T @ ft
    b = ft.T @ yt
    w = solve(g + alpha * np.eye(g.shape[0]), b, assume_a="pos", check_finite=False)
    return (np.asarray(f, dtype=np.float64) @ w).astype(np.float32)


def lcf(h0: np.ndarray, p: sp.csr_matrix, y: np.ndarray, train_ids: np.ndarray, k: int, alpha: float):
    target = np.eye(2, dtype=np.float32)[y]
    h = h0.copy()
    predictions = []
    for _ in range(k):
        agg = np.tanh(np.asarray(p @ h, dtype=np.float32))
        pred = ridge_multi(agg, train_ids, target, alpha)
        predictions.append(pred)
        h = np.column_stack([h, pred]).astype(np.float32)
    return h, predictions


def median_distance(h: np.ndarray, train_ids: np.ndarray, sample: int = 900) -> float:
    take = train_ids[np.linspace(0, len(train_ids) - 1, min(sample, len(train_ids))).astype(int)]
    z = np.asarray(h[take], dtype=np.float64)
    n2 = np.sum(z * z, axis=1)
    d2 = np.maximum(n2[:, None] + n2[None, :] - 2.0 * z @ z.T, 0.0)
    vals = np.sqrt(d2[np.triu_indices(len(z), 1)])
    vals = vals[vals > 1e-10]
    return float(np.median(vals))


def linear_score(h: np.ndarray, y: np.ndarray, train_ids: np.ndarray, alpha: float = 10.0) -> np.ndarray:
    z = np.asarray(h, dtype=np.float64)
    mu, sd = z[train_ids].mean(axis=0), z[train_ids].std(axis=0)
    z = (z - mu) / np.maximum(sd, 1e-8)
    z = np.column_stack([z, np.ones(len(z))])
    target = (2 * y - 1).astype(np.float64)
    gram = z[train_ids].T @ z[train_ids]
    rhs = z[train_ids].T @ target[train_ids]
    reg = np.eye(gram.shape[0]) * alpha
    reg[-1, -1] = 0.0
    return z @ solve(gram + reg, rhs, assume_a="sym", check_finite=False)


def nystrom_score(
    h: np.ndarray,
    y: np.ndarray,
    train_ids: np.ndarray,
    m: int,
    sigma_factor: float,
    lam: float,
):
    z = np.asarray(h, dtype=np.float64)
    mu, sd = z[train_ids].mean(axis=0), z[train_ids].std(axis=0)
    z = (z - mu) / np.maximum(sd, 1e-8)
    med = median_distance(z, train_ids)
    sigma = max(sigma_factor * med, 1e-8)
    m = min(m, len(train_ids))
    neg, pos = train_ids[y[train_ids] == 0], train_ids[y[train_ids] == 1]
    m0 = int(round(m * len(neg) / len(train_ids)))
    m1 = m - m0
    l0 = neg[np.linspace(0, len(neg) - 1, m0).astype(int)]
    l1 = pos[np.linspace(0, len(pos) - 1, m1).astype(int)]
    landmarks = np.sort(np.r_[l0, l1])
    zl = z[landmarks]
    nl = np.sum(zl * zl, axis=1)
    na = np.sum(z * z, axis=1)
    wd = np.maximum(nl[:, None] + nl[None, :] - 2.0 * zl @ zl.T, 0.0)
    wmat = np.exp(-wd / (2.0 * sigma * sigma))
    wmat = (wmat + wmat.T) * 0.5
    eig, vec = eigh(wmat, check_finite=False, driver="evr")
    keep = eig > max(float(eig.max()) * 1e-9, 1e-10)
    eig, vec = eig[keep], vec[:, keep]
    transform = vec / np.sqrt(eig)[None, :]
    feat = np.empty((len(z), len(eig)), dtype=np.float32)
    for start in range(0, len(z), 2000):
        q = z[start : start + 2000]
        dd = np.maximum(na[start : start + 2000, None] + nl[None, :] - 2.0 * q @ zl.T, 0.0)
        feat[start : start + 2000] = (np.exp(-dd / (2.0 * sigma * sigma)) @ transform).astype(np.float32)
    return linear_score(feat, y, train_ids, alpha=lam), {
        "m": m,
        "rank": int(len(eig)),
        "median": med,
        "sigma": sigma,
    }


def weighted_operator(u: np.ndarray, v: np.ndarray, n: int, edge_weight: np.ndarray) -> sp.csr_matrix:
    w = np.asarray(edge_weight, dtype=np.float32)
    mat = sp.coo_matrix((np.r_[w, w], (np.r_[u, v], np.r_[v, u])), shape=(n, n)).tocsr()
    den = np.asarray(mat.sum(axis=1)).ravel()
    return (sp.diags(1.0 / np.maximum(den, 1e-12)) @ mat).tocsr()


def make_operators(x: np.ndarray, a: sp.csr_matrix, u: np.ndarray, v: np.ndarray):
    n = len(x)
    xn = normalize_rows(x)
    degree = np.asarray(a.sum(axis=1)).ravel().astype(np.float64)
    sim = np.einsum("ij,ij->i", xn[u], xn[v]).astype(np.float64)
    education = x[:, 4:8].argmax(axis=1)
    same = education[u] == education[v]
    return {
        "unif": weighted_operator(u, v, n, np.ones(len(u))),
        "sim2": weighted_operator(u, v, n, np.exp(2.0 * sim)),
        "sim5": weighted_operator(u, v, n, np.exp(5.0 * sim)),
        "invnbrdeg": weighted_operator(u, v, n, 1.0 / np.sqrt(np.maximum(degree[u] * degree[v], 1.0))),
        "sameedu": weighted_operator(u, v, n, np.where(same, 1.0, 0.15)),
        "diffedu": weighted_operator(u, v, n, np.where(same, 0.15, 1.0)),
    }


def label_features(y: np.ndarray, seed_ids: np.ndarray, operators: dict[str, sp.csr_matrix]):
    n = len(y)
    mask = np.zeros(n, dtype=np.float64)
    label = np.zeros(n, dtype=np.float64)
    mask[seed_ids] = 1.0
    label[seed_ids] = y[seed_ids]
    prior = float(y[seed_ids].mean())
    features = []
    names = []
    for name, op in operators.items():
        hy, hm = label.copy(), mask.copy()
        max_hop = 8 if name == "unif" else 4
        for hop in range(1, max_hop + 1):
            hy, hm = op @ hy, op @ hm
            ratio = np.divide(hy, hm, out=np.full(n, prior), where=hm > 1e-10)
            centered = (ratio - prior) / max(math.sqrt(prior * (1.0 - prior)), 1e-6)
            support = np.log1p(100.0 * hm)
            features.extend([ratio, centered, support])
            names.extend([f"{name}_lp{hop}", f"{name}_center{hop}", f"{name}_support{hop}"])
        alphas = (0.03, 0.1, 0.2, 0.4) if name == "unif" else (0.1,)
        for alpha in alphas:
            hy, hm = label.copy(), mask.copy()
            for _ in range(100):
                new_y = alpha * label + (1.0 - alpha) * (op @ hy)
                new_m = alpha * mask + (1.0 - alpha) * (op @ hm)
                if max(np.max(np.abs(new_y - hy)), np.max(np.abs(new_m - hm))) < 1e-10:
                    hy, hm = new_y, new_m
                    break
                hy, hm = new_y, new_m
            ratio = np.divide(hy, hm, out=np.full(n, prior), where=hm > 1e-10)
            features.extend([ratio, (ratio - prior), np.log1p(100.0 * hm)])
            names.extend([f"{name}_ppr{alpha}", f"{name}_ppr_center{alpha}", f"{name}_ppr_support{alpha}"])

    unif = operators["unif"]
    for beta in (0.5, 0.8, 0.95, 0.99):
        f = np.full(n, prior, dtype=np.float64)
        f[seed_ids] = y[seed_ids]
        for _ in range(200):
            new = beta * (unif @ f) + (1.0 - beta) * prior
            new[seed_ids] = y[seed_ids]
            if np.max(np.abs(new - f)) < 1e-10:
                f = new
                break
            f = new
        features.append(f)
        names.append(f"clamped_{beta}")
    return np.column_stack(features).astype(np.float32), names


def unsupervised_meta(x: np.ndarray, a: sp.csr_matrix) -> np.ndarray:
    degree = np.asarray(a.sum(axis=1)).ravel().astype(np.float64)
    r = rownorm(a)
    logd = np.log1p(degree)
    parts = [x, logd[:, None], np.sqrt(degree)[:, None], (1.0 / np.sqrt(np.maximum(degree, 1.0)))[:, None]]
    for z in [logd, x[:, 0], x[:, 1], x[:, 2], x[:, 3]]:
        mean = np.asarray(r @ z).ravel()
        var = np.asarray(r @ (z * z)).ravel() - mean * mean
        two = np.asarray(r @ mean).ravel()
        parts.extend([mean[:, None], np.maximum(var, 0.0)[:, None], two[:, None], (z - mean)[:, None]])
    return standardize_all(np.column_stack(parts))


def base_features(
    h0: np.ndarray,
    p: sp.csr_matrix,
    y: np.ndarray,
    seed_ids: np.ndarray,
    config: dict,
    m: int,
):
    h, plist = lcf(h0, p, y, seed_ids, k=config["k"], alpha=config["lcf_alpha"])
    kernel, kcfg = nystrom_score(
        h,
        y,
        seed_ids,
        m=m,
        sigma_factor=config["sigma_factor"],
        lam=config["kernel_lambda"],
    )
    linear = linear_score(h, y, seed_ids, alpha=10.0)
    layers = np.column_stack([q[:, 1] - q[:, 0] for q in plist])
    return np.column_stack([kernel, linear, layers]).astype(np.float32), kcfg


def residual_features(base_score: np.ndarray, y: np.ndarray, seed_ids: np.ndarray, op: sp.csr_matrix):
    score = np.asarray(base_score, dtype=np.float64)
    mu, sd = score[seed_ids].mean(), max(score[seed_ids].std(), 1e-8)
    calibrated = 1.0 / (1.0 + np.exp(-np.clip((score - mu) / sd, -20.0, 20.0)))
    residual = np.zeros(len(y), dtype=np.float64)
    mask = np.zeros(len(y), dtype=np.float64)
    residual[seed_ids] = y[seed_ids] - calibrated[seed_ids]
    mask[seed_ids] = 1.0
    feats = []
    for alpha in (0.05, 0.1, 0.2, 0.4):
        rr, mm = residual.copy(), mask.copy()
        for _ in range(100):
            nr = alpha * residual + (1.0 - alpha) * (op @ rr)
            nm = alpha * mask + (1.0 - alpha) * (op @ mm)
            if max(np.max(np.abs(nr - rr)), np.max(np.abs(nm - mm))) < 1e-10:
                rr, mm = nr, nm
                break
            rr, mm = nr, nm
        corr = np.divide(rr, mm, out=np.zeros(len(y)), where=mm > 1e-10)
        feats.extend([corr, calibrated + corr])
    return np.column_stack(feats).astype(np.float32)


def fit_stacks(oof: np.ndarray, full: np.ndarray, y_train: np.ndarray, groups: dict[str, list[int]]):
    target = (2 * y_train - 1).astype(np.float64)
    scores = {}
    for group, cols in groups.items():
        a = np.asarray(oof[:, cols], dtype=np.float64)
        b = np.asarray(full[:, cols], dtype=np.float64)
        mu, sd = a.mean(axis=0), a.std(axis=0)
        a = (a - mu) / np.maximum(sd, 1e-8)
        b = (b - mu) / np.maximum(sd, 1e-8)
        a = np.column_stack([a, np.ones(len(a))])
        b = np.column_stack([b, np.ones(len(b))])
        for pos_weight in (0.5, 0.75, 1.0, 1.25):
            sample_weight = np.where(y_train == 1, pos_weight, 1.0)
            aw = a * np.sqrt(sample_weight[:, None])
            tw = target * np.sqrt(sample_weight)
            gram, rhs = aw.T @ aw, aw.T @ tw
            for alpha in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0):
                reg = np.eye(gram.shape[0]) * alpha
                reg[-1, -1] = 0.0
                key = f"{group}|pw={pos_weight}|a={alpha}"
                scores[key] = (b @ solve(gram + reg, rhs, assume_a="sym", check_finite=False)).astype(np.float32)
    return scores


def main():
    t0 = time.time()
    x, y, a, u, v, train_masks, val_masks = load_graph()
    n, n_splits = len(y), train_masks.shape[1]
    emit(
        stage="data",
        n=n,
        edges=int(a.nnz // 2),
        dims=int(x.shape[1]),
        splits=int(n_splits),
        positive=float(y.mean()),
    )
    operators = make_operators(x, a, u, v)
    unsup = unsupervised_meta(x, a)
    representations = {variant: build_h0(x, a, variant) for variant in ("basic", "rich")}

    configs = [
        {"name": "basic_k6_a0.5_sf2", "variant": "basic", "k": 6, "lcf_alpha": 0.5, "sigma_factor": 2.0, "kernel_lambda": 0.1},
        {"name": "basic_k9_a2_sf2", "variant": "basic", "k": 9, "lcf_alpha": 2.0, "sigma_factor": 2.0, "kernel_lambda": 0.1},
        {"name": "rich_k6_a0.5_sf2", "variant": "rich", "k": 6, "lcf_alpha": 0.5, "sigma_factor": 2.0, "kernel_lambda": 0.1},
        {"name": "rich_k9_a2_sf2", "variant": "rich", "k": 9, "lcf_alpha": 2.0, "sigma_factor": 2.0, "kernel_lambda": 0.1},
    ]

    records = defaultdict(list)
    split_details = []
    for split in range(n_splits):
        split_start = time.time()
        train_ids = np.flatnonzero(train_masks[:, split])
        val_ids = np.flatnonzero(val_masks[:, split])
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1729 + split)
        split_row = {"split": split, "train": int(len(train_ids)), "val": int(len(val_ids)), "configs": {}}

        # Label propagation is independent of the unary representation, so cache it once.
        lp_oof = None
        lp_full = None
        lp_names = None
        fold_indices = list(skf.split(train_ids, y[train_ids]))
        for fold, (sub_pos, hold_pos) in enumerate(fold_indices):
            seed_ids, hold_ids = train_ids[sub_pos], train_ids[hold_pos]
            lf, lp_names = label_features(y, seed_ids, operators)
            if lp_oof is None:
                lp_oof = np.empty((len(train_ids), lf.shape[1]), dtype=np.float32)
            lp_oof[hold_pos] = lf[hold_ids]
            emit(stage="lp_fold", split=split, fold=fold, dims=int(lf.shape[1]))
        lp_full, lp_names = label_features(y, train_ids, operators)

        for config in configs:
            cfg_start = time.time()
            h0, p = representations[config["variant"]]
            base_oof = None
            residual_oof = None
            fold_base_auc = []
            for fold, (sub_pos, hold_pos) in enumerate(fold_indices):
                seed_ids, hold_ids = train_ids[sub_pos], train_ids[hold_pos]
                bf, bcfg = base_features(h0, p, y, seed_ids, config, m=650)
                rf = residual_features(bf[:, 0], y, seed_ids, operators["unif"])
                if base_oof is None:
                    base_oof = np.empty((len(train_ids), bf.shape[1]), dtype=np.float32)
                    residual_oof = np.empty((len(train_ids), rf.shape[1]), dtype=np.float32)
                base_oof[hold_pos] = bf[hold_ids]
                residual_oof[hold_pos] = rf[hold_ids]
                fold_base_auc.append(100.0 * roc_auc_score(y[hold_ids], bf[hold_ids, 0]))
                emit(
                    stage="base_fold",
                    split=split,
                    config=config["name"],
                    fold=fold,
                    hold_auc=fold_base_auc[-1],
                    rank=bcfg["rank"],
                )

            bf, full_cfg = base_features(h0, p, y, train_ids, config, m=900)
            rf = residual_features(bf[:, 0], y, train_ids, operators["unif"])
            meta_oof = np.column_stack([base_oof, lp_oof, residual_oof, unsup[train_ids]]).astype(np.float32)
            meta_full = np.column_stack([bf, lp_full, rf, unsup]).astype(np.float32)
            bdim = base_oof.shape[1]
            lstart, lend = bdim, bdim + lp_oof.shape[1]
            rstart, rend = lend, lend + residual_oof.shape[1]
            ustart = rend
            groups = {
                "base": [0],
                "base_layers": list(range(bdim)),
                "base_lp": [0] + list(range(lstart, lend)),
                "base_lp_resid": [0] + list(range(lstart, rend)),
                "all": list(range(meta_full.shape[1])),
                "base_struct": [0] + list(range(ustart, meta_full.shape[1])),
            }
            score_map = fit_stacks(meta_oof, meta_full, y[train_ids], groups)
            score_map["kernel_direct"] = bf[:, 0]
            score_map["linear_direct"] = bf[:, 1]
            local = {}
            for key, score in score_map.items():
                val = 100.0 * roc_auc_score(y[val_ids], score[val_ids])
                global_key = f"{config['name']}|{key}"
                records[global_key].append(val)
                local[key] = val
            best_local = max(local, key=local.get)
            split_row["configs"][config["name"]] = {
                "best_key": best_local,
                "best_val": local[best_local],
                "kernel_val": local["kernel_direct"],
                "linear_val": local["linear_direct"],
                "fold_base_auc": fold_base_auc,
                "kernel_cfg": full_cfg,
                "seconds": time.time() - cfg_start,
            }
            emit(
                stage="config_done",
                split=split,
                config=config["name"],
                best=best_local,
                best_val=local[best_local],
                kernel_val=local["kernel_direct"],
                seconds=time.time() - cfg_start,
            )
        split_row["seconds"] = time.time() - split_start
        split_details.append(split_row)
        emit(stage="split_done", split=split, seconds=split_row["seconds"])

    summaries = {}
    for key, values in records.items():
        arr = np.asarray(values, dtype=np.float64)
        summaries[key] = {
            "mean_val": float(arr.mean()),
            "std_val": float(arr.std(ddof=1)),
            "values": arr.tolist(),
        }
    selected = max(summaries, key=lambda key: summaries[key]["mean_val"])
    ordered = sorted(summaries.items(), key=lambda kv: kv[1]["mean_val"], reverse=True)
    result = {
        "dataset": "tolokers",
        "stage": "validation_only",
        "protocol": "official_10_splits_global_mean_validation_selection; test_masks_not_read",
        "selected": selected,
        "selected_summary": summaries[selected],
        "top20": [{"key": key, **summary} for key, summary in ordered[:20]],
        "all_candidates": summaries,
        "split_details": split_details,
        "elapsed_seconds": time.time() - t0,
    }
    out = RESULT_DIR / "tolokers_dev_v1.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    emit(
        stage="FINAL_VALIDATION",
        selected=selected,
        mean_val=summaries[selected]["mean_val"],
        std_val=summaries[selected]["std_val"],
        elapsed_seconds=result["elapsed_seconds"],
    )


if __name__ == "__main__":
    main()
