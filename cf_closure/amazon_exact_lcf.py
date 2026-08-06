#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import scipy.sparse as sp
from scipy.linalg.lapack import spotrf, spotrs
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize

DATA_URL = "https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/main/data/amazon_ratings.npz"


def emit(**kw: object) -> None:
    print("RESULT " + json.dumps(kw, sort_keys=True), flush=True)


def get_data(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    p = root / "amazon_ratings.npz"
    if not p.exists():
        urllib.request.urlretrieve(DATA_URL, p)
    emit(event="data", bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    return p


def orient_mask(z: np.lib.npyio.NpzFile, key: str, n: int) -> np.ndarray:
    m = z[key].astype(bool, copy=False)
    return m.T if m.shape[0] != n else m


def build_graph_features(X: np.ndarray, edges: np.ndarray, tau: float) -> tuple[np.ndarray, sp.csr_matrix]:
    n = X.shape[0]
    u0 = edges[:, 0].astype(np.int64, copy=False)
    v0 = edges[:, 1].astype(np.int64, copy=False)
    r = np.concatenate([u0, v0, np.arange(n, dtype=np.int64)])
    c = np.concatenate([v0, u0, np.arange(n, dtype=np.int64)])
    A = sp.coo_matrix((np.ones(r.size, dtype=np.float32), (r, c)), shape=(n, n)).tocsr()
    A.sum_duplicates(); A.data.fill(1.0)
    deg = np.asarray(A.sum(axis=1)).ravel().astype(np.float32)
    inv = 1.0 / np.sqrt(np.maximum(deg, 1.0))
    An = (sp.diags(inv) @ A @ sp.diags(inv)).tocsr().astype(np.float32)

    Xn = normalize(X, norm="l2", axis=1).astype(np.float32, copy=False)
    h1 = np.asarray(An @ Xn, dtype=np.float32)
    h2 = np.asarray(An @ h1, dtype=np.float32)
    h3 = np.asarray(An @ h2, dtype=np.float32)
    var1 = np.asarray(An @ (Xn * Xn), dtype=np.float32) - h1 * h1
    var2 = np.asarray(An @ (h1 * h1), dtype=np.float32) - h2 * h2

    u = np.concatenate([u0, v0])
    v = np.concatenate([v0, u0])
    sim = np.einsum("ij,ij->i", Xn[u], Xn[v], optimize=True)
    w = np.exp(np.clip(sim / tau, -20.0, 20.0)).astype(np.float32)
    W = sp.coo_matrix((w, (u, v)), shape=(n, n)).tocsr()
    den = np.asarray(W.sum(axis=1)).ravel().astype(np.float32)
    att = np.asarray(sp.diags(1.0 / np.maximum(den, 1e-8)) @ W @ Xn, dtype=np.float32)

    H = np.concatenate(
        [Xn, h1, h2, h3, var1, var2, Xn - h1, h1 - h2, h2 - h3, att], axis=1
    ).astype(np.float32, copy=False)
    emit(event="H0", shape=list(H.shape), mib=H.nbytes / 2**20, tau=tau)
    return H, An


def lcf_hidden(H: np.ndarray, An: sp.csr_matrix, y: np.ndarray, train: np.ndarray,
               depth: int, alpha: float) -> np.ndarray:
    C = int(y.max()) + 1
    Y = np.eye(C, dtype=np.float32)[y]
    for k in range(depth):
        t0 = time.time()
        A_k = np.asarray(An @ H, dtype=np.float32)
        model = Ridge(alpha=alpha, fit_intercept=False, solver="lsqr", tol=1e-5, max_iter=5000)
        model.fit(A_k[train], Y[train])
        P = model.predict(A_k).astype(np.float32, copy=False)
        emit(event="layer", layer=k + 1,
             train=100.0 * accuracy_score(y[train], P[train].argmax(axis=1)),
             dims=int(H.shape[1] + C), seconds=time.time() - t0)
        H = np.concatenate([H, P], axis=1).astype(np.float32, copy=False)
        del A_k, P, model
        gc.collect()
    return H


def kernel_inplace(A: np.ndarray, B: np.ndarray, sigma: float) -> np.ndarray:
    G = np.matmul(A, B.T, dtype=np.float32)
    G *= np.float32(-2.0)
    G += np.einsum("ij,ij->i", A, A, optimize=True).astype(np.float32)[:, None]
    G += np.einsum("ij,ij->i", B, B, optimize=True).astype(np.float32)[None, :]
    np.maximum(G, 0.0, out=G)
    G *= np.float32(-0.5 / (sigma * sigma))
    np.exp(G, out=G)
    return G


def monotone_cumulative(Q: np.ndarray) -> np.ndarray:
    return np.minimum.accumulate(Q, axis=1)


def optimal_thresholds(score: np.ndarray, labels: np.ndarray, C: int) -> list[float]:
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    yy = labels[order]
    N = len(s)
    pref = np.zeros((C, N + 1), dtype=np.int32)
    for c in range(C):
        pref[c, 1:] = np.cumsum(yy == c)
    dp = np.full((C, N + 1), -10**9, dtype=np.int32)
    back = np.zeros((C, N + 1), dtype=np.int32)
    dp[0] = pref[0]
    for c in range(1, C):
        best = -10**9
        best_i = 0
        for j in range(N + 1):
            candidate = int(dp[c - 1, j] - pref[c, j])
            if candidate > best:
                best = candidate; best_i = j
            dp[c, j] = pref[c, j] + best
            back[c, j] = best_i
    positions: list[int] = []
    j = N
    for c in range(C - 1, 0, -1):
        j = int(back[c, j]); positions.append(j)
    positions.reverse()
    cuts: list[float] = []
    for j in positions:
        if j <= 0: cuts.append(float(s[0] - 1e-6))
        elif j >= N: cuts.append(float(s[-1] + 1e-6))
        else: cuts.append(float((s[j - 1] + s[j]) * 0.5))
    return cuts


def predict_heads(S: np.ndarray, y: np.ndarray, val: np.ndarray, C: int) -> list[dict[str, object]]:
    val_idx = np.flatnonzero(val)
    multi = S[:, :C]
    scalar = S[:, C]
    cum = monotone_cumulative(S[:, C + 1:C + 1 + C - 1])
    code = np.column_stack([np.arange(C)[:, None] > k for k in range(C - 1)]).astype(np.float32)
    dist = ((cum[:, None, :] - code[None, :, :]) ** 2).sum(axis=2)
    cls = np.arange(C, dtype=np.float32)[None, :] / float(C - 1)
    cuts = optimal_thresholds(scalar[val_idx], y[val_idx], C)
    out: list[dict[str, object]] = [
        {"name": "multiclass", "pred": multi.argmax(axis=1)},
        {"name": "scalar_threshold", "pred": np.searchsorted(np.asarray(cuts), scalar, side="right"), "cuts": cuts},
        {"name": "cumulative_code", "pred": dist.argmin(axis=1)},
    ]
    for gamma in [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
        for eta in [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
            score = multi - np.float32(gamma) * dist - np.float32(eta) * (scalar[:, None] - cls) ** 2
            out.append({"name": "hybrid", "gamma": gamma, "eta": eta, "pred": score.argmax(axis=1)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--depth", type=int, default=9)
    ap.add_argument("--ridge-alpha", type=float, default=2.0)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--kernel-lambda", type=float, default=0.01)
    ap.add_argument("--data-dir", type=Path, default=Path("cf_closure/_data"))
    ap.add_argument("--out-dir", type=Path, default=Path("cf_closure/_results/amazon_exact"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t_all = time.time()

    z = np.load(get_data(args.data_dir), allow_pickle=False)
    X = z["node_features"].astype(np.float32, copy=False)
    y = z["node_labels"].astype(np.int64, copy=False).reshape(-1)
    edges = z["edges"].astype(np.int64, copy=False)
    n = len(y); C = int(y.max()) + 1
    trm, vam, tem = [orient_mask(z, k, n) for k in ["train_masks", "val_masks", "test_masks"]]
    train, val, test = trm[:, args.split], vam[:, args.split], tem[:, args.split]
    tr_idx, va_idx, te_idx = np.flatnonzero(train), np.flatnonzero(val), np.flatnonzero(test)

    H, An = build_graph_features(X, edges, args.tau)
    H = lcf_hidden(H, An, y, train, args.depth, args.ridge_alpha)
    emit(event="Hfinal", shape=list(H.shape), mib=H.nbytes / 2**20,
         row_norm_median=float(np.median(np.linalg.norm(H, axis=1))))

    Y_multi = np.eye(C, dtype=np.float32)[y]
    Y_scalar = (y.astype(np.float32) / np.float32(C - 1))[:, None]
    Y_cum = np.column_stack([y > k for k in range(C - 1)]).astype(np.float32)
    Y = np.concatenate([Y_multi, Y_scalar, Y_cum], axis=1).astype(np.float32)

    Htr = np.ascontiguousarray(H[tr_idx])
    t0 = time.time()
    K = kernel_inplace(Htr, Htr, args.sigma)
    K.flat[::K.shape[0] + 1] += np.float32(args.kernel_lambda)
    Kf = np.asfortranarray(K)
    del K; gc.collect()
    chol, info = spotrf(Kf, lower=1, overwrite_a=1, clean=0)
    if info != 0:
        raise RuntimeError(f"spotrf failed: info={info}")
    alpha, info = spotrs(chol, np.asfortranarray(Y[tr_idx]), lower=1, overwrite_b=0)
    if info != 0:
        raise RuntimeError(f"spotrs failed: info={info}")
    emit(event="factor", n_train=len(tr_idx), seconds=time.time() - t0)
    del Kf, chol; gc.collect()

    S = np.empty((n, Y.shape[1]), dtype=np.float32)
    for idx_name, idx in [("train", tr_idx), ("val", va_idx), ("test", te_idx)]:
        for lo in range(0, len(idx), 256):
            ids = idx[lo:lo + 256]
            Kb = kernel_inplace(np.ascontiguousarray(H[ids]), Htr, args.sigma)
            S[ids] = Kb @ alpha
            del Kb
        emit(event="predict", split=idx_name, n=len(idx))
    heads = predict_heads(S, y, val, C)
    records: list[dict[str, object]] = []
    for rec in heads:
        pred = rec.pop("pred")
        row = dict(rec)
        row["train"] = 100.0 * accuracy_score(y[tr_idx], pred[tr_idx])
        row["val"] = 100.0 * accuracy_score(y[va_idx], pred[va_idx])
        row["test"] = 100.0 * accuracy_score(y[te_idx], pred[te_idx])
        records.append(row)
    best = max(records, key=lambda r: float(r["val"]))
    multi = next(r for r in records if r["name"] == "multiclass")
    payload = {
        "split": args.split,
        "config": vars(args) | {"data_dir": str(args.data_dir), "out_dir": str(args.out_dir)},
        "baseline_multiclass": multi,
        "best_validation_head": best,
        "all_heads": records,
        "seconds": time.time() - t_all,
    }
    (args.out_dir / f"split{args.split}.json").write_text(json.dumps(payload, indent=2))
    np.save(args.out_dir / f"split{args.split}_scores.npy", S)
    emit(event="FINAL", **{k: v for k, v in best.items()}, baseline_test=multi["test"], total_seconds=time.time() - t_all)


if __name__ == "__main__":
    main()
