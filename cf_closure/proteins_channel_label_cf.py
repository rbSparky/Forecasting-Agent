from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from ogb.nodeproppred import NodePropPredDataset
from sklearn.metrics import roc_auc_score

ROOT = Path("cf_closure/_ogb")
OUT = Path("cf_closure/_results")
PRIOR = Path("cf_closure/_prior")
OUT.mkdir(parents=True, exist_ok=True)


def emit(**kw):
    print(json.dumps(kw, sort_keys=True), flush=True)


def macro_auc(y: np.ndarray, score: np.ndarray, idx: np.ndarray) -> float:
    return 100.0 * float(roc_auc_score(y[idx], score[idx], average="macro"))


def load_prior_score() -> np.ndarray:
    hits = list(PRIOR.rglob("proteins_typed_scores.npy"))
    if len(hits) != 1:
        raise RuntimeError(f"Expected one prior typed score artifact, found {hits}")
    x = np.load(hits[0]).astype(np.float32, copy=False)
    emit(event="prior", path=str(hits[0]), shape=list(x.shape))
    return x


def build_csr_structure(src: np.ndarray, dst: np.ndarray, n: int):
    sorted_src = bool(np.all(src[1:] >= src[:-1]))
    emit(event="edge_order", src_sorted=sorted_src)
    if sorted_src:
        counts = np.bincount(src, minlength=n).astype(np.int64)
        indptr = np.empty(n + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(counts, out=indptr[1:])
        return dst.astype(np.int32, copy=False), indptr, None
    order = np.lexsort((dst, src))
    src2 = src[order]
    dst2 = dst[order].astype(np.int32, copy=False)
    counts = np.bincount(src2, minlength=n).astype(np.int64)
    indptr = np.empty(n + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    return dst2, indptr, order


def make_csr(data: np.ndarray, indices: np.ndarray, indptr: np.ndarray, n: int) -> sp.csr_matrix:
    return sp.csr_matrix((data, indices, indptr), shape=(n, n), copy=False)


def row_apply(A: sp.csr_matrix, X: np.ndarray, deg: np.ndarray) -> np.ndarray:
    H = np.asarray(A @ X, dtype=np.float32)
    H /= np.maximum(deg, 1e-12)[:, None]
    H[deg <= 1e-12] = 0
    return H


def ridge_bank(X: np.ndarray, y: np.ndarray, tr: np.ndarray, va: np.ndarray, te: np.ndarray,
               tag: str, alphas: list[float]) -> tuple[list[np.ndarray], list[dict]]:
    mu = X[tr].mean(0, dtype=np.float64)
    sd = X[tr].std(0, dtype=np.float64)
    keep = sd > 1e-7
    Z = ((X[:, keep] - mu[keep]) / sd[keep]).astype(np.float32)
    Z = np.concatenate([Z, np.ones((len(Z), 1), np.float32)], axis=1)
    Zt = Z[tr].astype(np.float64)
    Yt = y[tr].astype(np.float64)
    G = Zt.T @ Zt
    B = Zt.T @ Yt
    I = np.eye(G.shape[0], dtype=np.float64)
    I[-1, -1] = 0.0
    scores: list[np.ndarray] = []
    recs: list[dict] = []
    for alpha in alphas:
        tic = time.time()
        W = np.linalg.solve(G + alpha * I, B)
        S = (Z @ W).astype(np.float32)
        rec = {
            "event": "ridge", "tag": tag, "alpha": alpha,
            "dim": int(Z.shape[1] - 1),
            "train": macro_auc(y, S, tr),
            "val": macro_auc(y, S, va),
            "test": macro_auc(y, S, te),
            "solve_sec": time.time() - tic,
        }
        emit(**rec)
        scores.append(S)
        recs.append(rec)
    return scores, recs


def zscore_train(S: np.ndarray, tr: np.ndarray) -> np.ndarray:
    mu = S[tr].mean(0, dtype=np.float64)
    sd = S[tr].std(0, dtype=np.float64)
    return ((S - mu) / np.maximum(sd, 1e-6)).astype(np.float32)


def taskwise_alpha(scores: list[np.ndarray], recs: list[dict], y: np.ndarray,
                   va: np.ndarray, te: np.ndarray, tag: str) -> tuple[np.ndarray, dict]:
    # Each of the 112 OGB outputs is an independent binary task. Select its
    # closed-form Ridge regularizer on validation AUC, never on test.
    c = y.shape[1]
    val_mat = np.full((len(scores), c), -np.inf, dtype=np.float64)
    test_mat = np.full((len(scores), c), np.nan, dtype=np.float64)
    for i, S in enumerate(scores):
        for j in range(c):
            val_mat[i, j] = roc_auc_score(y[va, j], S[va, j])
            test_mat[i, j] = roc_auc_score(y[te, j], S[te, j])
    pick = np.argmax(val_mat, axis=0)
    out = np.empty_like(scores[0])
    for j in range(c):
        out[:, j] = scores[int(pick[j])][:, j]
    rec = {
        "event": "taskwise_alpha", "tag": tag,
        "val": 100.0 * float(np.mean(val_mat[pick, np.arange(c)])),
        "test": 100.0 * float(np.mean(test_mat[pick, np.arange(c)])),
        "alpha_hist": {str(recs[i]["alpha"]): int(np.sum(pick == i)) for i in range(len(recs))},
    }
    emit(**rec)
    return out, rec


def main() -> None:
    t0 = time.time()
    ds = NodePropPredDataset(name="ogbn-proteins", root=str(ROOT))
    graph, labels = ds[0]
    split = ds.get_idx_split()
    y = labels.astype(np.float32)
    tr = np.asarray(split["train"], dtype=np.int64)
    va = np.asarray(split["valid"], dtype=np.int64)
    te = np.asarray(split["test"], dtype=np.int64)
    n, c = y.shape

    edge = np.asarray(graph["edge_index"])
    src = edge[0].astype(np.int32, copy=True)
    dst = edge[1].astype(np.int32, copy=True)
    edge_feat = np.asarray(graph["edge_feat"], dtype=np.float32)
    del graph, edge
    gc.collect()
    emit(event="loaded", n=n, c=c, edges=len(src), sec=time.time() - t0)

    indices, indptr, order = build_csr_structure(src, dst, n)
    if order is not None:
        edge_feat = edge_feat[order]
        src = src[order]
        dst = indices
        del order
        gc.collect()
    counts = np.diff(indptr).astype(np.float32)
    if int(indptr[-1]) != len(src):
        raise RuntimeError("CSR pointer mismatch")

    # Official node input: mean incident edge evidence. With the official
    # undirected edge list stored in both directions, outgoing and incoming
    # aggregation are equivalent.
    x_native = np.empty((n, 8), dtype=np.float32)
    for r in range(8):
        x_native[:, r] = np.bincount(src, weights=edge_feat[:, r], minlength=n) / np.maximum(counts, 1)
    emit(event="native", shape=list(x_native.shape), sec=time.time() - t0)

    # Macro-AUC-aware deterministic output code: standardize every task by its
    # train prevalence, then retain the leading train-only covariance modes.
    prior = y[tr].mean(0, dtype=np.float64)
    scale = np.sqrt(np.maximum(prior * (1.0 - prior), 1e-5))
    yt_std = (y[tr].astype(np.float64) - prior) / scale
    cov = (yt_std.T @ yt_std) / len(tr)
    ew, ev = np.linalg.eigh(cov)
    rank = 48
    V = ev[:, -rank:].astype(np.float32)
    explained = float(ew[-rank:].sum() / np.maximum(ew.sum(), 1e-30))
    z0 = np.zeros((n, rank), dtype=np.float32)
    z0[tr] = (((y[tr] - prior) / scale).astype(np.float32) @ V)
    y0 = np.zeros_like(y, dtype=np.float32)
    y0[tr] = y[tr]
    emit(event="label_code", rank=rank, explained=explained)

    s_struct = load_prior_score()
    emit(event="struct_score", val=macro_auc(y, s_struct, va), test=macro_auc(y, s_struct, te))

    blocks: list[np.ndarray] = [s_struct, x_native, np.log1p(counts)[:, None].astype(np.float32)]
    label_blocks: list[np.ndarray] = []

    # Untyped one- and two-hop label evidence. No self-loops are inserted. The
    # exact diagonal of P^2 is removed from the second hop, preventing a train
    # node's own target from returning along an immediate backtrack.
    ones = np.ones(len(indices), dtype=np.float32)
    A0 = make_csr(ones, indices, indptr, n)
    h1 = row_apply(A0, y0, counts)
    invdeg_dst = 1.0 / np.maximum(counts[indices], 1.0)
    diag0 = np.bincount(src, weights=invdeg_dst, minlength=n) / np.maximum(counts, 1.0)
    h2 = row_apply(A0, h1, counts)
    h2 -= diag0[:, None].astype(np.float32) * y0
    blocks.extend([h1, h2])
    label_blocks.extend([h1, h2])
    emit(event="global_label", h1_val=macro_auc(y, h1, va), h1_test=macro_auc(y, h1, te), sec=time.time() - t0)
    del A0, ones, invdeg_dst, diag0
    gc.collect()

    # True eight-channel typed adjacencies. The value 0.001 is OGB's evidence
    # floor; subtracting it avoids turning every channel into an almost
    # identical dense graph. Each operator carries both a train-label code and
    # the official node evidence. A diagonal correction removes the exact
    # two-step self-return of training labels.
    channel_stats = []
    combo0 = np.concatenate([z0, x_native], axis=1).astype(np.float32)
    for r in range(8):
        tic = time.time()
        raw = edge_feat[:, r]
        w = np.maximum(raw - 0.001, 0.0).astype(np.float32)
        positive = w > 0
        degw = np.bincount(src, weights=w, minlength=n).astype(np.float32)
        A = make_csr(w, indices, indptr, n)
        q1 = row_apply(A, combo0, degw)
        q2 = row_apply(A, q1, degw)
        inv_dst = 1.0 / np.maximum(degw[indices], 1e-12)
        diag_num = np.bincount(src, weights=w * w * inv_dst, minlength=n).astype(np.float32)
        diag = diag_num / np.maximum(degw, 1e-12)
        q2[:, :rank] -= diag[:, None] * z0
        q2[degw <= 1e-12] = 0

        # Label code, native feature messages, and typed coverage statistics.
        blocks.extend([q1[:, :rank], q2[:, :rank], q1[:, rank:], q2[:, rank:]])
        label_blocks.extend([q1[:, :rank], q2[:, :rank]])
        stat = np.stack([
            np.log1p(degw),
            np.bincount(src, weights=positive.astype(np.float32), minlength=n) / np.maximum(counts, 1),
            degw / np.maximum(np.bincount(src, weights=raw, minlength=n).astype(np.float32), 1e-12),
        ], axis=1).astype(np.float32)
        blocks.append(stat)
        channel_stats.append({
            "r": r, "positive_edges": int(positive.sum()),
            "positive_fraction": float(positive.mean()),
            "nonzero_nodes": int(np.sum(degw > 0)),
            "seconds": time.time() - tic,
        })
        emit(event="channel", **channel_stats[-1], total_sec=time.time() - t0)
        del raw, w, positive, degw, A, q1, q2, inv_dst, diag_num, diag, stat
        gc.collect()

    del edge_feat, src, dst, indices, indptr, z0, combo0, yt_std, cov, ew, ev, V
    gc.collect()
    X = np.concatenate(blocks, axis=1).astype(np.float32)
    emit(event="features", shape=list(X.shape), mb=X.nbytes / 1e6, channel_stats=channel_stats, sec=time.time() - t0)

    alphas = [1e-3, 1e-2, .1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1e6]
    all_scores, all_recs = ridge_bank(X, y, tr, va, te, "all_channel_label", alphas)
    label_X = np.concatenate(label_blocks + [np.log1p(counts)[:, None].astype(np.float32)], axis=1)
    label_scores, label_recs = ridge_bank(label_X, y, tr, va, te, "label_only", alphas)

    candidates: list[tuple[str, np.ndarray, dict]] = []
    for S, r in zip(all_scores, all_recs):
        candidates.append((f"all_a{r['alpha']}", S, r))
    for S, r in zip(label_scores, label_recs):
        candidates.append((f"label_a{r['alpha']}", S, r))
    candidates.append(("typed_structural", s_struct, {
        "val": macro_auc(y, s_struct, va), "test": macro_auc(y, s_struct, te), "tag": "typed_structural"}))

    Sall_tw, rall_tw = taskwise_alpha(all_scores, all_recs, y, va, te, "all_channel_label")
    Slab_tw, rlab_tw = taskwise_alpha(label_scores, label_recs, y, va, te, "label_only")
    candidates.extend([("all_taskwise", Sall_tw, rall_tw), ("label_taskwise", Slab_tw, rlab_tw)])

    # Validation-selected deterministic score fusion. AUC is invariant to each
    # task's affine calibration, so normalize on train before interpolation.
    bank = {
        "struct": zscore_train(s_struct, tr),
        "all": zscore_train(max(zip(all_scores, all_recs), key=lambda p: p[1]["val"])[0], tr),
        "all_tw": zscore_train(Sall_tw, tr),
        "label": zscore_train(max(zip(label_scores, label_recs), key=lambda p: p[1]["val"])[0], tr),
        "label_tw": zscore_train(Slab_tw, tr),
    }
    fusion_records = []
    keys = list(bank)
    grid = np.linspace(0.0, 1.0, 41)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            for lam in grid:
                S = (lam * bank[keys[i]] + (1.0 - lam) * bank[keys[j]]).astype(np.float32)
                rec = {"event": "fusion", "a": keys[i], "b": keys[j], "lambda": float(lam),
                       "val": macro_auc(y, S, va), "test": macro_auc(y, S, te)}
                fusion_records.append((S, rec))
    fusion_records.sort(key=lambda p: p[1]["val"], reverse=True)
    for _, r in fusion_records[:25]:
        emit(**r)
    if fusion_records:
        candidates.append(("fusion", fusion_records[0][0], fusion_records[0][1]))

    best_name, best_score, best_rec = max(candidates, key=lambda p: p[2]["val"])
    np.save(OUT / "proteins_channel_label_scores.npy", best_score.astype(np.float32))
    final = {
        "event": "FINAL", "best_name": best_name, "best": best_rec,
        "target_gat": 85.01, "target_deepergcn": 85.80,
        "closed_gat": bool(best_rec["test"] >= 85.01),
        "closed_sota": bool(best_rec["test"] >= 85.80),
        "channel_stats": channel_stats,
        "seconds": time.time() - t0,
    }
    (OUT / "proteins_channel_label_cf.json").write_text(json.dumps(final, indent=2, sort_keys=True))
    emit(**final)


if __name__ == "__main__":
    main()
