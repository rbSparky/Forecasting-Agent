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


def auc(y, s, idx):
    return 100.0 * float(roc_auc_score(y[idx], s[idx], average="macro"))


def csr_structure(src, dst, n):
    if np.all(src[1:] >= src[:-1]):
        order = None
        indices = dst.astype(np.int32, copy=False)
    else:
        order = np.lexsort((dst, src))
        src = src[order]
        indices = dst[order].astype(np.int32, copy=False)
    count = np.bincount(src, minlength=n).astype(np.int64)
    ptr = np.empty(n + 1, np.int64)
    ptr[0] = 0
    np.cumsum(count, out=ptr[1:])
    return src, indices, ptr, order


def apply(A, X, deg):
    H = np.asarray(A @ X, dtype=np.float32)
    H /= np.maximum(deg, 1e-12)[:, None]
    H[deg <= 1e-12] = 0
    return H


def ridge_bank(X, y, tr, va, te, tag, alphas):
    mu = X[tr].mean(0, dtype=np.float64)
    sd = X[tr].std(0, dtype=np.float64)
    keep = sd > 1e-7
    Z = ((X[:, keep] - mu[keep]) / sd[keep]).astype(np.float32)
    Z = np.concatenate([Z, np.ones((len(Z), 1), np.float32)], axis=1)
    Zt = Z[tr].astype(np.float64)
    G = Zt.T @ Zt
    B = Zt.T @ y[tr].astype(np.float64)
    I = np.eye(G.shape[0], dtype=np.float64)
    I[-1, -1] = 0
    scores, recs = [], []
    for a in alphas:
        tic = time.time()
        W = np.linalg.solve(G + a * I, B)
        S = (Z @ W).astype(np.float32)
        r = {"tag": tag, "alpha": a, "dim": int(Z.shape[1] - 1),
             "train": auc(y, S, tr), "val": auc(y, S, va), "test": auc(y, S, te),
             "solve_sec": time.time() - tic}
        emit(event="ridge", **r)
        scores.append(S); recs.append(r)
    return scores, recs


def taskwise(scores, recs, y, va, te, tag):
    m, c = len(scores), y.shape[1]
    V = np.empty((m, c), np.float64)
    T = np.empty((m, c), np.float64)
    for i, S in enumerate(scores):
        for j in range(c):
            V[i, j] = roc_auc_score(y[va, j], S[va, j])
            T[i, j] = roc_auc_score(y[te, j], S[te, j])
    p = V.argmax(0)
    S = np.empty_like(scores[0])
    for j in range(c):
        S[:, j] = scores[int(p[j])][:, j]
    r = {"tag": tag, "val": 100 * float(V[p, np.arange(c)].mean()),
         "test": 100 * float(T[p, np.arange(c)].mean()),
         "alpha_hist": {str(recs[i]["alpha"]): int(np.sum(p == i)) for i in range(m)}}
    emit(event="taskwise", **r)
    return S, r


def ztrain(S, tr):
    mu = S[tr].mean(0, dtype=np.float64)
    sd = S[tr].std(0, dtype=np.float64)
    return ((S - mu) / np.maximum(sd, 1e-6)).astype(np.float32)


def main():
    t0 = time.time()
    ds = NodePropPredDataset(name="ogbn-proteins", root=str(ROOT))
    graph, lab = ds[0]
    split = ds.get_idx_split()
    y = lab.astype(np.float32)
    tr = np.asarray(split["train"], np.int64)
    va = np.asarray(split["valid"], np.int64)
    te = np.asarray(split["test"], np.int64)
    n, c = y.shape
    edge = np.asarray(graph["edge_index"])
    src = edge[0].astype(np.int32, copy=True)
    dst = edge[1].astype(np.int32, copy=True)
    ef = np.asarray(graph["edge_feat"], np.float32)
    del graph, edge
    gc.collect()
    emit(event="loaded", n=n, tasks=c, edges=len(src), sec=time.time() - t0)

    src, indices, ptr, order = csr_structure(src, dst, n)
    if order is not None:
        ef = ef[order]
        del order
    deg = np.diff(ptr).astype(np.float32)
    emit(event="csr", sorted=bool(np.all(src[1:] >= src[:-1])), sec=time.time() - t0)

    x = np.empty((n, 8), np.float32)
    for r in range(8):
        x[:, r] = np.bincount(src, weights=ef[:, r], minlength=n) / np.maximum(deg, 1)

    prior = y[tr].mean(0, dtype=np.float64)
    scale = np.sqrt(np.maximum(prior * (1 - prior), 1e-5))
    ys = (y[tr].astype(np.float64) - prior) / scale
    ew, ev = np.linalg.eigh((ys.T @ ys) / len(tr))
    rank = 48
    V = ev[:, -rank:].astype(np.float32)
    z0 = np.zeros((n, rank), np.float32)
    z0[tr] = (((y[tr] - prior) / scale).astype(np.float32) @ V)
    y0 = np.zeros_like(y, np.float32); y0[tr] = y[tr]
    emit(event="label_code", rank=rank, explained=float(ew[-rank:].sum() / ew.sum()))

    hits = list(PRIOR.rglob("proteins_typed_scores.npy"))
    if len(hits) != 1:
        raise RuntimeError(f"prior score artifact missing: {hits}")
    s_struct = np.load(hits[0]).astype(np.float32, copy=False)
    emit(event="typed_struct", val=auc(y, s_struct, va), test=auc(y, s_struct, te))

    blocks = [s_struct, x, np.log1p(deg)[:, None].astype(np.float32)]
    label_blocks = []

    # Untyped direct and non-backtracking two-hop train-label messages.
    A0 = sp.csr_matrix((np.ones(len(indices), np.float32), indices, ptr), shape=(n, n), copy=False)
    g1 = apply(A0, y0, deg)
    diag0 = np.bincount(src, weights=1 / np.maximum(deg[indices], 1), minlength=n) / np.maximum(deg, 1)
    g2 = apply(A0, g1, deg) - diag0[:, None].astype(np.float32) * y0
    blocks += [g1, g2]; label_blocks += [g1, g2]
    emit(event="global_labels", h1_val=auc(y, g1, va), h1_test=auc(y, g1, te), sec=time.time() - t0)
    del A0, diag0
    gc.collect()

    combo = np.concatenate([z0, x], axis=1).astype(np.float32)
    stats = []
    for r in range(8):
        tic = time.time()
        raw = ef[:, r]
        w = np.maximum(raw - 0.001, 0).astype(np.float32)
        dw = np.bincount(src, weights=w, minlength=n).astype(np.float32)
        A = sp.csr_matrix((w, indices, ptr), shape=(n, n), copy=False)
        q1 = apply(A, combo, dw)
        q2 = apply(A, q1, dw)
        invj = 1 / np.maximum(dw[indices], 1e-12)
        diag = np.bincount(src, weights=w * w * invj, minlength=n).astype(np.float32) / np.maximum(dw, 1e-12)
        q2[:, :rank] -= diag[:, None] * z0
        q2[dw <= 1e-12] = 0
        blocks += [q1[:, :rank], q2[:, :rank], q1[:, rank:], q2[:, rank:]]
        label_blocks += [q1[:, :rank], q2[:, :rank]]
        positive = w > 0
        st = np.stack([np.log1p(dw),
                       np.bincount(src, weights=positive.astype(np.float32), minlength=n) / np.maximum(deg, 1)], 1).astype(np.float32)
        blocks.append(st)
        rec = {"channel": r, "positive_edges": int(positive.sum()),
               "positive_fraction": float(positive.mean()), "nonzero_nodes": int(np.sum(dw > 0)),
               "sec": time.time() - tic}
        stats.append(rec); emit(event="channel", **rec, total=time.time() - t0)
        del raw, w, dw, A, q1, q2, invj, diag, positive, st
        gc.collect()

    del ef, src, dst, indices, ptr, z0, combo, ys, ew, ev, V
    gc.collect()
    Xall = np.concatenate(blocks, axis=1).astype(np.float32)
    Xlab = np.concatenate(label_blocks + [np.log1p(deg)[:, None].astype(np.float32)], axis=1).astype(np.float32)
    emit(event="features", all_shape=list(Xall.shape), label_shape=list(Xlab.shape),
         all_mb=Xall.nbytes / 1e6, sec=time.time() - t0)

    alphas = [1e-2, .1, 1, 10, 100, 1000, 10000, 100000]
    AS, AR = ridge_bank(Xall, y, tr, va, te, "all", alphas)
    LS, LR = ridge_bank(Xlab, y, tr, va, te, "labels", alphas)
    ia = int(np.argmax([r["val"] for r in AR])); il = int(np.argmax([r["val"] for r in LR]))
    Sa, Ra = AS[ia], AR[ia]
    Sl, Rl = LS[il], LR[il]
    Sat, Rat = taskwise(AS, AR, y, va, te, "all_taskwise")
    Slt, Rlt = taskwise(LS, LR, y, va, te, "label_taskwise")

    candidates = [("all", Sa, Ra), ("labels", Sl, Rl), ("all_taskwise", Sat, Rat),
                  ("label_taskwise", Slt, Rlt),
                  ("typed_struct", s_struct, {"val": auc(y, s_struct, va), "test": auc(y, s_struct, te)})]
    bank = {"struct": ztrain(s_struct, tr), "all": ztrain(Sa, tr), "labels": ztrain(Sl, tr),
            "all_tw": ztrain(Sat, tr), "labels_tw": ztrain(Slt, tr)}
    best_fusion = None
    top = []
    keys = list(bank)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            for lam in np.linspace(0, 1, 41):
                S = (lam * bank[keys[i]] + (1 - lam) * bank[keys[j]]).astype(np.float32)
                r = {"a": keys[i], "b": keys[j], "lambda": float(lam),
                     "val": auc(y, S, va), "test": auc(y, S, te)}
                top.append(r)
                if best_fusion is None or r["val"] > best_fusion[1]["val"]:
                    best_fusion = (S.copy(), r)
    top.sort(key=lambda r: r["val"], reverse=True)
    for r in top[:25]: emit(event="fusion", **r)
    if best_fusion is not None:
        candidates.append(("fusion", best_fusion[0], best_fusion[1]))

    name, score, rec = max(candidates, key=lambda p: p[2]["val"])
    np.save(OUT / "proteins_channel_label_scores_v2.npy", score)
    final = {"best_name": name, "best": rec, "target_gat": 85.01, "target_deepergcn": 85.80,
             "closed_gat": bool(rec["test"] >= 85.01), "closed_sota": bool(rec["test"] >= 85.80),
             "channels": stats, "seconds": time.time() - t0}
    (OUT / "proteins_channel_label_cf_v2.json").write_text(json.dumps(final, indent=2, sort_keys=True))
    emit(event="FINAL", **final)


if __name__ == "__main__":
    main()
