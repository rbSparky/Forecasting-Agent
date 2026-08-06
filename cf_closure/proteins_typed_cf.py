from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from ogb.nodeproppred import NodePropPredDataset
from sklearn.metrics import roc_auc_score

ROOT = Path("cf_closure/_ogb")
OUT = Path("cf_closure/_results")
OUT.mkdir(parents=True, exist_ok=True)


def emit(**kw):
    print(json.dumps(kw, sort_keys=True), flush=True)


def auc(y: np.ndarray, s: np.ndarray, idx: np.ndarray) -> float:
    return 100.0 * float(roc_auc_score(y[idx], s[idx], average="macro"))


def ridge_scores(H: np.ndarray, Y: np.ndarray, tr: np.ndarray, va: np.ndarray, te: np.ndarray,
                 alphas: list[float], tag: str) -> tuple[np.ndarray, dict]:
    # Closed-form multi-output primal Ridge, with train-only centering/scaling and intercept.
    mu = H[tr].mean(0, dtype=np.float64)
    sd = H[tr].std(0, dtype=np.float64)
    keep = sd > 1e-7
    X = ((H[:, keep] - mu[keep]) / sd[keep]).astype(np.float32)
    X = np.concatenate([X, np.ones((len(X), 1), np.float32)], axis=1)
    Xt = X[tr].astype(np.float64)
    Yt = Y[tr].astype(np.float64)
    G = Xt.T @ Xt
    B = Xt.T @ Yt
    best = None
    best_s = None
    I = np.eye(G.shape[0], dtype=np.float64)
    I[-1, -1] = 0.0
    for a in alphas:
        W = np.linalg.solve(G + a * I, B)
        S = (X @ W).astype(np.float32)
        rec = {"event": "ridge", "tag": tag, "alpha": a, "dim": int(X.shape[1] - 1),
               "val": auc(Y, S, va), "test": auc(Y, S, te), "train": auc(Y, S, tr)}
        emit(**rec)
        if best is None or rec["val"] > best["val"]:
            best, best_s = rec, S
    assert best is not None and best_s is not None
    return best_s, best


def row_normalize(A: sp.csr_matrix) -> sp.csr_matrix:
    d = np.asarray(A.sum(1)).ravel().astype(np.float32)
    A = A.tocsr().astype(np.float32)
    A.data *= np.repeat(1.0 / np.maximum(d, 1e-12), np.diff(A.indptr))
    return A


def relation_blocks(src: np.ndarray, dst: np.ndarray, edge_feat: np.ndarray, X: np.ndarray,
                    types: np.ndarray, scheme: str) -> list[np.ndarray]:
    n = len(X)
    blocks: list[np.ndarray] = []
    for t in range(8):
        tic = time.time()
        m = types == t
        k = int(m.sum())
        if k == 0:
            blocks.extend([np.zeros_like(X), np.zeros_like(X), np.zeros((n, 2), np.float32)])
            continue
        rr = dst[m]
        cc = src[m]
        # Confidence is the evidence value in the assigned relation, floored for stability.
        ww = np.maximum(edge_feat[m, t], 1e-3).astype(np.float32, copy=False)
        A = sp.coo_matrix((ww, (rr, cc)), shape=(n, n), dtype=np.float32).tocsr()
        A.sum_duplicates()
        counts = np.bincount(rr, minlength=n).astype(np.float32)
        wdeg = np.asarray(A.sum(1)).ravel().astype(np.float32)
        P = row_normalize(A)
        h1 = np.asarray(P @ X, dtype=np.float32)
        h2 = np.asarray(P @ h1, dtype=np.float32)
        total_deg = np.maximum(np.bincount(dst, minlength=n).astype(np.float32), 1.0)
        stats = np.stack([np.log1p(counts), counts / total_deg], axis=1).astype(np.float32)
        blocks.extend([h1, h2, stats])
        emit(event="relation", scheme=scheme, relation=t, edges=k, sec=time.time() - tic)
        del rr, cc, ww, A, P, h1, h2, stats
        gc.collect()
    return blocks


def main() -> None:
    t0 = time.time()
    ds = NodePropPredDataset(name="ogbn-proteins", root=str(ROOT))
    graph, labels = ds[0]
    split = ds.get_idx_split()
    Y = labels.astype(np.float32)
    n = int(graph["num_nodes"])
    tr = np.asarray(split["train"], dtype=np.int64)
    va = np.asarray(split["valid"], dtype=np.int64)
    te = np.asarray(split["test"], dtype=np.int64)
    edge_index = np.asarray(graph["edge_index"])
    src = edge_index[0].astype(np.int32, copy=True)
    dst = edge_index[1].astype(np.int32, copy=True)
    edge_feat = np.asarray(graph["edge_feat"], dtype=np.float32)
    del graph, edge_index
    gc.collect()
    E = len(src)
    emit(event="loaded", n=n, edges=E, sec=time.time() - t0)

    sid = np.full(n, -1, np.int8)
    sid[tr] = 0; sid[va] = 1; sid[te] = 2
    edge_split = np.zeros((3, 3), np.int64)
    np.add.at(edge_split, (sid[src], sid[dst]), 1)
    species = np.asarray(ds[0][0]["node_species"]).reshape(-1)
    emit(event="audit", edge_split=edge_split.tolist(), fraction_cross=float(np.mean(sid[src] != sid[dst])),
         fraction_same_species=float(np.mean(species[src] == species[dst])),
         species_sets=[int(len(np.unique(species[i]))) for i in [tr, va, te]],
         species_intersections=[[int(len(np.intersect1d(np.unique(species[a]), np.unique(species[b]))))
                                for b in [tr, va, te]] for a in [tr, va, te]])
    del species, sid, edge_split

    # Official 8-D node feature (mean incident edge evidence) plus richer local edge statistics.
    deg = np.bincount(dst, minlength=n).astype(np.float32)
    means = np.empty((n, 8), np.float32)
    sqmeans = np.empty((n, 8), np.float32)
    maxes = np.full((n, 8), -np.inf, np.float32)
    hist = []
    for r in range(8):
        v = edge_feat[:, r]
        means[:, r] = np.bincount(dst, weights=v, minlength=n) / np.maximum(deg, 1)
        sqmeans[:, r] = np.bincount(dst, weights=v * v, minlength=n) / np.maximum(deg, 1)
        np.maximum.at(maxes[:, r], dst, v)
        for th in [.002, .005, .01, .03, .1, .3, .7]:
            hist.append((np.bincount(dst, weights=(v > th).astype(np.float32), minlength=n) /
                         np.maximum(deg, 1)).astype(np.float32))
        emit(event="local_channel", relation=r, sec=time.time() - t0)
    maxes[~np.isfinite(maxes)] = 0
    stds = np.sqrt(np.maximum(sqmeans - means * means, 0))
    Hblocks: list[np.ndarray] = [means, stds, maxes, np.log1p(deg)[:, None].astype(np.float32),
                                 np.stack(hist, axis=1).astype(np.float32)]

    # Global evidence-weighted message passing.
    wtot = np.maximum(edge_feat.sum(1), 1e-3).astype(np.float32)
    A = sp.coo_matrix((wtot, (dst, src)), shape=(n, n), dtype=np.float32).tocsr()
    A.sum_duplicates(); P = row_normalize(A)
    h = means
    for k in range(1, 4):
        h = np.asarray(P @ h, dtype=np.float32)
        Hblocks.extend([h, h * h, h - means])
        emit(event="global_hop", hop=k, sec=time.time() - t0)
    del A, P, wtot, h
    gc.collect()

    # Two deterministic edge typings: raw dominant evidence and log-standardized dominant evidence.
    raw_type = np.argmax(edge_feat, axis=1).astype(np.int8)
    logef = np.log(np.maximum(edge_feat, 1e-5))
    lmu = logef.mean(0, dtype=np.float64); lsd = logef.std(0, dtype=np.float64)
    ztype = np.argmax((logef - lmu) / np.maximum(lsd, 1e-9), axis=1).astype(np.int8)
    del logef
    for scheme, typ in [("raw", raw_type), ("logz", ztype)]:
        Hblocks.extend(relation_blocks(src, dst, edge_feat, means, typ, scheme))
        emit(event="scheme_done", scheme=scheme, sec=time.time() - t0)
    del raw_type, ztype
    gc.collect()

    H = np.concatenate(Hblocks, axis=1).astype(np.float32)
    del Hblocks
    emit(event="features", shape=list(H.shape), mb=H.nbytes / 1e6, sec=time.time() - t0)
    np.save(OUT / "proteins_typed_features.npy", H)

    alphas = [1e-3, 1e-2, .1, 1, 10, 100, 1000, 10000, 100000]
    score_bank = []
    S0, b0 = ridge_scores(H, Y, tr, va, te, alphas, "typed_linear")
    score_bank.append(("linear", S0, b0))
    # Fixed analytic nonlinear lifts preserve a single final Ridge solve.
    for scale in [.5, 1., 2.]:
        Z = np.concatenate([H, np.tanh(H / scale).astype(np.float32)], axis=1)
        S, b = ridge_scores(Z, Y, tr, va, te, alphas, f"typed_tanh_{scale}")
        score_bank.append((f"tanh_{scale}", S, b)); del Z; gc.collect()
    Z = np.concatenate([H, np.sign(H) * np.sqrt(np.abs(H) + 1e-8), H * H], axis=1).astype(np.float32)
    S, b = ridge_scores(Z, Y, tr, va, te, alphas, "typed_power")
    score_bank.append(("power", S, b)); del Z; gc.collect()

    best_name, best_score, best_rec = max(score_bank, key=lambda x: x[2]["val"])
    np.save(OUT / "proteins_typed_scores.npy", best_score)
    final = {"event": "FINAL", "best": best_rec, "best_name": best_name,
             "target_gat": 85.01, "target_deepergcn": 85.80,
             "closed_gat": bool(best_rec["test"] >= 85.01),
             "closed_sota": bool(best_rec["test"] >= 85.80), "sec": time.time() - t0}
    (OUT / "proteins_typed_cf.json").write_text(json.dumps(final, indent=2, sort_keys=True))
    emit(**final)


if __name__ == "__main__":
    main()
