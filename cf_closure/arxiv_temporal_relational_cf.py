from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from ogb.nodeproppred import NodePropPredDataset

ROOT = Path("cf_closure/_ogb")
OUT = Path("cf_closure/_results")
OUT.mkdir(parents=True, exist_ok=True)


def emit(**kw):
    print(json.dumps(kw, sort_keys=True), flush=True)


def acc(y: np.ndarray, s: np.ndarray, idx: np.ndarray) -> float:
    return 100.0 * float(np.mean(np.argmax(s[idx], axis=1) == y[idx]))


def row_norm(A: sp.csr_matrix) -> sp.csr_matrix:
    A = A.tocsr().astype(np.float32)
    d = np.asarray(A.sum(1)).ravel().astype(np.float32)
    A.data *= np.repeat(1.0 / np.maximum(d, 1e-12), np.diff(A.indptr))
    return A


def ridge_closed(X: np.ndarray, Y: np.ndarray, tr: np.ndarray, va: np.ndarray, te: np.ndarray,
                 alphas: list[float], tag: str) -> tuple[np.ndarray, dict]:
    mu = X[tr].mean(0, dtype=np.float64)
    sd = X[tr].std(0, dtype=np.float64)
    keep = sd > 1e-7
    Z = ((X[:, keep] - mu[keep]) / sd[keep]).astype(np.float32)
    Z = np.concatenate([Z, np.ones((len(Z), 1), np.float32)], axis=1)
    Zt = Z[tr].astype(np.float64)
    G = Zt.T @ Zt
    B = Zt.T @ Y[tr].astype(np.float64)
    I = np.eye(G.shape[0], dtype=np.float64); I[-1, -1] = 0
    best = None; best_s = None
    for a in alphas:
        W = np.linalg.solve(G + a * I, B)
        S = (Z @ W).astype(np.float32)
        rec = {"tag": tag, "alpha": a, "dim": int(Z.shape[1] - 1),
               "train": acc(labels, S, tr), "val": acc(labels, S, va), "test": acc(labels, S, te)}
        emit(event="ridge", **rec)
        if best is None or rec["val"] > best["val"]:
            best, best_s = rec, S
    assert best is not None and best_s is not None
    return best_s, best


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(e.sum(1, keepdims=True), 1e-30)


t0 = time.time()
ds = NodePropPredDataset(name="ogbn-arxiv", root=str(ROOT))
graph, lab = ds[0]
labels = lab.reshape(-1).astype(np.int64)
split = ds.get_idx_split(); tr = np.asarray(split["train"]); va = np.asarray(split["valid"]); te = np.asarray(split["test"])
n = int(graph["num_nodes"]); C = int(labels.max() + 1)
Y = np.eye(C, dtype=np.float32)[labels]
X = np.asarray(graph["node_feat"], dtype=np.float32)
year = np.asarray(graph["node_year"]).reshape(-1).astype(np.int16)
e = np.asarray(graph["edge_index"]); src = e[0].astype(np.int32); dst = e[1].astype(np.int32)
del graph, e; gc.collect()
emit(event="loaded", n=n, edges=len(src), d=X.shape[1], sec=time.time() - t0)

# Full transductive ZCA whitening and a year-residual view.
mu = X.mean(0, dtype=np.float64); Xc = X.astype(np.float64) - mu
cov = (Xc.T @ Xc) / len(Xc)
w, V = np.linalg.eigh(cov); floor = max(float(np.median(w)) * 1e-3, 1e-6)
Xw = (Xc @ (V * (1.0 / np.sqrt(np.maximum(w, floor)))) @ V.T).astype(np.float32)
del Xc, cov, w, V
Xyr = Xw.copy()
for yy in np.unique(year):
    m = year == yy
    Xyr[m] -= Xyr[m].mean(0)
emit(event="whitened", sec=time.time() - t0)

# Directed and undirected operators.
Aout = sp.coo_matrix((np.ones(len(src), np.float32), (src, dst)), shape=(n, n)).tocsr(); Aout.sum_duplicates(); Aout.data[:] = 1
Ain = Aout.T.tocsr()
Pout = row_norm(Aout.copy()); Pin = row_norm(Ain.copy())
Au = (Aout + Ain).tocsr(); Au.data[:] = 1; Au.setdiag(1); Au.eliminate_zeros()
deg_u = np.asarray(Au.sum(1)).ravel().astype(np.float32); inv = 1.0 / np.sqrt(np.maximum(deg_u, 1))
Psym = (sp.diags(inv) @ Au @ sp.diags(inv)).tocsr().astype(np.float32)
outdeg = np.asarray(Aout.sum(1)).ravel().astype(np.float32); indeg = np.asarray(Ain.sum(1)).ravel().astype(np.float32)

# Temporal edge relations: anomalous reverse, same year, gap 1, gap 2, gap 3-5, gap >=6.
delta = year[src].astype(np.int32) - year[dst].astype(np.int32)
type_id = np.where(delta < 0, 0, np.where(delta == 0, 1, np.where(delta == 1, 2, np.where(delta == 2, 3, np.where(delta <= 5, 4, 5))))).astype(np.int8)
rel_ops: list[sp.csr_matrix] = []
for t in range(6):
    m = type_id == t
    A = sp.coo_matrix((np.ones(int(m.sum()), np.float32), (src[m], dst[m])), shape=(n, n)).tocsr(); A.sum_duplicates(); A.data[:] = 1
    rel_ops.append(row_norm(A))
    emit(event="relation", relation=t, edges=int(m.sum()), sec=time.time() - t0)

# Feature blocks; each gets an independent closed-form Ridge expert to 40 logits.
blocks: list[tuple[str, np.ndarray]] = [("xw", Xw), ("xyr", Xyr)]
h = Xw
for k in range(1, 5):
    h = np.asarray(Psym @ h, dtype=np.float32); blocks.append((f"sym{k}", h)); blocks.append((f"symdiff{k}", h - Xw))
h = Xw
for k in range(1, 4):
    h = np.asarray(Pout @ h, dtype=np.float32); blocks.append((f"out{k}", h)); blocks.append((f"outdiff{k}", h - Xw))
h = Xw
for k in range(1, 3):
    h = np.asarray(Pin @ h, dtype=np.float32); blocks.append((f"in{k}", h)); blocks.append((f"indiff{k}", h - Xw))
for t, P in enumerate(rel_ops):
    blocks.append((f"temp{t}", np.asarray(P @ Xw, dtype=np.float32)))

experts = []
expert_records = []
for name, B in blocks:
    S, rec = ridge_closed(B, Y, tr, va, te, [1, 10, 100, 1000, 10000], f"expert_{name}")
    experts.append(S); expert_records.append(rec)
    emit(event="expert_best", name=name, best=rec, sec=time.time() - t0)

# Train-label relational sufficient statistics. No validation/test labels are used.
Y0 = np.zeros((n, C), np.float32); Y0[tr] = Y[tr]
label_blocks = []
# Directed powers and symmetric/reverse controls.
L = Y0
for k in range(1, 5):
    L = np.asarray(Pout @ L, dtype=np.float32); label_blocks.append(L); emit(event="label_out", hop=k, coverage=float(np.mean(L.sum(1) > 0)))
L = Y0
for k in range(1, 3):
    L = np.asarray(Pin @ L, dtype=np.float32); label_blocks.append(L)
L = Y0
for k in range(1, 4):
    L = np.asarray(Psym @ L, dtype=np.float32); label_blocks.append(L)

# Relation-specific raw and compatibility-corrected messages.
prior = np.bincount(labels[tr], minlength=C).astype(np.float64); prior /= prior.sum()
for t, P in enumerate(rel_ops):
    Lt = np.asarray(P @ Y0, dtype=np.float32)
    label_blocks.append(Lt)
    m = (type_id == t) & np.isin(src, tr) & np.isin(dst, tr)
    cnt = np.zeros((C, C), np.float64)
    np.add.at(cnt, (labels[src[m]], labels[dst[m]]), 1.0)
    # P(source class | cited-neighbor class), smoothed toward independent prior.
    cnt += 2.0 * prior[:, None] * prior[None, :]
    M = cnt / np.maximum(cnt.sum(0, keepdims=True), 1e-15)
    label_blocks.append((Lt @ M.T).astype(np.float32))
    emit(event="compat", relation=t, train_edges=int(m.sum()), diag=float(np.trace(cnt) / cnt.sum()))

# Coverage, temporal and structural scalars.
struct = [np.log1p(outdeg), np.log1p(indeg), np.log1p(deg_u), year.astype(np.float32),
          (year.astype(np.float32) - 2017.0), np.asarray(Pout @ year.astype(np.float32)[:, None]).ravel(),
          np.asarray(Pin @ year.astype(np.float32)[:, None]).ravel()]
for P in rel_ops:
    struct.append(np.asarray(P.sum(1)).ravel().astype(np.float32))
struct = np.stack(struct, axis=1).astype(np.float32)

Meta = np.concatenate(experts + label_blocks + [struct], axis=1).astype(np.float32)
emit(event="meta", shape=list(Meta.shape), mb=Meta.nbytes / 1e6, sec=time.time() - t0)
Smeta, bmeta = ridge_closed(Meta, Y, tr, va, te, [1, 10, 100, 1000, 10000, 100000], "temporal_relational_meta")

# One closed-form prediction-feedback round through the same typed operators.
Q = softmax(Smeta.astype(np.float64)).astype(np.float32)
feedback = [Q]
for P in [Pout, Pin, Psym] + rel_ops:
    feedback.append(np.asarray(P @ Q, dtype=np.float32))
Meta2 = np.concatenate([Meta] + feedback, axis=1).astype(np.float32)
emit(event="meta2", shape=list(Meta2.shape), mb=Meta2.nbytes / 1e6)
S2, b2 = ridge_closed(Meta2, Y, tr, va, te, [1, 10, 100, 1000, 10000, 100000], "feedback_meta")

# Direct Bayesian blends of class scores and directional label evidence.
results = []
for base_name, S in [("meta", Smeta), ("feedback", S2)]:
    for temp in [.25, .4, .6, .8, 1., 1.3, 1.7, 2.5]:
        U = S / temp
        for li, L in enumerate(label_blocks):
            LP = (L + 1e-5 * prior) / np.maximum(L.sum(1, keepdims=True) + 1e-5, 1e-12)
            logL = np.log(np.clip(LP, 1e-20, 1))
            for beta in [.02, .05, .1, .2, .35, .5, .75, 1., 1.5, 2., 3., 5.]:
                Z = U + beta * logL
                rec = {"base": base_name, "temp": temp, "label_block": li, "beta": beta,
                       "val": acc(labels, Z, va), "test": acc(labels, Z, te)}
                results.append(rec)
results.sort(key=lambda r: r["val"], reverse=True)
for r in results[:50]: emit(event="blend_top", **r)

candidates = [("meta", Smeta, bmeta), ("feedback", S2, b2)]
if results:
    r = results[0]; base = Smeta if r["base"] == "meta" else S2
    L = label_blocks[r["label_block"]]
    LP = (L + 1e-5 * prior) / np.maximum(L.sum(1, keepdims=True) + 1e-5, 1e-12)
    Sb = base / r["temp"] + r["beta"] * np.log(np.clip(LP, 1e-20, 1))
    candidates.append(("blend", Sb.astype(np.float32), r))
best_name, best_score, best_rec = max(candidates, key=lambda x: x[2]["val"])
np.save(OUT / "arxiv_temporal_relational_scores.npy", best_score)
final = {"event": "FINAL", "best_name": best_name, "best": best_rec,
         "target": 73.60, "closed": bool(best_rec["test"] >= 73.60), "sec": time.time() - t0}
(OUT / "arxiv_temporal_relational_cf.json").write_text(json.dumps(final, indent=2, sort_keys=True))
emit(**final)
