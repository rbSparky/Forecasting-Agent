from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from ogb.nodeproppred import NodePropPredDataset


def arr_info(x: np.ndarray | None) -> dict | None:
    if x is None:
        return None
    a = np.asarray(x)
    out = {"shape": list(a.shape), "dtype": str(a.dtype), "bytes": int(a.nbytes)}
    if a.size and np.issubdtype(a.dtype, np.number):
        flat = a.reshape(-1)
        take = flat if flat.size <= 2_000_000 else flat[np.linspace(0, flat.size - 1, 2_000_000, dtype=np.int64)]
        out.update({
            "min": float(np.nanmin(take)),
            "max": float(np.nanmax(take)),
            "mean": float(np.nanmean(take)),
            "std": float(np.nanstd(take)),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ogbn-arxiv", "ogbn-proteins"])
    ap.add_argument("--root", default="cf_closure/_ogb")
    ap.add_argument("--out", default="cf_closure/_results")
    args = ap.parse_args()

    ds = NodePropPredDataset(name=args.dataset, root=args.root)
    graph, labels = ds[0]
    split = ds.get_idx_split()
    report: dict = {
        "dataset": args.dataset,
        "num_nodes": int(graph["num_nodes"]),
        "labels": arr_info(labels),
        "split": {k: {"n": int(len(v)), "min": int(np.min(v)), "max": int(np.max(v))} for k, v in split.items()},
        "graph": {k: arr_info(v) if isinstance(v, np.ndarray) else v for k, v in graph.items()},
    }

    edge = np.asarray(graph["edge_index"])
    src, dst = edge
    report["edge_checks"] = {
        "self_loops": int(np.sum(src == dst)),
        "src_degree_quantiles": np.quantile(np.bincount(src, minlength=graph["num_nodes"]), [0, .1, .5, .9, .99, 1]).tolist(),
        "dst_degree_quantiles": np.quantile(np.bincount(dst, minlength=graph["num_nodes"]), [0, .1, .5, .9, .99, 1]).tolist(),
        "sample_sha256": hashlib.sha256(edge[:, : min(edge.shape[1], 1_000_000)].astype("<i8", copy=False).tobytes()).hexdigest(),
    }

    if args.dataset == "ogbn-arxiv":
        year = np.asarray(graph["node_year"]).reshape(-1)
        delta = year[src] - year[dst]
        report["temporal"] = {
            "year_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(year, return_counts=True))},
            "edge_year_delta_quantiles": np.quantile(delta, [0, .01, .1, .5, .9, .99, 1]).tolist(),
            "fraction_src_newer": float(np.mean(delta > 0)),
            "fraction_same_year": float(np.mean(delta == 0)),
            "fraction_src_older": float(np.mean(delta < 0)),
        }
    else:
        ef = np.asarray(graph["edge_feat"], dtype=np.float32)
        report["edge_features"] = {
            "per_channel_mean": ef.mean(0).tolist(),
            "per_channel_std": ef.std(0).tolist(),
            "per_channel_min": ef.min(0).tolist(),
            "per_channel_max": ef.max(0).tolist(),
            "correlation": np.corrcoef(ef[: min(len(ef), 2_000_000)].T).round(6).tolist(),
        }
        if graph.get("node_feat") is not None:
            nf = np.asarray(graph["node_feat"], dtype=np.float32)
            report["node_edge_feature_relation"] = {
                "node_feat_mean": nf.mean(0).tolist(),
                "node_feat_std": nf.std(0).tolist(),
            }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.dataset}-probe.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(path.read_text(), flush=True)


if __name__ == "__main__":
    main()
