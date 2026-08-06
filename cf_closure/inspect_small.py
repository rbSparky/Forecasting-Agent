from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path("cf_closure/_data")
OUT = Path("cf_closure/_results")
ROOT.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
BASE = "https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/main/data"


def download(name: str) -> Path:
    path = ROOT / name
    if not path.exists():
        urllib.request.urlretrieve(f"{BASE}/{name}", path)
    return path


def describe(name: str) -> dict:
    path = download(name)
    z = np.load(path, allow_pickle=False)
    return {
        "name": name,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "arrays": {
            k: {
                "shape": list(z[k].shape),
                "dtype": str(z[k].dtype),
                "min": float(np.nanmin(z[k])) if np.issubdtype(z[k].dtype, np.number) and z[k].size else None,
                "max": float(np.nanmax(z[k])) if np.issubdtype(z[k].dtype, np.number) and z[k].size else None,
            }
            for k in z.files
        },
    }


if __name__ == "__main__":
    report = [describe("tolokers.npz"), describe("amazon_ratings.npz")]
    p = OUT / "inspect_small.json"
    p.write_text(json.dumps(report, indent=2))
    print(p.read_text(), flush=True)
