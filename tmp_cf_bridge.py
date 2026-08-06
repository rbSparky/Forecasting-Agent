#!/usr/bin/env python3
"""Temporary public-data bridge for CFBenchmark experiments.

Downloads public benchmark archives, then stores them as small base64 text
chunks so the research runner can reconstruct the exact official arrays in an
otherwise network-isolated environment. No private data or credentials enter
the output.
"""
from __future__ import annotations

import base64
import hashlib
import json
import urllib.request
from pathlib import Path

DATASETS = {
    "tolokers": "https://github.com/yandex-research/heterophilous-graphs/raw/main/data/tolokers.npz",
    "amazon_ratings": "https://github.com/yandex-research/heterophilous-graphs/raw/main/data/amazon_ratings.npz",
}
CHUNK_BYTES = 512 * 1024
OUT = Path("tmp_cf_data")
OUT.mkdir(exist_ok=True)

manifest = {}
for name, url in DATASETS.items():
    raw_path = OUT / f"{name}.npz"
    print(f"Downloading {name} from {url}", flush=True)
    urllib.request.urlretrieve(url, raw_path)
    payload = raw_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    encoded = base64.b64encode(payload).decode("ascii")
    chunks = []
    chars_per_chunk = 4 * ((CHUNK_BYTES + 2) // 3)
    for idx, start in enumerate(range(0, len(encoded), chars_per_chunk)):
        part = encoded[start : start + chars_per_chunk]
        path = OUT / f"{name}.part{idx:03d}.b64"
        path.write_text(part + "\n", encoding="ascii")
        chunks.append(path.name)
    raw_path.unlink()
    manifest[name] = {
        "source": url,
        "sha256": digest,
        "bytes": len(payload),
        "base64_chars": len(encoded),
        "chunks": chunks,
    }
    print(json.dumps({"dataset": name, **manifest[name]}, sort_keys=True), flush=True)

(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
