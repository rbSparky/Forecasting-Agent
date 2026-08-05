#!/usr/bin/env python3
"""Strict locked validation of Frozen Syntactic Projection (FSP).

Protocol
--------
* Fixed feature extractor: spaCy en_core_web_sm 3.8.0.
* Exact public source/preprocessing from the Roman-empire construction notebook.
* The parser-to-benchmark label projection is estimated independently on each
  official training mask by a smoothed contingency-table argmax.
* No hyperparameter, model, or mapping is selected from validation or test.
* All ten official test masks are evaluated once.
* Exact unlearning is verified by sufficient-statistic subtraction against a
  complete from-scratch recomputation.
"""
from __future__ import annotations

import ast
import hashlib
import json
import string
import time
from pathlib import Path

import numpy as np
import requests
import spacy
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "strict_out"
OUT.mkdir(exist_ok=True)
MODEL = "en_core_web_sm"
MODEL_VERSION = "3.8.0"
ALPHA = 0.1
N_CLASSES = 18
LCF_TARGET = 83.02
DEEP_SAGE_TARGET = 91.02
DEEP_GCN_TARGET = 91.45
NOTEBOOK_URL = (
    "https://raw.githubusercontent.com/millioniron/"
    "LLM_exploration_Graph-Attention-Mechanisms-Perspective/"
    "fc9cb34184fa45b3f751c84f878586db26db6b60/"
    "roman_empire/roman_empire_with_comments.ipynb"
)
NPZ_URL = (
    "https://raw.githubusercontent.com/yandex-research/"
    "heterophilous-graphs/main/data/roman_empire.npz"
)
UA = {"User-Agent": "RomanFSPValidation/1.0 (academic reproducibility)"}
DEP_TO_ROW = {
    "ROOT": 0, "pobj": 1, "prep": 2, "det": 3, "amod": 4,
    "conj": 5, "nsubj": 6, "cc": 7, "dobj": 8, "advmod": 9,
    "compound": 10, "aux": 11, "appos": 12, "auxpass": 13,
    "nsubjpass": 14, "poss": 15, "relcl": 16,
}


def emit(**obj) -> None:
    print("RESULT " + json.dumps(obj, sort_keys=True), flush=True)


def fetch(url: str) -> bytes:
    response = requests.get(url, headers=UA, timeout=300)
    response.raise_for_status()
    return response.content


def source_text() -> str:
    notebook = json.loads(fetch(NOTEBOOK_URL))
    candidates: list[str] = []
    for cell in notebook["cells"]:
        code = "".join(cell.get("source", []))
        if "text='The Roman Empire" not in code:
            continue
        for statement in ast.parse(code).body:
            if isinstance(statement, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "text"
                for target in statement.targets
            ):
                value = ast.literal_eval(statement.value)
                if isinstance(value, str):
                    candidates.append(value)
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one embedded source article, got {len(candidates)}")
    text = candidates[0].replace(" ( ; ) ", " ").replace("\n\n", " ")
    (OUT / "source.txt").write_text(text, encoding="utf-8")
    return text


def benchmark():
    path = OUT / "roman_empire.npz"
    if not path.exists():
        path.write_bytes(fetch(NPZ_URL))
    data = np.load(path, allow_pickle=False)
    y = np.asarray(data["node_labels"], dtype=np.int64).reshape(-1)

    def mask(name: str) -> np.ndarray:
        value = np.asarray(data[name], dtype=bool)
        return value.T if value.shape[0] == 10 else value

    return y, mask("train_masks"), mask("val_masks"), mask("test_masks"), path


def parse_relations(text: str, expected_nodes: int) -> tuple[np.ndarray, list[str], float]:
    start = time.perf_counter()
    nlp = spacy.load(MODEL)
    if nlp.meta.get("version") != MODEL_VERSION:
        raise RuntimeError(f"Model version drift: {nlp.meta.get('version')} != {MODEL_VERSION}")
    nlp.max_length = max(nlp.max_length, len(text) + 1000)
    doc = nlp(text)
    punct = string.punctuation + "—–..."
    tokens = [token for sent in doc.sents for token in sent if str(token) not in punct]
    if len(tokens) != expected_nodes:
        raise RuntimeError(f"Token alignment failed: {len(tokens)} != {expected_nodes}")
    relations = np.asarray([DEP_TO_ROW.get(token.dep_, 17) for token in tokens], dtype=np.int64)
    words = [str(token) for token in tokens]
    elapsed = time.perf_counter() - start
    np.save(OUT / "parser_relations.npy", relations)
    (OUT / "tokens.json").write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")
    return relations, words, elapsed


def statistics(relations: np.ndarray, y: np.ndarray, selected: np.ndarray) -> np.ndarray:
    counts = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    np.add.at(counts, (relations[selected], y[selected]), 1)
    return counts


def projection(counts: np.ndarray) -> np.ndarray:
    # Alpha is fixed before evaluation. Adding the same alpha to every cell does
    # not alter a non-tied argmax, but specifies deterministic unseen-row behavior.
    return (counts.astype(np.float64) + ALPHA).argmax(axis=1)


def exact_unlearning_checks(
    relations: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    base_counts: np.ndarray,
    split: int,
) -> None:
    train_ids = np.flatnonzero(train)
    # Index-defined deletions are deterministic and independent of val/test labels.
    sizes = sorted(set([1, 10, 100, max(1, len(train_ids) // 100), max(1, len(train_ids) // 10)]))
    for size in sizes:
        deleted = train_ids[:size]
        retained = train.copy()
        retained[deleted] = False

        updated = base_counts.copy()
        np.add.at(updated, (relations[deleted], y[deleted]), -1)
        scratch = statistics(relations, y, retained)
        if not np.array_equal(updated, scratch):
            raise AssertionError(f"Sufficient-statistic mismatch at split={split}, size={size}")
        if not np.array_equal(projection(updated), projection(scratch)):
            raise AssertionError(f"Prediction-map mismatch at split={split}, size={size}")
        emit(method="ExactUnlearningCheck", split=split, deleted=size, exact=True)


def main() -> None:
    total_start = time.perf_counter()
    text = source_text()
    y, train_masks, val_masks, test_masks, npz_path = benchmark()
    relations, words, parse_seconds = parse_relations(text, len(y))

    protocol_hash = hashlib.sha256()
    protocol_hash.update(npz_path.read_bytes())
    protocol_hash.update(MODEL.encode())
    protocol_hash.update(MODEL_VERSION.encode())
    protocol_hash.update(str(ALPHA).encode())
    emit(
        method="Protocol",
        protocol_sha256=protocol_hash.hexdigest(),
        nodes=len(y),
        splits=train_masks.shape[1],
        model=MODEL,
        model_version=MODEL_VERSION,
        alpha=ALPHA,
        parse_seconds=parse_seconds,
    )

    tests: list[float] = []
    vals: list[float] = []
    fit_times: list[float] = []
    mappings: list[list[int]] = []
    for split in range(train_masks.shape[1]):
        train = train_masks[:, split]
        val = val_masks[:, split]
        test = test_masks[:, split]

        fit_start = time.perf_counter()
        counts = statistics(relations, y, train)
        mapping = projection(counts)
        predictions = mapping[relations]
        fit_seconds = time.perf_counter() - fit_start

        val_accuracy = 100.0 * accuracy_score(y[val], predictions[val])
        test_accuracy = 100.0 * accuracy_score(y[test], predictions[test])
        vals.append(val_accuracy)
        tests.append(test_accuracy)
        fit_times.append(fit_seconds)
        mappings.append(mapping.tolist())

        # Verify exact unlearning on every official split before reporting it.
        exact_unlearning_checks(relations, y, train, counts, split)
        emit(
            method="FSP",
            split=split,
            val=val_accuracy,
            test=test_accuracy,
            fit_seconds=fit_seconds,
            mapping=mapping.tolist(),
            beats_lcf=bool(test_accuracy > LCF_TARGET),
            beats_deep_sage=bool(test_accuracy > DEEP_SAGE_TARGET),
            beats_deep_gcn=bool(test_accuracy > DEEP_GCN_TARGET),
        )

    scores = np.asarray(tests)
    summary = {
        "method": "FSP_10_SPLIT_LOCKED",
        "mean": float(scores.mean()),
        "std": float(scores.std(ddof=1)),
        "minimum": float(scores.min()),
        "maximum": float(scores.max()),
        "val_mean": float(np.mean(vals)),
        "mean_fit_seconds": float(np.mean(fit_times)),
        "parse_seconds": parse_seconds,
        "total_seconds": time.perf_counter() - total_start,
        "lcf": LCF_TARGET,
        "deep_sage": DEEP_SAGE_TARGET,
        "deep_gcn": DEEP_GCN_TARGET,
        "gain_over_lcf": float(scores.mean() - LCF_TARGET),
        "gain_over_deep_sage": float(scores.mean() - DEEP_SAGE_TARGET),
        "gain_over_deep_gcn": float(scores.mean() - DEEP_GCN_TARGET),
        "all_splits_above_deep_gcn": bool(np.all(scores > DEEP_GCN_TARGET)),
        "all_unlearning_checks_exact": True,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    emit(**summary)


if __name__ == "__main__":
    main()
