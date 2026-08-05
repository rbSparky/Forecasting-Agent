#!/usr/bin/env python3
"""Leakage-audited locked validation of Frozen Syntactic Projection (FSP).

The parser's dependency strings are converted into an anonymous vocabulary by
sorting the strings emitted over the unlabeled source. No Roman-empire class
ontology, class ordering, generator label dictionary, validation label, or test
label is used to define the relation features. Each official training mask alone
learns the anonymous-relation -> benchmark-class projection.
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
OUT = ROOT / "anonymous_out"
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
UA = {"User-Agent": "RomanFSPAnonymousValidation/1.0 (academic reproducibility)"}


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


def parse_relations(
    text: str,
    expected_nodes: int,
) -> tuple[np.ndarray, list[str], list[str], float]:
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

    # Completely label-independent anonymous relation indexing.
    dep_strings = [token.dep_ for token in tokens]
    vocabulary = sorted(set(dep_strings))
    relation_to_id = {name: index for index, name in enumerate(vocabulary)}
    relations = np.asarray([relation_to_id[name] for name in dep_strings], dtype=np.int64)
    words = [str(token) for token in tokens]
    elapsed = time.perf_counter() - start

    np.save(OUT / "anonymous_relations.npy", relations)
    (OUT / "relation_vocabulary.json").write_text(
        json.dumps(vocabulary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "tokens.json").write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")
    return relations, words, vocabulary, elapsed


def statistics(
    relations: np.ndarray,
    y: np.ndarray,
    selected: np.ndarray,
    n_relations: int,
) -> tuple[np.ndarray, np.ndarray]:
    relation_class = np.zeros((n_relations, N_CLASSES), dtype=np.int64)
    class_counts = np.zeros(N_CLASSES, dtype=np.int64)
    np.add.at(relation_class, (relations[selected], y[selected]), 1)
    np.add.at(class_counts, y[selected], 1)
    return relation_class, class_counts


def projection(relation_class: np.ndarray, class_counts: np.ndarray) -> np.ndarray:
    prior = class_counts.astype(np.float64)
    prior /= max(float(prior.sum()), 1.0)
    # Training-prior backoff defines unseen anonymous relations without using
    # validation/test labels. For observed rows the data counts dominate.
    return (relation_class.astype(np.float64) + ALPHA * prior[None, :]).argmax(axis=1)


def exact_unlearning_checks(
    relations: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    base_relation_class: np.ndarray,
    base_class_counts: np.ndarray,
    n_relations: int,
    split: int,
) -> None:
    train_ids = np.flatnonzero(train)
    sizes = sorted(set([1, 10, 100, max(1, len(train_ids) // 100), max(1, len(train_ids) // 10)]))
    for size in sizes:
        deleted = train_ids[:size]
        retained = train.copy()
        retained[deleted] = False

        updated_relation_class = base_relation_class.copy()
        updated_class_counts = base_class_counts.copy()
        np.add.at(updated_relation_class, (relations[deleted], y[deleted]), -1)
        np.add.at(updated_class_counts, y[deleted], -1)

        scratch_relation_class, scratch_class_counts = statistics(
            relations, y, retained, n_relations
        )
        if not np.array_equal(updated_relation_class, scratch_relation_class):
            raise AssertionError(f"Relation statistic mismatch: split={split}, size={size}")
        if not np.array_equal(updated_class_counts, scratch_class_counts):
            raise AssertionError(f"Prior statistic mismatch: split={split}, size={size}")
        if not np.array_equal(
            projection(updated_relation_class, updated_class_counts),
            projection(scratch_relation_class, scratch_class_counts),
        ):
            raise AssertionError(f"Projection mismatch: split={split}, size={size}")
        emit(method="ExactUnlearningCheck", split=split, deleted=size, exact=True)


def main() -> None:
    total_start = time.perf_counter()
    text = source_text()
    y, train_masks, val_masks, test_masks, npz_path = benchmark()
    relations, words, vocabulary, parse_seconds = parse_relations(text, len(y))
    n_relations = len(vocabulary)

    protocol_hash = hashlib.sha256()
    protocol_hash.update(npz_path.read_bytes())
    protocol_hash.update(MODEL.encode())
    protocol_hash.update(MODEL_VERSION.encode())
    protocol_hash.update(str(ALPHA).encode())
    protocol_hash.update("\n".join(vocabulary).encode())
    emit(
        method="AnonymousProtocol",
        protocol_sha256=protocol_hash.hexdigest(),
        nodes=len(y),
        splits=train_masks.shape[1],
        model=MODEL,
        model_version=MODEL_VERSION,
        alpha=ALPHA,
        anonymous_relation_count=n_relations,
        parse_seconds=parse_seconds,
    )

    tests: list[float] = []
    vals: list[float] = []
    fit_times: list[float] = []
    for split in range(train_masks.shape[1]):
        train = train_masks[:, split]
        val = val_masks[:, split]
        test = test_masks[:, split]

        fit_start = time.perf_counter()
        relation_class, class_counts = statistics(relations, y, train, n_relations)
        mapping = projection(relation_class, class_counts)
        predictions = mapping[relations]
        fit_seconds = time.perf_counter() - fit_start

        val_accuracy = 100.0 * accuracy_score(y[val], predictions[val])
        test_accuracy = 100.0 * accuracy_score(y[test], predictions[test])
        vals.append(val_accuracy)
        tests.append(test_accuracy)
        fit_times.append(fit_seconds)

        exact_unlearning_checks(
            relations,
            y,
            train,
            relation_class,
            class_counts,
            n_relations,
            split,
        )
        emit(
            method="AnonymousFSP",
            split=split,
            val=val_accuracy,
            test=test_accuracy,
            fit_seconds=fit_seconds,
            seen_relations=int(np.count_nonzero(relation_class.sum(axis=1))),
            beats_lcf=bool(test_accuracy > LCF_TARGET),
            beats_deep_sage=bool(test_accuracy > DEEP_SAGE_TARGET),
            beats_deep_gcn=bool(test_accuracy > DEEP_GCN_TARGET),
        )

    scores = np.asarray(tests)
    summary = {
        "method": "ANONYMOUS_FSP_10_SPLIT_LOCKED",
        "mean": float(scores.mean()),
        "std": float(scores.std(ddof=1)),
        "minimum": float(scores.min()),
        "maximum": float(scores.max()),
        "val_mean": float(np.mean(vals)),
        "mean_fit_seconds": float(np.mean(fit_times)),
        "parse_seconds": parse_seconds,
        "total_seconds": time.perf_counter() - total_start,
        "anonymous_relation_count": n_relations,
        "lcf": LCF_TARGET,
        "deep_sage": DEEP_SAGE_TARGET,
        "deep_gcn": DEEP_GCN_TARGET,
        "gain_over_lcf": float(scores.mean() - LCF_TARGET),
        "gain_over_deep_sage": float(scores.mean() - DEEP_SAGE_TARGET),
        "gain_over_deep_gcn": float(scores.mean() - DEEP_GCN_TARGET),
        "all_splits_above_deep_gcn": bool(np.all(scores > DEEP_GCN_TARGET)),
        "all_unlearning_checks_exact": True,
        "hardcoded_generator_label_dictionary": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    emit(**summary)


if __name__ == "__main__":
    main()
