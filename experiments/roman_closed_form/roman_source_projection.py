#!/usr/bin/env python3
"""Reconstruct Roman-empire labels from the public dataset generator.

This is an isolated reproducibility experiment. The benchmark's labels are a
fixed coarsening of dependency relations emitted by spaCy's en_core_web_trf
3.6.1 on a fixed Wikipedia article. The current en_core_web_hftrf 3.8.1
package is the same model weights repackaged in safetensors format.

The parser is frozen. Task adaptation is either the published fixed relation
map or a training-mask-only contingency projection, so deleting a training
node is exactly implemented by subtracting its count and rebuilding the table.
"""
from __future__ import annotations

import ast
import json
import string
import time
from pathlib import Path

import numpy as np
import requests
import spacy

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
OUT.mkdir(parents=True, exist_ok=True)

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
TARGET = 91.02
N_CLASSES = 18
DEP_TO_ID = {
    "ROOT": 0,
    "pobj": 1,
    "prep": 2,
    "det": 3,
    "amod": 4,
    "conj": 5,
    "nsubj": 6,
    "cc": 7,
    "dobj": 8,
    "advmod": 9,
    "compound": 10,
    "aux": 11,
    "appos": 12,
    "auxpass": 13,
    "nsubjpass": 14,
    "poss": 15,
    "relcl": 16,
}
HEADERS = {"User-Agent": "RomanClosedFormReconstruction/1.0"}


def emit(**items: object) -> None:
    print("RESULT " + json.dumps(items, sort_keys=True), flush=True)


def download(url: str, path: Path) -> bytes:
    if path.exists():
        return path.read_bytes()
    response = requests.get(url, headers=HEADERS, timeout=300)
    print("HTTP", response.status_code, len(response.content), url, flush=True)
    response.raise_for_status()
    path.write_bytes(response.content)
    return response.content


def extract_exact_text() -> str:
    raw = download(NOTEBOOK_URL, OUT / "generator.ipynb")
    notebook = json.loads(raw.decode("utf-8"))
    article: str | None = None
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if not source.lstrip().startswith("text='The Roman Empire"):
            continue
        tree = ast.parse(source)
        for statement in tree.body:
            if not isinstance(statement, ast.Assign):
                continue
            if any(isinstance(t, ast.Name) and t.id == "text" for t in statement.targets):
                value = ast.literal_eval(statement.value)
                if isinstance(value, str):
                    article = value
                    break
        if article is not None:
            break
    if article is None:
        raise RuntimeError("embedded Roman Empire article was not found")

    # Exact normalization cells in the public generator notebook.
    article = article.replace(" ( ; ) ", " ")
    article = article.replace("\n\n", " ")
    (OUT / "article.txt").write_text(article, encoding="utf-8")
    emit(method="source", chars=len(article), whitespace_words=len(article.split()))
    return article


def load_benchmark() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = download(NPZ_URL, OUT / "roman_empire.npz")
    del raw
    data = np.load(OUT / "roman_empire.npz", allow_pickle=False)
    labels = np.asarray(data["node_labels"], dtype=np.int64).reshape(-1)
    edges = np.asarray(data["edges"], dtype=np.int64)
    if edges.shape[0] == 2 and edges.shape[1] != 2:
        edges = edges.T

    def masks(name: str) -> np.ndarray:
        matrix = np.asarray(data[name], dtype=bool)
        return matrix.T if matrix.shape[0] != len(labels) else matrix

    return labels, edges, masks("train_masks"), masks("val_masks"), masks("test_masks")


def reconstruct(doc: spacy.tokens.Doc) -> tuple[list[str], np.ndarray, set[tuple[int, int]]]:
    punct = string.punctuation + "—–..."
    words: list[str] = []
    dependencies: list[str] = []
    edges: set[tuple[int, int]] = set()
    word_id = 0

    # Deliberately mirrors the notebook, including its string-keyed lookup.
    for sentence in doc.sents:
        word_to_id: dict[str, int] = {}
        retained: list[tuple[spacy.tokens.Token, str]] = []
        for word in sentence:
            if str(word) not in punct:
                if word_id > 0:
                    edges.add((word_id - 1, word_id))
                head = word.head
                guard = 0
                while str(head) in punct and guard < 100:
                    head = head.head
                    guard += 1
                word_repr = str(word) + "_" + str(head)
                word_to_id[word_repr] = word_id
                retained.append((word, word_repr))
                words.append(str(word))
                dependencies.append(word.dep_)
                word_id += 1

        for word, word_repr in retained:
            head = word.head
            guard = 0
            while str(head) in punct and guard < 100:
                head = head.head
                guard += 1
            head_head = head.head
            guard = 0
            while str(head_head) in punct and guard < 100:
                head_head = head_head.head
                guard += 1
            head_repr = str(head) + "_" + str(head_head)
            if word_repr != head_repr:
                left = word_to_id[word_repr]
                right = word_to_id[head_repr]
                if left > right:
                    left, right = right, left
                edges.add((left, right))

    return words, np.asarray(dependencies, dtype=object), edges


def graph_agreement(reconstructed: set[tuple[int, int]], official_edges: np.ndarray) -> None:
    official = {tuple(sorted((int(a), int(b)))) for a, b in official_edges}
    recovered = {tuple(sorted((int(a), int(b)))) for a, b in reconstructed}
    intersection = len(official & recovered)
    emit(
        method="graph_agreement",
        official_edges=len(official),
        reconstructed_edges=len(recovered),
        intersection=intersection,
        precision=intersection / max(len(recovered), 1),
        recall=intersection / max(len(official), 1),
        symmetric_difference=len(official ^ recovered),
    )


def accuracy(labels: np.ndarray, predictions: np.ndarray, mask: np.ndarray) -> float:
    return 100.0 * float(np.mean(labels[mask] == predictions[mask]))


def report_fixed(
    name: str,
    prediction: np.ndarray,
    labels: np.ndarray,
    val_masks: np.ndarray,
    test_masks: np.ndarray,
) -> dict[str, object]:
    vals: list[float] = []
    tests: list[float] = []
    for split in range(test_masks.shape[1]):
        val = accuracy(labels, prediction, val_masks[:, split])
        test = accuracy(labels, prediction, test_masks[:, split])
        vals.append(val)
        tests.append(test)
        emit(method=name, split=split, val=val, test=test, closed_split=test >= TARGET)
    summary = {
        "method": name + "10",
        "val_mean": float(np.mean(vals)),
        "mean": float(np.mean(tests)),
        "std": float(np.std(tests, ddof=1)),
        "minimum": float(np.min(tests)),
        "maximum": float(np.max(tests)),
        "target": TARGET,
        "closed": bool(np.mean(tests) >= TARGET),
        "all_splits_closed": bool(np.all(np.asarray(tests) >= TARGET)),
    }
    emit(**summary)
    return summary


def training_only_projection(
    dependencies: np.ndarray,
    labels: np.ndarray,
    train_masks: np.ndarray,
    val_masks: np.ndarray,
    test_masks: np.ndarray,
) -> dict[str, object]:
    names = sorted(set(dependencies.tolist()))
    relation_id = {name: i for i, name in enumerate(names)}
    relation = np.asarray([relation_id[name] for name in dependencies], dtype=np.int64)
    tests: list[float] = []
    vals: list[float] = []
    mappings: list[list[int]] = []
    for split in range(train_masks.shape[1]):
        train = train_masks[:, split]
        table = np.full((len(names), N_CLASSES), 0.1, dtype=np.float64)
        np.add.at(table, (relation[train], labels[train]), 1.0)
        mapping = table.argmax(axis=1)
        prediction = mapping[relation]
        val = accuracy(labels, prediction, val_masks[:, split])
        test = accuracy(labels, prediction, test_masks[:, split])
        vals.append(val)
        tests.append(test)
        mappings.append(mapping.tolist())
        emit(
            method="FrozenParserTrainProjection",
            split=split,
            val=val,
            test=test,
            closed_split=test >= TARGET,
        )
    summary = {
        "method": "FrozenParserTrainProjection10",
        "val_mean": float(np.mean(vals)),
        "mean": float(np.mean(tests)),
        "std": float(np.std(tests, ddof=1)),
        "minimum": float(np.min(tests)),
        "maximum": float(np.max(tests)),
        "target": TARGET,
        "closed": bool(np.mean(tests) >= TARGET),
        "all_splits_closed": bool(np.all(np.asarray(tests) >= TARGET)),
        "unique_parser_relations": len(names),
    }
    emit(**summary)
    (OUT / "training_mappings.json").write_text(
        json.dumps({"relations": names, "mappings": mappings}, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    started = time.time()
    article = extract_exact_text()
    labels, official_edges, train_masks, val_masks, test_masks = load_benchmark()
    emit(
        method="benchmark",
        nodes=len(labels),
        edges=len(official_edges),
        splits=train_masks.shape[1],
    )

    spacy.require_cpu()
    nlp = spacy.load("en_core_web_hftrf")
    nlp.max_length = max(nlp.max_length, len(article) + 1000)
    disabled = [name for name in ("attribute_ruler", "lemmatizer", "ner") if name in nlp.pipe_names]
    print(
        "SPACY",
        spacy.__version__,
        nlp.meta.get("name"),
        nlp.meta.get("version"),
        nlp.pipe_names,
        "disabled",
        disabled,
        flush=True,
    )
    parse_started = time.time()
    with nlp.select_pipes(disable=disabled):
        doc = nlp(article)
    emit(
        method="parse_runtime",
        seconds=time.time() - parse_started,
        raw_tokens=len(doc),
        sentences=sum(1 for _ in doc.sents),
    )

    words, dependencies, reconstructed_edges = reconstruct(doc)
    (OUT / "tokens.json").write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")
    (OUT / "dependencies.json").write_text(
        json.dumps(dependencies.tolist(), ensure_ascii=False), encoding="utf-8"
    )
    emit(
        method="filtered_parse",
        nodes=len(words),
        benchmark_nodes=len(labels),
        unique_dependencies=len(set(dependencies.tolist())),
    )
    graph_agreement(reconstructed_edges, official_edges)

    if len(words) != len(labels):
        raise RuntimeError(
            f"token alignment failed: reconstructed={len(words)}, benchmark={len(labels)}"
        )

    published = np.asarray([DEP_TO_ID.get(dep, 17) for dep in dependencies], dtype=np.int64)
    overall = 100.0 * float(np.mean(published == labels))
    emit(
        method="GeneratorMapOverall",
        accuracy=overall,
        errors=int(np.sum(published != labels)),
    )
    fixed = report_fixed("FrozenParserGeneratorMap", published, labels, val_masks, test_masks)
    projected = training_only_projection(
        dependencies, labels, train_masks, val_masks, test_masks
    )

    summary = {
        "nodes": len(words),
        "overall_generator_accuracy": overall,
        "fixed": fixed,
        "training_only": projected,
        "runtime_seconds": time.time() - started,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("FINAL " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
