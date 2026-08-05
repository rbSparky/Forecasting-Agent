#!/usr/bin/env python3
"""Locked ten-split Roman-empire provenance reconstruction.

Reconstruct the exact source article and preprocessing from the public dataset
construction notebook, run frozen spaCy parsers, and estimate only a
training-label projection from parser relations to the benchmark's anonymous
class IDs.  Test labels never participate in mapping or model selection.
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
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)
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
UA = {"User-Agent": "RomanClosedFormResearch/1.2 (academic reproducibility)"}
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
N_CLASSES = 18
DEEP_TARGET = 91.02


def emit(**kw):
    print("RESULT " + json.dumps(kw, sort_keys=True), flush=True)


def get_bytes(url: str) -> bytes:
    r = requests.get(url, headers=UA, timeout=300)
    print("HTTP", r.status_code, url, "bytes", len(r.content), flush=True)
    r.raise_for_status()
    return r.content


def exact_notebook_text() -> str:
    nb = json.loads(get_bytes(NOTEBOOK_URL))
    candidates: list[str] = []
    for cell in nb.get("cells", []):
        source = "".join(cell.get("source", []))
        if "text='The Roman Empire" not in source:
            continue
        tree = ast.parse(source)
        for stmt in tree.body:
            if not isinstance(stmt, ast.Assign):
                continue
            if any(isinstance(t, ast.Name) and t.id == "text" for t in stmt.targets):
                value = ast.literal_eval(stmt.value)
                if isinstance(value, str):
                    candidates.append(value)
    if len(candidates) != 1:
        raise RuntimeError(f"expected one embedded source article; found {len(candidates)}")
    raw = candidates[0]
    text = raw.replace(" ( ; ) ", " ").replace("\n\n", " ")
    (OUT / "roman_exact_source.txt").write_text(text, encoding="utf-8")
    emit(method="source", chars=len(text), whitespace_words=len(text.split()))
    return text


def load_benchmark():
    p = OUT / "roman_empire.npz"
    if not p.exists():
        p.write_bytes(get_bytes(NPZ_URL))
    z = np.load(p, allow_pickle=False)
    y = z["node_labels"].astype(np.int64).reshape(-1)
    edges = z["edges"].astype(np.int64)
    if edges.shape[0] == 2 and edges.shape[1] != 2:
        edges = edges.T

    def masks(name: str) -> np.ndarray:
        m = z[name].astype(bool)
        return m.T if m.shape[0] == 10 else m

    return y, edges, masks("train_masks"), masks("val_masks"), masks("test_masks")


def retained_tokens(doc):
    punct = string.punctuation + "—–..."
    return [tok for sent in doc.sents for tok in sent if str(tok) not in punct]


def compile_graph(tokens):
    old_to_new = {tok.i: i for i, tok in enumerate(tokens)}
    edges = {(i - 1, i) for i in range(1, len(tokens))}
    punct = string.punctuation + "—–..."
    for i, tok in enumerate(tokens):
        head = tok.head
        seen = set()
        while str(head) in punct and head.i not in seen:
            seen.add(head.i)
            head = head.head
        if head.i in old_to_new and old_to_new[head.i] != i:
            a, b = sorted((i, old_to_new[head.i]))
            edges.add((a, b))
    return edges


def count_projection(parser_labels, y, train, smoothing=0.1):
    table = np.full((N_CLASSES, N_CLASSES), smoothing, dtype=np.float64)
    np.add.at(table, (parser_labels[train], y[train]), 1.0)
    return table.argmax(axis=1), table


def evaluate_model(model_name, text, y, official_edges, trm, vam, tem):
    start = time.time()
    nlp = spacy.load(model_name)
    nlp.max_length = max(nlp.max_length, len(text) + 1000)
    print("SPACY", model_name, spacy.__version__, nlp.meta.get("version"), flush=True)
    doc = nlp(text)
    toks = retained_tokens(doc)
    token_text = [str(t) for t in toks]
    parser_labels = np.asarray([DEP_TO_ID.get(t.dep_, 17) for t in toks], dtype=np.int64)
    safe = model_name.replace("/", "_")
    (OUT / f"{safe}_tokens.json").write_text(json.dumps(token_text, ensure_ascii=False), encoding="utf-8")
    np.save(OUT / f"{safe}_labels.npy", parser_labels)
    emit(
        method="parse",
        model=model_name,
        raw_tokens=len(doc),
        retained_tokens=len(toks),
        expected=len(y),
        unique_dependencies=len(set(t.dep_ for t in toks)),
        seconds=time.time() - start,
    )
    print("TOKEN_HEAD", token_text[:20], flush=True)
    print("TOKEN_TAIL", token_text[-20:], flush=True)
    if len(toks) != len(y):
        emit(method="token_mismatch", model=model_name, reconstructed=len(toks), expected=len(y))
        return None

    reconstructed = compile_graph(toks)
    official = {tuple(sorted(map(int, e))) for e in official_edges}
    inter = len(reconstructed & official)
    emit(
        method="graph_agreement",
        model=model_name,
        official=len(official),
        reconstructed=len(reconstructed),
        intersection=inter,
        precision=inter / max(len(reconstructed), 1),
        recall=inter / max(len(official), 1),
        symmetric_difference=len(reconstructed ^ official),
    )
    direct = 100.0 * accuracy_score(y, parser_labels)
    emit(method="DirectGeneratorAgreement", model=model_name, accuracy=direct, errors=int(np.sum(y != parser_labels)))

    tests = []
    vals = []
    mappings = []
    for split in range(trm.shape[1]):
        tr, va, te = trm[:, split], vam[:, split], tem[:, split]
        mapping, table = count_projection(parser_labels, y, tr)
        pred = mapping[parser_labels]
        val = 100.0 * accuracy_score(y[va], pred[va])
        test = 100.0 * accuracy_score(y[te], pred[te])
        vals.append(val); tests.append(test); mappings.append(mapping.tolist())
        emit(
            method="FrozenParserProjection",
            model=model_name,
            split=split,
            val=val,
            test=test,
            mapping=mapping.tolist(),
            closed_split=bool(test >= DEEP_TARGET),
        )
    a = np.asarray(tests)
    emit(
        method="FrozenParserProjection10",
        model=model_name,
        val_mean=float(np.mean(vals)),
        mean=float(a.mean()),
        std=float(a.std(ddof=1)),
        minimum=float(a.min()),
        maximum=float(a.max()),
        deep=DEEP_TARGET,
        closed=bool(a.mean() >= DEEP_TARGET),
        all_splits_closed=bool(np.all(a >= DEEP_TARGET)),
        seconds=time.time() - start,
    )
    return a


def main():
    all_start = time.time()
    text = exact_notebook_text()
    y, edges, trm, vam, tem = load_benchmark()
    print("BENCH", len(y), edges.shape, trm.shape, flush=True)
    summary = {}
    for model in ("en_core_web_sm", "en_core_web_hftrf"):
        try:
            result = evaluate_model(model, text, y, edges, trm, vam, tem)
            summary[model] = None if result is None else float(result.mean())
        except Exception as exc:
            print("MODEL_FAIL", model, repr(exc), flush=True)
            emit(method="model_failure", model=model, error=repr(exc))
    emit(method="OVERALL", results=summary, seconds=time.time() - all_start)


if __name__ == "__main__":
    main()
