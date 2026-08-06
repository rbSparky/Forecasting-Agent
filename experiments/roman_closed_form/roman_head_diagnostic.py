#!/usr/bin/env python3
"""Archive frozen-parser head indices and sentence IDs for structural diagnostics.

This file is not part of the proposed standard-input method. It records the
benchmark generator's latent directed tree only to measure how much of the
remaining error is attributable to head/orientation recovery.
"""
from __future__ import annotations

import json
import os
import string

import numpy as np
import spacy

import roman_source_projection as base


def main() -> None:
    article = base.extract_exact_text()
    requested = os.environ.get("ROMAN_SPACY_MODEL", "en_core_web_trf")
    nlp = spacy.load(requested)
    nlp.max_length = max(nlp.max_length, len(article) + 1000)
    disabled = [name for name in ("attribute_ruler", "lemmatizer", "ner") if name in nlp.pipe_names]
    with nlp.select_pipes(disable=disabled):
        doc = nlp(article)

    punct = string.punctuation + "—–..."
    retained = [tok for sent in doc.sents for tok in sent if str(tok) not in punct]
    old_to_new = {tok.i: i for i, tok in enumerate(retained)}
    heads = np.full(len(retained), -1, dtype=np.int64)
    sent_ids = np.full(len(retained), -1, dtype=np.int64)
    sent_bounds: list[list[int]] = []

    cursor = 0
    for sid, sent in enumerate(doc.sents):
        kept = [tok for tok in sent if str(tok) not in punct]
        start = cursor
        for tok in kept:
            i = old_to_new[tok.i]
            sent_ids[i] = sid
            head = tok.head
            seen: set[int] = set()
            while head.i not in old_to_new and head.i not in seen:
                seen.add(head.i)
                nxt = head.head
                if nxt.i == head.i:
                    break
                head = nxt
            if tok.dep_ != "ROOT" and head.i in old_to_new:
                heads[i] = old_to_new[head.i]
            cursor += 1
        sent_bounds.append([start, cursor])

    np.save(base.OUT / "head_indices.npy", heads)
    np.save(base.OUT / "sentence_ids.npy", sent_ids)
    (base.OUT / "sentence_bounds.json").write_text(json.dumps(sent_bounds), encoding="utf-8")
    directed = [[i, int(h)] for i, h in enumerate(heads) if h >= 0]
    (base.OUT / "directed_dependency_edges.json").write_text(json.dumps(directed), encoding="utf-8")
    base.emit(
        method="head_diagnostic",
        nodes=len(retained),
        heads=int(np.sum(heads >= 0)),
        roots=int(np.sum(heads < 0)),
        sentences=len(sent_bounds),
        max_sentence=max(b - a for a, b in sent_bounds),
    )


if __name__ == "__main__":
    main()
