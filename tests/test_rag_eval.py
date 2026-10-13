"""Uji evaluasi retrieval & pemilihan tau2 (``rag/eval_retrieval.py``)."""

from __future__ import annotations

import numpy as np
import pytest

from rag import eval_retrieval as E
from rag.chunking import Chunk
from rag.index import RagIndex
from rag.retrieve import Retrieved


def _chunk(cid, doc, page):
    return Chunk(cid, doc, page, f"teks {cid}", 0, 1)


def test_hit_at_k_and_reciprocal_rank():
    results = [
        Retrieved(_chunk(0, "a.pdf", 1), 0.9),
        Retrieved(_chunk(1, "b.pdf", 2), 0.8),
    ]
    assert E.hit_at_k(results, "b.pdf", 2, k=2) is True
    assert E.hit_at_k(results, "b.pdf", 2, k=1) is False
    assert E.reciprocal_rank(results, "b.pdf", 2) == pytest.approx(0.5)
    assert E.reciprocal_rank(results, "z.pdf", 9) == pytest.approx(0.0)


def test_tau2_values_grid():
    assert E.tau2_values(0.3, 0.5, 0.05) == [0.3, 0.35, 0.4, 0.45, 0.5]


def test_metrics_at_tau2_counts():
    m = E.metrics_at_tau2([0.9, 0.7, 0.2], [0.8, 0.1], 0.6)
    assert m["in_doc_recall"] == pytest.approx(2 / 3)
    assert m["out_doc_false_accept"] == pytest.approx(0.5)
    assert m["n_in_doc"] == 3 and m["n_out_doc"] == 2


def test_select_tau2_ok_smallest_eligible():
    table = [
        {"tau": 0.5, "in_doc_recall": 0.8, "out_doc_false_accept": 0.0,
         "n_in_doc": 4, "n_out_doc": 2},
        {"tau": 0.7, "in_doc_recall": 0.95, "out_doc_false_accept": 0.0,
         "n_in_doc": 4, "n_out_doc": 2},
    ]
    chosen, status = E.select_tau2(table, recall_min=0.9, false_accept_max=0.05)
    assert status == "ok"
    assert chosen["tau"] == pytest.approx(0.7)


def test_select_tau2_relaxed_when_recall_unreachable():
    table = [
        {"tau": 0.5, "in_doc_recall": 0.5, "out_doc_false_accept": 0.0,
         "n_in_doc": 4, "n_out_doc": 2},
        {"tau": 0.8, "in_doc_recall": 0.7, "out_doc_false_accept": 0.0,
         "n_in_doc": 4, "n_out_doc": 2},
    ]
    chosen, status = E.select_tau2(table, recall_min=0.99, false_accept_max=0.05)
    assert status == "relaxed"
    assert chosen["tau"] == pytest.approx(0.8)  # recall tertinggi


def test_select_tau2_fallback_min_false_accept():
    table = [
        {"tau": 0.3, "in_doc_recall": 1.0, "out_doc_false_accept": 0.5,
         "n_in_doc": 4, "n_out_doc": 2},
        {"tau": 0.9, "in_doc_recall": 0.2, "out_doc_false_accept": 0.1,
         "n_in_doc": 4, "n_out_doc": 2},
    ]
    chosen, status = E.select_tau2(table, recall_min=0.99, false_accept_max=0.0)
    assert status == "relaxed"
    assert chosen["tau"] == pytest.approx(0.9)  # false-accept terkecil


class FakeEmbedder:
    def __init__(self, mapping):
        self.mapping = mapping

    def encode(self, texts):
        return np.asarray([self.mapping[t] for t in texts], dtype=np.float32)


def _index():
    chunks = [
        _chunk(0, "a.pdf", 1),
        _chunk(1, "a.pdf", 2),
    ]
    emb = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
    return RagIndex(chunks, "lowercase_space", embeddings=emb, embedding_model="fake")


def test_evaluate_qa_reports_recall_and_mrr():
    idx = _index()
    rows = [{"question": "q0", "doc": "a.pdf", "page": 1}]
    emb = FakeEmbedder({"q0": [1.0, 0.0]})
    metrics = E.evaluate_qa(
        rows, idx, embedder=emb, method="dense", top_k=2, pool=2, rrf_k=60
    )
    assert metrics["n"] == 1
    assert metrics["recall_at_k"][1] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(1.0)


def test_top1_similarity_requires_embeddings():
    idx = _index()
    idx.embeddings = None
    with pytest.raises(ValueError):
        E.top1_similarity("q", idx, FakeEmbedder({"q": [1.0, 0.0]}))
