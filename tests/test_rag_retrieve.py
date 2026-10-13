"""Uji retrieval dense/BM25/RRF (``rag/retrieve.py``) dengan embedder palsu."""

from __future__ import annotations

import numpy as np
import pytest

from rag import retrieve as R
from rag.chunking import Chunk
from rag.index import RagIndex


class FakeEmbedder:
    """Embedder deterministik: peta teks -> vektor tetap (tanpa model/jaringan)."""

    def __init__(self, mapping: dict[str, list[float]]):
        self.mapping = mapping

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray([self.mapping[t] for t in texts], dtype=np.float32)


def _index() -> RagIndex:
    chunks = [
        Chunk(0, "a.pdf", 1, "syarat pendaftaran mahasiswa aktif", 0, 5),
        Chunk(1, "a.pdf", 2, "alur seleksi dan verifikasi berkas", 0, 5),
        Chunk(2, "b.pdf", 1, "jadwal pendampingan usaha bulanan", 0, 5),
    ]
    emb = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    return RagIndex(chunks, "lowercase_space", embeddings=emb, embedding_model="fake")


def test_rank_desc_top_pool():
    assert R.rank_desc(np.asarray([0.1, 0.9, 0.5]), pool=2) == [1, 2]


def test_rrf_fuse_orders_by_combined_rank():
    fused = R.rrf_fuse([[0, 1], [1, 2]], k=60)
    order = sorted(fused, key=lambda i: (-fused[i], i))
    assert order == [1, 0, 2]


def test_dense_scores_is_cosine_for_normalized_rows():
    emb = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    q = np.asarray([1.0, 0.0], dtype=np.float32)
    assert R.dense_scores(q, emb).tolist() == [1.0, 0.0]


def test_retrieve_dense_picks_matching_chunk():
    idx = _index()
    emb = FakeEmbedder({"syarat": [1.0, 0.0, 0.0]})
    out = R.retrieve("syarat", idx, embedder=emb, method="dense", top_k=2)
    assert out[0].chunk.chunk_id == 0
    assert out[0].dense_score == pytest.approx(1.0)


def test_retrieve_bm25_picks_matching_chunk():
    idx = _index()
    out = R.retrieve("pendaftaran", idx, method="bm25", top_k=1)
    assert out[0].chunk.chunk_id == 0


def test_retrieve_rrf_combines_both_signals():
    idx = _index()
    emb = FakeEmbedder({"pendaftaran": [1.0, 0.0, 0.0]})
    out = R.retrieve("pendaftaran", idx, embedder=emb, method="rrf", top_k=3, pool=3, rrf_k=60)
    assert out[0].chunk.chunk_id == 0
    assert out[0].dense_score is not None and out[0].bm25_score is not None


def test_retrieve_dense_requires_embeddings():
    idx = _index()
    idx.embeddings = None
    with pytest.raises(ValueError):
        R.retrieve("apa", idx, embedder=FakeEmbedder({"apa": [1.0, 0.0, 0.0]}), method="dense")


def test_retrieve_unknown_method_raises():
    idx = _index()
    with pytest.raises(ValueError):
        R.retrieve("apa", idx, method="tidak_ada")
