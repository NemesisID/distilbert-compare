"""Pencarian retrieval: dense (cosine), BM25, dan fusi Reciprocal Rank Fusion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from rag.chunking import Chunk
from rag.index import Embedder, RagIndex
from rag.textnorm import get_tokenizer


@dataclass
class Retrieved:
    """Hasil satu chunk yang terambil."""

    chunk: Chunk
    score: float
    dense_score: float | None = None
    bm25_score: float | None = None


def dense_scores(query_emb: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """Cosine similarity (vektor sudah ternormalisasi) antara query dan chunk."""
    return np.asarray(embeddings @ np.asarray(query_emb), dtype=float).reshape(-1)


def bm25_scores(bm25: BM25Okapi, query_tokens: list[str]) -> np.ndarray:
    """Skor BM25 untuk satu query."""
    return np.asarray(bm25.get_scores(query_tokens), dtype=float)


def rank_desc(scores: np.ndarray, pool: int) -> list[int]:
    """Indeks chunk terurut skor menurun, ambil ``pool`` teratas."""
    order = np.argsort(-np.asarray(scores, dtype=float))
    return [int(i) for i in order[: int(pool)]]


def rrf_fuse(rankings: list[list[int]], k: int) -> dict[int, float]:
    """Gabungkan beberapa peringkat dengan Reciprocal Rank Fusion."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (int(k) + rank + 1)
    return fused


def retrieve(
    query: str,
    index: RagIndex,
    *,
    embedder: Embedder | None = None,
    method: str = "rrf",
    top_k: int = 3,
    pool: int = 20,
    rrf_k: int = 60,
    bm25: BM25Okapi | None = None,
) -> list[Retrieved]:
    """Ambil ``top_k`` chunk teratas untuk ``query`` dengan metode terpilih.

    ``method`` salah satu dari: ``"dense"``, ``"bm25"``, ``"rrf"``.
    """
    if method not in {"dense", "bm25", "rrf"}:
        raise ValueError(f"method tidak dikenal: {method!r}")

    tokenizer = get_tokenizer(index.tokenizer_name)
    query_tokens = tokenizer(query)
    if bm25 is None:
        bm25 = index.bm25()
    b_scores = bm25_scores(bm25, query_tokens)

    d_scores: np.ndarray | None = None
    if method in {"dense", "rrf"}:
        if index.embeddings is None or embedder is None:
            raise ValueError("Metode dense/rrf membutuhkan embeddings dan embedder.")
        query_emb = embedder.encode([query])[0]
        d_scores = dense_scores(query_emb, index.embeddings)

    if method == "dense":
        order = rank_desc(d_scores, top_k)
        return [
            Retrieved(index.chunks[i], float(d_scores[i]), float(d_scores[i]), float(b_scores[i]))
            for i in order
        ]

    if method == "bm25":
        order = rank_desc(b_scores, top_k)
        return [
            Retrieved(index.chunks[i], float(b_scores[i]), None, float(b_scores[i]))
            for i in order
        ]

    # rrf
    dense_rank = rank_desc(d_scores, pool)
    bm25_rank = rank_desc(b_scores, pool)
    fused = rrf_fuse([dense_rank, bm25_rank], rrf_k)
    order = sorted(fused, key=lambda i: (-fused[i], i))[: int(top_k)]
    return [
        Retrieved(index.chunks[i], float(fused[i]), float(d_scores[i]), float(b_scores[i]))
        for i in order
    ]
