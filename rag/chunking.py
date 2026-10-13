"""Pemotongan teks menjadi chunk berbasis kata dengan tumpang tindih."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Chunk:
    """Satu potongan teks beserta metadata asal."""

    chunk_id: int
    doc: str
    page: int
    text: str
    start_word: int
    end_word: int


def chunk_text(text: str, chunk_words: int, overlap: int) -> list[tuple[str, int, int]]:
    """Potong ``text`` menjadi chunk ``chunk_words`` kata dengan ``overlap``.

    Mengembalikan daftar ``(teks_chunk, start_word, end_word)``.
    """
    if chunk_words <= 0:
        raise ValueError("chunk_words harus > 0")
    if overlap < 0 or overlap >= chunk_words:
        raise ValueError("overlap harus berada di rentang 0 <= overlap < chunk_words")

    words = str(text).split()
    if not words:
        return []

    step = chunk_words - overlap
    out: list[tuple[str, int, int]] = []
    start = 0
    n = len(words)
    while start < n:
        end = min(start + chunk_words, n)
        out.append((" ".join(words[start:end]), start, end))
        if end >= n:
            break
        start += step
    return out


def chunk_pages(pages: Iterable, chunk_words: int, overlap: int) -> list[Chunk]:
    """Potong hasil ekstraksi halaman (punya ``doc``, ``page``, ``text``)."""
    chunks: list[Chunk] = []
    cid = 0
    for page in pages:
        for text, start, end in chunk_text(page.text, chunk_words, overlap):
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    doc=str(page.doc),
                    page=int(page.page),
                    text=text,
                    start_word=start,
                    end_word=end,
                )
            )
            cid += 1
    return chunks


def chunks_to_records(chunks: Sequence[Chunk]) -> list[dict]:
    """Ubah chunk menjadi list dict (untuk serialisasi JSON)."""
    return [
        {
            "chunk_id": c.chunk_id,
            "doc": c.doc,
            "page": c.page,
            "text": c.text,
            "start_word": c.start_word,
            "end_word": c.end_word,
        }
        for c in chunks
    ]


def records_to_chunks(records: Sequence[dict]) -> list[Chunk]:
    """Kebalikan :func:`chunks_to_records`."""
    return [
        Chunk(
            chunk_id=int(r["chunk_id"]),
            doc=str(r["doc"]),
            page=int(r["page"]),
            text=str(r["text"]),
            start_word=int(r["start_word"]),
            end_word=int(r["end_word"]),
        )
        for r in records
    ]
