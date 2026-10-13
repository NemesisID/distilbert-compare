"""Uji pemotongan chunk (``rag/chunking.py``)."""

from __future__ import annotations

import pytest

from rag import chunking
from rag.pdf_extract import PageText


def test_chunk_text_single_chunk_when_short():
    out = chunking.chunk_text("satu dua tiga", chunk_words=10, overlap=2)
    assert out == [("satu dua tiga", 0, 3)]


def test_chunk_text_overlap_matches_step():
    words = " ".join(str(i) for i in range(10))
    out = chunking.chunk_text(words, chunk_words=4, overlap=1)
    # step = 3 -> start 0, 3, 6; chunk 6..10 sudah sampai akhir teks.
    assert [s for _, s, _ in out] == [0, 3, 6]
    # chunk pertama 0..4, kedua mulai di kata ke-3 (tumpang tindih 1 kata).
    assert out[0][0].split() == ["0", "1", "2", "3"]
    assert out[1][0].split()[0] == "3"
    # chunk terakhir berhenti tepat di akhir teks.
    assert out[-1][2] == 10


def test_chunk_text_empty_returns_empty():
    assert chunking.chunk_text("   ", chunk_words=5, overlap=1) == []


def test_chunk_text_invalid_params():
    with pytest.raises(ValueError):
        chunking.chunk_text("a b c", chunk_words=0, overlap=0)
    with pytest.raises(ValueError):
        chunking.chunk_text("a b c", chunk_words=3, overlap=3)
    with pytest.raises(ValueError):
        chunking.chunk_text("a b c", chunk_words=3, overlap=-1)


def test_chunk_pages_assigns_metadata_and_ids():
    pages = [
        PageText(doc="a.pdf", page=1, text=" ".join(str(i) for i in range(6))),
        PageText(doc="b.pdf", page=2, text=" ".join(str(i) for i in range(6))),
    ]
    chunks = chunking.chunk_pages(pages, chunk_words=4, overlap=0)
    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))
    assert chunks[0].doc == "a.pdf" and chunks[0].page == 1
    assert chunks[-1].doc == "b.pdf" and chunks[-1].page == 2


def test_chunk_records_roundtrip():
    pages = [PageText(doc="a.pdf", page=1, text=" ".join(str(i) for i in range(5)))]
    chunks = chunking.chunk_pages(pages, chunk_words=3, overlap=1)
    again = chunking.records_to_chunks(chunking.chunks_to_records(chunks))
    assert again == chunks
