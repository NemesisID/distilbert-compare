"""Uji ekstraksi PDF (``rag/pdf_extract.py``) memakai PDF sintetis."""

from __future__ import annotations

import pymupdf
import pytest

from rag import pdf_extract


def _make_pdf(path, pages_text):
    doc = pymupdf.open()
    for lines in pages_text:
        page = doc.new_page()
        y = 72.0
        for line in lines:
            page.insert_text((72, y), line, fontname="helv", fontsize=12)
            y += 18.0
    doc.save(str(path))
    doc.close()


def test_extract_pdf_reads_text_per_page(tmp_path):
    p = tmp_path / "dok.pdf"
    _make_pdf(p, [["halaman satu tentang pendaftaran"], ["halaman dua tentang seleksi"]])
    pages = pdf_extract.extract_pdf(p)
    assert [pg.page for pg in pages] == [1, 2]
    assert all(pg.doc == "dok.pdf" for pg in pages)
    assert "pendaftaran" in pages[0].text
    assert "seleksi" in pages[1].text


def test_extract_pdf_warns_on_scanned(tmp_path):
    p = tmp_path / "scan.pdf"
    _make_pdf(p, [[]])  # halaman kosong -> tanpa lapisan teks
    with pytest.warns(UserWarning, match="scan"):
        pages = pdf_extract.extract_pdf(p)
    assert pages[0].text == ""


def test_iter_pdfs_sorted(tmp_path):
    for name in ("b.pdf", "a.pdf", "c.txt"):
        (tmp_path / name).write_bytes(b"")
    names = [p.name for p in pdf_extract.iter_pdfs(tmp_path)]
    assert names == ["a.pdf", "b.pdf"]


def test_extract_dir_collects_all(tmp_path):
    _make_pdf(tmp_path / "a.pdf", [["isi a"]])
    _make_pdf(tmp_path / "b.pdf", [["isi b", "lebih panjang"]])
    pages = pdf_extract.extract_dir(tmp_path)
    assert {pg.doc for pg in pages} == {"a.pdf", "b.pdf"}
    assert len(pages) == 2
