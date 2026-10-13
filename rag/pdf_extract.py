"""Ekstraksi teks per halaman dari PDF memakai PyMuPDF (``import pymupdf``).

PDF hasil scan (tanpa lapisan teks) akan memicu peringatan karena OCR
(Tesseract) belum terpasang di lingkungan ini.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pymupdf


@dataclass(frozen=True)
class PageText:
    """Teks satu halaman PDF."""

    doc: str
    page: int  # 1-based
    text: str


def iter_pdfs(pdf_dir: str | Path) -> Iterator[Path]:
    """Iterasi berkas ``*.pdf`` (terurut) dalam sebuah folder."""
    yield from sorted(Path(pdf_dir).glob("*.pdf"))


def extract_pdf(path: str | Path) -> list[PageText]:
    """Ekstrak teks tiap halaman dari satu PDF."""
    path = Path(path)
    pages: list[PageText] = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            pages.append(PageText(doc=path.name, page=i, text=text))
    if pages and not any(p.text for p in pages):
        warnings.warn(
            f"{path.name}: tidak ada teks terekstrak (kemungkinan PDF hasil scan; "
            "OCR/Tesseract belum terpasang)",
            stacklevel=2,
        )
    return pages


def extract_dir(pdf_dir: str | Path) -> list[PageText]:
    """Ekstrak seluruh PDF dalam folder menjadi daftar halaman."""
    out: list[PageText] = []
    for pdf in iter_pdfs(pdf_dir):
        out.extend(extract_pdf(pdf))
    return out
