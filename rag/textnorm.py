"""Normalisasi teks & tokenisasi untuk BM25.

Tokenisasi ``lowercase_space`` (dari config) = NFKC + rapikan spasi + huruf kecil
lalu dipisah spasi. Tidak ada angka ajaib di sini.
"""

from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """NFKC + rapikan spasi berlebih + buang spasi tepi."""
    text = unicodedata.normalize("NFKC", str(text))
    return _WS.sub(" ", text).strip()


def tokenize_lowercase_space(text: str) -> list[str]:
    """Pecah teks menjadi token huruf kecil berbasis spasi."""
    return [t for t in normalize(text).lower().split(" ") if t]


_TOKENIZERS = {
    "lowercase_space": tokenize_lowercase_space,
}


def get_tokenizer(name: str):
    """Ambil fungsi tokenizer berdasarkan nama di config."""
    try:
        return _TOKENIZERS[name]
    except KeyError as exc:  # pragma: no cover - penjaga konfigurasi
        raise ValueError(f"Tokenizer tidak dikenal: {name!r}") from exc
