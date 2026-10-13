"""Uji normalisasi & tokenisasi (``rag/textnorm.py``)."""

from __future__ import annotations

import pytest

from rag import textnorm


def test_normalize_collapses_whitespace():
    assert textnorm.normalize("  halo   dunia \n baru ") == "halo dunia baru"


def test_normalize_nfkc():
    # Ligatur "ﬁ" dinormalisasi NFKC menjadi "fi".
    assert textnorm.normalize("\ufb01nal") == "final"


def test_tokenize_lowercase_space():
    assert textnorm.tokenize_lowercase_space("Halo, Dunia  Baru") == ["halo,", "dunia", "baru"]


def test_tokenize_empty():
    assert textnorm.tokenize_lowercase_space("   ") == []


def test_get_tokenizer_known():
    assert textnorm.get_tokenizer("lowercase_space") is textnorm.tokenize_lowercase_space


def test_get_tokenizer_unknown_raises():
    with pytest.raises(ValueError):
        textnorm.get_tokenizer("tidak_ada")
