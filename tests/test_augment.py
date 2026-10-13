"""Tes augmentasi."""

from __future__ import annotations

import pandas as pd

from src import augment as aug
from src.data_io import INTENT_COL, SOURCE_COL, TEXT_COL

GROUPS = [
    ["yang", "yg"],
    ["tidak", "gak", "ga"],
    ["bagaimana", "gimana", "gmn"],
    ["dokumen", "berkas", "file"],
]


def test_load_dictionary_from_file() -> None:
    groups = aug.load_dictionary("data/augment_dictionary.yaml")
    assert len(groups) >= 10
    # Semua istilah huruf kecil dan unik dalam satu grup.
    for g in groups:
        assert all(t == t.lower() for t in g)
        assert len(set(g)) == len(g)


def test_augment_once_replaces_term() -> None:
    import random

    rng = random.Random(42)
    out = aug.augment_once("bagaimana cara melihat dokumen", GROUPS, rng, 1)
    assert out is not None
    assert out.lower() != "bagaimana cara melihat dokumen"


def test_augment_once_returns_none_when_no_terms() -> None:
    import random

    rng = random.Random(42)
    assert aug.augment_once("xyz qqq", GROUPS, rng, 2) is None


def test_augment_keeps_labels_and_limits_variants() -> None:
    df = pd.DataFrame(
        [
            (f"bagaimana cara melihat dokumen nomor {i}", "tanya_dokumen", "synthetic")
            for i in range(5)
        ],
        columns=[TEXT_COL, INTENT_COL, SOURCE_COL],
    )
    variants = aug.augment_dataframe(
        df, GROUPS, seed=42, max_variants=3, min_variants=1, max_replacements=2
    )
    assert set(variants[INTENT_COL]) == {"tanya_dokumen"}
    assert (variants[SOURCE_COL] == aug.AUGMENTED_SOURCE).all()
    # maksimum 3 variasi per kalimat asli
    assert len(variants) <= 15
    # tidak ada variasi yang sama dengan aslinya
    assert not set(variants[TEXT_COL]) & set(df[TEXT_COL])


def test_augment_is_deterministic() -> None:
    df = pd.DataFrame(
        [("bagaimana cara mendaftar", "alur_seleksi", "x")] * 1,
        columns=[TEXT_COL, INTENT_COL, SOURCE_COL],
    )
    a = aug.augment_dataframe(df, GROUPS, seed=7, max_variants=3, min_variants=1, max_replacements=2)
    b = aug.augment_dataframe(df, GROUPS, seed=7, max_variants=3, min_variants=1, max_replacements=2)
    assert a[TEXT_COL].tolist() == b[TEXT_COL].tolist()


def test_run_augment_combines_and_reviews() -> None:
    df = pd.DataFrame(
        [
            (f"bagaimana dokumen nomor {i}", "tanya_dokumen", "synthetic")
            for i in range(10)
        ],
        columns=[TEXT_COL, INTENT_COL, SOURCE_COL],
    )
    combined, review = aug.run_augment(
        df, GROUPS, seed=42, max_variants=3, min_variants=1, max_replacements=2, review_fraction=0.1
    )
    # train asli tetap ada seluruhnya
    assert len(combined) >= len(df)
    assert set(df[TEXT_COL]) <= set(combined[TEXT_COL])
    n_variants = len(combined) - len(df)
    if n_variants > 0:
        assert 1 <= len(review) <= n_variants
