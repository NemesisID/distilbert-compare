"""Tes pemeriksaan kebocoran."""

from __future__ import annotations

import pandas as pd

from src import leakage_check as lc
from src.data_io import INTENT_COL, SOURCE_COL, TEXT_COL


def _df(rows: list[tuple[str, str]], source: str = "original") -> pd.DataFrame:
    return pd.DataFrame(
        [(t, i, source) for t, i in rows], columns=[TEXT_COL, INTENT_COL, SOURCE_COL]
    )


def test_normalize_text() -> None:
    assert lc.normalize_text("  Bagaimana, CARA?! ") == "bagaimana cara"
    assert lc.normalize_text("a   b\nc") == "a b c"


def test_exact_overlap_detected() -> None:
    aug = _df([("bagaimana cara daftar", "alur_seleksi")])
    val = _df([("Bagaimana cara daftar!", "alur_seleksi")])  # identik setelah normalisasi
    report = lc.run_leakage(aug, val, threshold=0.85)
    assert report.has_error
    assert len(report.exact) == 1


def test_clean_data_has_no_error() -> None:
    aug = _df([("bagaimana cara daftar inkubator", "alur_seleksi")])
    val = _df([("apa saja syarat pendaftaran", "syarat_pendaftaran")])
    report = lc.run_leakage(aug, val, threshold=0.85)
    assert not report.has_error
    assert report.exact == []


def test_similar_pairs_reported_above_threshold() -> None:
    aug = _df([("bagaimana cara daftar inkubator bisnis", "alur_seleksi")])
    val = _df([("bagaimana cara daftar inkubator bisnis ya", "alur_seleksi")])
    report = lc.run_leakage(aug, val, threshold=0.5)
    assert report.similar
    assert report.similar[0][2] >= 0.5


def test_internal_duplicates_counted() -> None:
    aug = _df([("halo dunia", "sapaan"), ("HALO dunia", "sapaan")])
    val = _df([("x y z", "out_of_scope")])
    report = lc.run_leakage(aug, val, threshold=0.9)
    assert report.internal_dupes == 1


def test_render_markdown_runs() -> None:
    aug = _df([("halo", "sapaan")])
    val = _df([("hai", "sapaan")])
    md = lc.render_markdown(lc.run_leakage(aug, val, 0.85), "val")
    assert "Laporan Pemeriksaan Kebocoran" in md
