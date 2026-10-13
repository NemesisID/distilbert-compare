"""Tes logika validasi dataset."""

from __future__ import annotations

import pandas as pd

from src import data_io
from src.validate_data import validate_df

INTENTS = ["sapaan", "out_of_scope", "tanya_dokumen"]
MIN_COUNTS = {"default": 2, "tanya_dokumen": 2, "out_of_scope": 2}


def _df(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["text", "intent"])


def _validate(rows: list[tuple[str, str]]):
    return validate_df(
        _df(rows), INTENTS, MIN_COUNTS, min_chars=3, max_chars=50, source_label="tes"
    )


def test_valid_data_has_no_errors() -> None:
    report = _validate(
        [
            ("halo", "sapaan"),
            ("hai", "sapaan"),
            ("apa itu", "out_of_scope"),
            ("bukan urusan", "out_of_scope"),
            ("dokumen mana", "tanya_dokumen"),
            ("cari file", "tanya_dokumen"),
        ]
    )
    assert report.ok, report.errors


def test_unknown_intent_is_error() -> None:
    report = _validate([("halo", "sapaan"), ("x", "intent_ajaib")])
    assert not report.ok
    assert any("tak dikenal" in e for e in report.errors)


def test_same_text_different_label_is_error() -> None:
    report = _validate([("halo", "sapaan"), ("halo", "out_of_scope")])
    assert not report.ok
    assert any("lebih dari satu label" in e for e in report.errors)


def test_duplicate_text_is_warning() -> None:
    report = _validate([("halo", "sapaan"), ("halo", "sapaan")])
    assert report.ok  # duplikat bukan error
    assert any("duplikat" in w for w in report.warnings)


def test_low_count_is_warning() -> None:
    report = _validate([("halo", "sapaan"), ("hai", "sapaan")])
    assert report.ok
    assert any("out_of_scope" in w and "contoh" in w for w in report.warnings)


def test_extreme_length_is_warning() -> None:
    report = _validate([("hi", "sapaan"), ("x" * 60, "out_of_scope")])
    assert any("lebih pendek" in w for w in report.warnings)
    assert any("lebih panjang" in w for w in report.warnings)


def test_missing_columns_is_error() -> None:
    df = pd.DataFrame({"foo": ["bar"]})
    report = validate_df(df, INTENTS, MIN_COUNTS, 3, 50)
    assert not report.ok
    assert any("Kolom wajib hilang" in e for e in report.errors)


def test_sandbox_flagged() -> None:
    df = pd.DataFrame(
        [("halo", "sapaan", "synthetic")], columns=["text", "intent", "source"]
    )
    report = validate_df(df, INTENTS, MIN_COUNTS, 3, 50)
    assert report.is_sandbox


def test_sandbox_dataset_passes_without_errors() -> None:
    cfg = data_io.load_config()
    df = data_io.load_sandbox()
    report = validate_df(
        df,
        intents=data_io.get_intents(cfg),
        min_counts=cfg["intent_min_counts"],
        min_chars=int(cfg["validation"]["min_chars"]),
        max_chars=int(cfg["validation"]["max_chars"]),
        source_label="sandbox",
    )
    assert report.ok, report.errors
    assert report.is_sandbox
