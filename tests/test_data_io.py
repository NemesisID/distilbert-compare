"""Tes penjaga test set dan pemuat data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src import data_io
from src.data_io import ForbiddenTestSetError


@pytest.fixture(autouse=True)
def _clear_allow_test(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pastikan ALLOW_TEST tidak bocor dari lingkungan."""
    monkeypatch.delenv(data_io.ALLOW_TEST_ENV, raising=False)


def _write(path: Path, rows: list[dict[str, str]]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    return path


def test_is_test_path_detects_test_variants() -> None:
    assert data_io.is_test_path("test.csv")
    assert data_io.is_test_path("data/processed/test.csv")
    assert data_io.is_test_path("mixed_test.csv")
    assert not data_io.is_test_path("train.csv")
    assert not data_io.is_test_path("val.csv")
    assert not data_io.is_test_path("contest.csv")  # token 'test' bukan kata utuh


def test_guard_blocks_without_env() -> None:
    with pytest.raises(ForbiddenTestSetError):
        data_io.guard_path("data/processed/test.csv")


def test_guard_allows_with_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(data_io.ALLOW_TEST_ENV, "1")
    assert data_io.is_test_path("test.csv")
    # Tidak melempar:
    data_io.guard_path("data/processed/test.csv")


def test_read_csv_blocks_test(tmp_path: Path) -> None:
    p = _write(tmp_path / "test.csv", [{"text": "a", "intent": "sapaan"}])
    with pytest.raises(ForbiddenTestSetError):
        data_io.read_csv(p, data_io.INTENT_TABLE)


def test_read_csv_allows_test_with_both_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _write(tmp_path / "test.csv", [{"text": "a", "intent": "sapaan"}])
    monkeypatch.setenv(data_io.ALLOW_TEST_ENV, "1")
    df = data_io.read_csv(p, data_io.INTENT_TABLE, allow_test=True)
    assert len(df) == 1


def test_read_csv_missing_column(tmp_path: Path) -> None:
    p = _write(tmp_path / "train.csv", [{"text": "a"}])
    with pytest.raises(ValueError):
        data_io.read_csv(p, data_io.INTENT_TABLE)


def test_read_csv_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        data_io.read_csv("data/processed/tidak_ada_train.csv", data_io.INTENT_TABLE)


def test_load_sandbox_shape() -> None:
    df = data_io.load_sandbox()
    assert {"text", "intent", "source"} <= set(df.columns)
    assert (df["source"] == "synthetic").all()
    assert len(df) >= 150


def test_config_intents_present() -> None:
    cfg = data_io.load_config()
    intents = data_io.get_intents(cfg)
    assert "out_of_scope" in intents
    assert "tanya_dokumen" in intents
    assert len(intents) == 11
