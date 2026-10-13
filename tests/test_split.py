"""Tes split data dan penguncian test set."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src import data_io, split as split_mod
from src.data_io import ForbiddenTestSetError


@pytest.fixture(autouse=True)
def _clear_allow_test(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(data_io.ALLOW_TEST_ENV, raising=False)


def _toy_df() -> pd.DataFrame:
    rows = []
    for intent in ("sapaan", "tanya_dokumen", "out_of_scope"):
        for i in range(10):
            rows.append(
                {
                    "text": f"contoh {intent} nomor {i}",
                    "intent": intent,
                    "source": "synthetic",
                }
            )
    return pd.DataFrame(rows)


def test_deduplicate_removes_exact_duplicates() -> None:
    df = pd.DataFrame(
        [
            ("halo", "sapaan", "s"),
            ("Halo", "sapaan", "s"),
            ("hai", "sapaan", "s"),
        ],
        columns=["text", "intent", "source"],
    )
    clean, removed = split_mod.deduplicate(df)
    assert removed == 1
    assert len(clean) == 2


def test_split_proportions_and_lock(tmp_path: Path) -> None:
    paths = split_mod.run_split(_toy_df(), tmp_path, seed=42, lock_test=True)
    try:
        assert "test" in paths
        # Test set terkunci: data_io menolak membacanya.
        with pytest.raises(ForbiddenTestSetError):
            data_io.read_csv(paths["test"], data_io.INTENT_TABLE)

        # SHA-256 tercatat dan cocok.
        recorded = paths["sha"].read_text(encoding="utf-8").split()[0]
        assert recorded == split_mod.sha256_file(paths["test"])

        # Read-only benar-benar aktif.
        assert not (paths["test"].stat().st_mode & 0o200)

        n_train = len(pd.read_csv(paths["train"]))
        n_val = len(pd.read_csv(paths["val"]))
        n_test = len(pd.read_csv(paths["test"]))
        assert n_train + n_val + n_test == 30
        assert n_train > n_val and n_train > n_test

        # Semua intent hadir di train (stratified).
        train_intents = set(pd.read_csv(paths["train"])["intent"])
        assert train_intents == {"sapaan", "tanya_dokumen", "out_of_scope"}
    finally:
        split_mod.set_readonly(paths["test"], readonly=False)


def test_split_is_reproducible(tmp_path: Path) -> None:
    a = split_mod.run_split(_toy_df(), tmp_path / "a", seed=42, lock_test=False)
    b = split_mod.run_split(_toy_df(), tmp_path / "b", seed=42, lock_test=False)
    da = pd.read_csv(a["test"]).to_dict("records")
    db = pd.read_csv(b["test"]).to_dict("records")
    assert da == db


def test_shuffled_source_columns_preserved(tmp_path: Path) -> None:
    paths = split_mod.run_split(_toy_df(), tmp_path, seed=7, lock_test=False)
    for key in ("train", "val", "test"):
        df = pd.read_csv(paths[key])
        assert set(df.columns) >= {"text", "intent", "source"}
