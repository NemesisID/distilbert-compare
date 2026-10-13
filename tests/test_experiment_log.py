"""Tes pencatat eksperimen."""

from __future__ import annotations

import csv
from pathlib import Path

from src import experiment_log


def test_log_experiment_appends_row_and_csv(tmp_path: Path) -> None:
    exp = tmp_path / "EXPERIMENTS.md"
    exp.write_text(
        "# EXPERIMENTS.md\n\n| a | b |\n|---|---|\n"
        + experiment_log.SENTINEL
        + "\n",
        encoding="utf-8",
    )
    csv_path = tmp_path / "experiments.csv"

    experiment_log.log_experiment(
        fase="Fase 2 - baseline",
        model="TF-IDF + LinearSVC",
        config="ngram=(1,2)",
        seed=42,
        metrics={"accuracy": 0.9, "f1_macro": 0.88},
        duration_s=1.234,
        notes="uji",
        sandbox=True,
        experiments_path=exp,
        csv_path=csv_path,
    )

    text = exp.read_text(encoding="utf-8")
    assert "TF-IDF + LinearSVC" in text
    assert "[SANDBOX]" in text
    # baris disisipkan tepat setelah sentinel, sebelum konten sesudahnya
    idx_sentinel = text.index(experiment_log.SENTINEL)
    idx_row = text.index("TF-IDF + LinearSVC")
    assert idx_row > idx_sentinel

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["sumber"] == "sandbox"
    assert rows[0]["metrik"].startswith("[SANDBOX]")


def test_log_experiment_twice_appends_two_rows(tmp_path: Path) -> None:
    exp = tmp_path / "EXPERIMENTS.md"
    exp.write_text(f"{experiment_log.SENTINEL}\n", encoding="utf-8")
    csv_path = tmp_path / "experiments.csv"
    for acc in (0.8, 0.9):
        experiment_log.log_experiment(
            fase="f",
            model="m",
            config="c",
            seed=42,
            metrics={"accuracy": acc},
            duration_s=0.1,
            experiments_path=exp,
            csv_path=csv_path,
        )
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
