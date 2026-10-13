"""Tes baseline TF-IDF + LinearSVC."""

from __future__ import annotations

import pandas as pd

from src import baseline
from src.data_io import INTENT_COL, TEXT_COL


def _toy(n_per_intent: int = 12) -> pd.DataFrame:
    rows = []
    templates = {
        "sapaan": "halo selamat datang di layanan {}",
        "tanya_dokumen": "di mana dokumen panduan nomor {}",
        "out_of_scope": "resep kue yang enak nomor {}",
    }
    for intent, tpl in templates.items():
        for i in range(n_per_intent):
            rows.append((tpl.format(i), intent))
    return pd.DataFrame(rows, columns=[TEXT_COL, INTENT_COL])


def test_compute_metrics_perfect() -> None:
    y = ["a", "b", "a", "b"]
    metrics = baseline.compute_metrics(y, y)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_macro"] == 1.0


def test_evaluate_baseline_returns_metrics() -> None:
    train = _toy(12)
    val = _toy(3).assign(**{TEXT_COL: lambda d: d[TEXT_COL] + " ekstra"})
    result = baseline.evaluate_baseline(train, val)
    assert 0.0 <= result["metrics"]["accuracy"] <= 1.0
    assert set(result["metrics"]) >= {"accuracy", "f1_macro", "f1_weighted"}
    assert result["duration_s"] >= 0.0
    assert "precision" in result["report"] or "accuracy" in result["report"]


def test_render_markdown_contains_metrics() -> None:
    result = baseline.evaluate_baseline(_toy(10), _toy(3))
    md = baseline.render_markdown(result, train_label="train.csv", val_label="val.csv", sandbox=True)
    assert "f1_macro" in md
    assert "SANDBOX" in md
