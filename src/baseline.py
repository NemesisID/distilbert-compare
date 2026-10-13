"""Baseline pembanding: TF-IDF + LinearSVC.

Dilatih pada train (biasanya ``train_aug``) dan dievaluasi pada ``val``.
Metrik dan waktu dicatat otomatis ke ``EXPERIMENTS.md``.

Contoh:
    python -m src.baseline --sandbox
    python -m src.baseline
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src import data_io, experiment_log
from src.data_io import INTENT_COL, SOURCE_COL, TEXT_COL

SANDBOX_SOURCE = "synthetic"


def build_pipeline(
    *,
    ngram_max: int = 2,
    sublinear_tf: bool = True,
    C: float = 1.0,
    max_iter: int = 2000,
) -> Pipeline:
    """Bangun pipeline TF-IDF + LinearSVC."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(ngram_range=(1, ngram_max), sublinear_tf=sublinear_tf),
            ),
            ("clf", LinearSVC(C=C, max_iter=max_iter)),
        ]
    )


def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    """Hitung metrik klasifikasi utama."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def evaluate_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    ngram_max: int = 2,
    sublinear_tf: bool = True,
    C: float = 1.0,
    max_iter: int = 2000,
) -> dict:
    """Latih baseline dan evaluasi pada val.

    Returns:
        Dict berisi ``model``, ``metrics``, ``report`` (str), ``y_true``,
        ``y_pred``, ``duration_s``.
    """
    model = build_pipeline(
        ngram_max=ngram_max, sublinear_tf=sublinear_tf, C=C, max_iter=max_iter
    )
    x_train = train_df[TEXT_COL].astype(str).tolist()
    y_train = train_df[INTENT_COL].astype(str).tolist()
    x_val = val_df[TEXT_COL].astype(str).tolist()
    y_true = val_df[INTENT_COL].astype(str).tolist()

    start = time.perf_counter()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val).tolist()
    duration = time.perf_counter() - start

    return {
        "model": model,
        "metrics": compute_metrics(y_true, y_pred),
        "report": classification_report(y_true, y_pred, zero_division=0),
        "y_true": y_true,
        "y_pred": y_pred,
        "duration_s": duration,
    }


def render_markdown(result: dict, *, train_label: str, val_label: str, sandbox: bool) -> str:
    """Susun laporan baseline dalam Markdown."""
    tag = " [SANDBOX - uji pipeline, BUKAN hasil penelitian]" if sandbox else ""
    lines = [
        f"# Baseline TF-IDF + LinearSVC{tag}",
        "",
        f"- train: {train_label}",
        f"- val: {val_label}",
        f"- durasi latih+prediksi: {result['duration_s']:.2f}s",
        "",
        "## Metrik (val)",
        "",
        "| metrik | nilai |",
        "|--------|-------|",
    ]
    for key, val in result["metrics"].items():
        lines.append(f"| {key} | {val:.4f} |")
    lines += ["", "## Laporan per intent", "", "```", result["report"].rstrip(), "```", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Titik masuk CLI."""
    parser = argparse.ArgumentParser(description="Baseline TF-IDF + LinearSVC.")
    parser.add_argument("--train", default=None)
    parser.add_argument("--val", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--sandbox", action="store_true")
    args = parser.parse_args(argv)

    cfg = data_io.load_config()
    bcfg = cfg["baseline"]

    if args.sandbox:
        base = data_io.resolve_path(cfg["paths"]["sandbox"]).parent
        train_path = base / "augmented" / "train_aug.csv"
        val_path = base / "processed" / "val.csv"
        out = data_io.resolve_path(cfg["paths"]["reports_dir"]) / "sandbox_baseline_val.md"
    else:
        train_path = (
            data_io.resolve_path(args.train)
            if args.train
            else data_io.resolve_path(cfg["paths"]["augmented_dir"]) / "train_aug.csv"
        )
        val_path = (
            data_io.resolve_path(args.val)
            if args.val
            else data_io.resolve_path(cfg["paths"]["processed_dir"]) / "val.csv"
        )
        out = (
            data_io.resolve_path(args.out)
            if args.out
            else data_io.resolve_path(cfg["paths"]["reports_dir"]) / "baseline_val.md"
        )

    train_df = data_io.read_csv(train_path, data_io.INTENT_TABLE)
    val_df = data_io.read_csv(val_path, data_io.INTENT_TABLE)
    sandbox = SOURCE_COL in train_df.columns and (train_df[SOURCE_COL] == SANDBOX_SOURCE).any()

    result = evaluate_baseline(
        train_df,
        val_df,
        ngram_max=int(bcfg["ngram_max"]),
        sublinear_tf=bool(bcfg["sublinear_tf"]),
        C=float(bcfg["C"]),
        max_iter=int(bcfg["max_iter"]),
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        render_markdown(result, train_label=str(train_path.name), val_label=str(val_path.name), sandbox=sandbox),
        encoding="utf-8",
    )
    print(render_markdown(result, train_label=str(train_path.name), val_label=str(val_path.name), sandbox=sandbox))
    print(f"\nLaporan: {out}")

    experiment_log.log_experiment(
        fase="Fase 2 - baseline",
        model="TF-IDF + LinearSVC",
        config=(
            f"ngram=(1,{bcfg['ngram_max']}) sublinear_tf={bcfg['sublinear_tf']} "
            f"C={bcfg['C']} max_iter={bcfg['max_iter']} train={train_path.name}"
        ),
        seed=data_io.get_primary_seed(cfg),
        metrics=result["metrics"],
        duration_s=result["duration_s"],
        notes="baseline pembanding",
        sandbox=sandbox,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
