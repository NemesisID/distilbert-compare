"""Pemeriksaan kebocoran (leakage) antara data train augmentasi dan val.

Aturan:
- GAGAL (exit code != 0) bila ada teks identik setelah normalisasi antara
  ``train_aug`` dan ``val``.
- MELAPORKAN pasangan dengan kemiripan TF-IDF cosine di atas ambang
  (default 0.85) untuk ditinjau manual.

Catatan penting: **test set TIDAK dibaca** di sini (dilindungi ``data_io``).
Karena augmentasi hanya diterapkan pada train SETELAH split, teks augmentasi
tidak mungkin masuk ke test; kebocoran ke test dicegah secara konstruksi.

Contoh:
    python -m src.leakage_check --sandbox
    python -m src.leakage_check                 # train_aug vs val asli
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src import data_io
from src.data_io import SOURCE_COL, TEXT_COL

SANDBOX_SOURCE = "synthetic"


def normalize_text(text: object) -> str:
    """Normalisasi ringan: huruf kecil, buang tanda baca, rapikan spasi."""
    s = str(text).lower()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


@dataclass
class LeakageReport:
    """Ringkasan hasil pemeriksaan kebocoran."""

    n_aug: int
    n_val: int
    threshold: float
    exact: list[tuple[str, str]] = field(default_factory=list)
    similar: list[tuple[str, str, float]] = field(default_factory=list)
    internal_dupes: int = 0
    is_sandbox: bool = False

    @property
    def has_error(self) -> bool:
        """True bila ada kebocoran identik (menggagalkan proses)."""
        return len(self.exact) > 0


def _exact_overlaps(aug: list[str], val: list[str]) -> list[tuple[str, str]]:
    """Cari teks identik (setelah normalisasi) antara aug dan val."""
    val_map: dict[str, str] = {}
    for v in val:
        val_map.setdefault(normalize_text(v), v)
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for a in aug:
        key = normalize_text(a)
        if key and key in val_map and key not in seen:
            seen.add(key)
            found.append((a, val_map[key]))
    return found


def _similar_pairs(
    aug: list[str], val: list[str], threshold: float, top_n: int = 30
) -> list[tuple[str, str, float]]:
    """Cari pasangan aug-val dengan cosine TF-IDF di atas ambang."""
    if not aug or not val:
        return []
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform([normalize_text(t) for t in aug + val])
    aug_mat = matrix[: len(aug)]
    val_mat = matrix[len(aug):]
    sims = cosine_similarity(aug_mat, val_mat)

    pairs: list[tuple[str, str, float]] = []
    for i in range(sims.shape[0]):
        for j in range(sims.shape[1]):
            if sims[i, j] >= threshold:
                pairs.append((aug[i], val[j], float(sims[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_n]


def run_leakage(
    train_aug: pd.DataFrame,
    val: pd.DataFrame,
    threshold: float,
) -> LeakageReport:
    """Jalankan pemeriksaan kebocoran dan kembalikan laporan."""
    aug_texts = train_aug[TEXT_COL].astype(str).tolist()
    val_texts = val[TEXT_COL].astype(str).tolist()

    report = LeakageReport(
        n_aug=len(aug_texts),
        n_val=len(val_texts),
        threshold=threshold,
        is_sandbox=(SOURCE_COL in train_aug.columns and (train_aug[SOURCE_COL] == SANDBOX_SOURCE).any()),
    )
    report.exact = _exact_overlaps(aug_texts, val_texts)
    report.similar = _similar_pairs(aug_texts, val_texts, threshold)

    norm = [normalize_text(t) for t in aug_texts]
    report.internal_dupes = len(norm) - len(set(norm))
    return report


def render_markdown(report: LeakageReport, val_label: str) -> str:
    """Susun laporan kebocoran dalam Markdown."""
    tag = " [SANDBOX - uji pipeline]" if report.is_sandbox else ""
    lines = [
        f"# Laporan Pemeriksaan Kebocoran{tag}",
        "",
        f"- train_aug: {report.n_aug} baris",
        f"- {val_label}: {report.n_val} baris",
        f"- ambang cosine: {report.threshold}",
        f"- teks identik (setelah normalisasi): **{len(report.exact)}**",
        f"- duplikat internal dalam train_aug: {report.internal_dupes}",
        f"- status: **{'GAGAL (ada kebocoran identik)' if report.has_error else 'BERSIH'}**",
        "",
        "> Catatan: test set tidak dibaca di sini (dilindungi data_io).",
        "",
        "## Teks identik train_aug <-> val",
        "",
    ]
    if report.exact:
        lines.append("| train_aug | val |")
        lines.append("|-----------|-----|")
        for a, b in report.exact[:30]:
            lines.append(f"| {a} | {b} |")
    else:
        lines.append("- (tidak ada)")

    lines += [
        "",
        f"## Pasangan mirip (cosine >= {report.threshold}) - untuk ditinjau",
        "",
    ]
    if report.similar:
        lines.append("| skor | train_aug | val |")
        lines.append("|------|-----------|-----|")
        for a, b, s in report.similar:
            lines.append(f"| {s:.4f} | {a} | {b} |")
    else:
        lines.append("- (tidak ada)")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Titik masuk CLI."""
    parser = argparse.ArgumentParser(description="Periksa kebocoran train_aug vs val.")
    parser.add_argument("--train-aug", default=None)
    parser.add_argument("--val", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--sandbox", action="store_true")
    args = parser.parse_args(argv)

    cfg = data_io.load_config()
    threshold = float(
        args.threshold if args.threshold is not None else cfg["leakage"]["cosine_threshold"]
    )

    if args.sandbox:
        base = data_io.resolve_path(cfg["paths"]["sandbox"]).parent
        aug_path = base / "augmented" / "train_aug.csv"
        val_path = base / "processed" / "val.csv"
        out = data_io.resolve_path(cfg["paths"]["reports_dir"]) / "sandbox_leakage_report.md"
    else:
        aug_path = (
            data_io.resolve_path(args.train_aug)
            if args.train_aug
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
            else data_io.resolve_path(cfg["paths"]["reports_dir"]) / "leakage_report.md"
        )

    train_aug = data_io.read_csv(aug_path, data_io.INTENT_TABLE)
    val = data_io.read_csv(val_path, data_io.INTENT_TABLE)
    report = run_leakage(train_aug, val, threshold)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(report, "val"), encoding="utf-8")
    print(render_markdown(report, "val"))
    print(f"\nLaporan: {out}")
    return 1 if report.has_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
