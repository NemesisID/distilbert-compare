"""Pencatat eksperimen otomatis.

Setiap eksperimen menambahkan satu baris ke ``EXPERIMENTS.md`` (di bawah
penanda ``<!-- APPEND_ROWS_HERE -->``) dan satu baris ke
``reports/experiments.csv`` agar mudah diolah.
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path
from typing import Mapping

from src.data_io import PROJECT_ROOT

SENTINEL = "<!-- APPEND_ROWS_HERE -->"
CSV_FIELDS = [
    "tanggal",
    "fase",
    "model",
    "konfigurasi",
    "seed",
    "metrik",
    "durasi",
    "sumber",
    "catatan",
]


def _clean(value: str) -> str:
    """Bersihkan nilai agar aman ditaruh di sel tabel Markdown."""
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def _fmt_metrics(metrics: Mapping[str, float]) -> str:
    """Rangkum metrik menjadi satu string."""
    parts = []
    for key, val in metrics.items():
        if isinstance(val, float):
            parts.append(f"{key}={val:.4f}")
        else:
            parts.append(f"{key}={val}")
    return " | ".join(parts) if parts else "-"


def log_experiment(
    *,
    fase: str,
    model: str,
    config: str,
    seed: object,
    metrics: Mapping[str, float],
    duration_s: float,
    notes: str = "",
    sandbox: bool = False,
    experiments_path: str | Path | None = None,
    csv_path: str | Path | None = None,
) -> None:
    """Catat satu eksperimen.

    Args:
        fase: Label fase/skrip (mis. ``"Fase 2 - baseline"``).
        model: Nama model/metode.
        config: Ringkasan konfigurasi.
        seed: Seed yang dipakai.
        metrics: Dict metrik, mis. ``{"accuracy": 0.9, "f1_macro": 0.88}``.
        duration_s: Durasi (detik).
        notes: Catatan tambahan.
        sandbox: Tandai bahwa hasil berasal dari data sandbox.
        experiments_path: Lokasi EXPERIMENTS.md.
        csv_path: Lokasi CSV log.
    """
    tag = "[SANDBOX] " if sandbox else ""
    if sandbox:
        notes = f"{tag}{notes}".strip()

    row = {
        "tanggal": date.today().isoformat(),
        "fase": _clean(fase),
        "model": _clean(model),
        "konfigurasi": _clean(f"{tag}{config}"),
        "seed": _clean(str(seed)),
        "metrik": _clean(f"{tag}{_fmt_metrics(metrics)}"),
        "durasi": f"{duration_s:.2f}s",
        "sumber": "sandbox" if sandbox else "real",
        "catatan": _clean(notes),
    }

    exp_path = Path(experiments_path) if experiments_path else PROJECT_ROOT / "EXPERIMENTS.md"
    _append_markdown_row(exp_path, row)

    log_csv = Path(csv_path) if csv_path else PROJECT_ROOT / "reports" / "experiments.csv"
    _append_csv_row(log_csv, row)


def _append_markdown_row(path: Path, row: Mapping[str, str]) -> None:
    """Sisipkan baris tabel tepat setelah penanda di EXPERIMENTS.md."""
    line = (
        f"| {row['tanggal']} | {row['fase']} | {row['model']} | "
        f"{row['konfigurasi']} | {row['seed']} | {row['metrik']} | {row['durasi']} |"
    )
    if path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        text = f"# EXPERIMENTS.md\n\n{SENTINEL}\n"

    if SENTINEL in text:
        before, _, after = text.partition(SENTINEL)
        new_text = f"{before}{SENTINEL}\n{line}\n{after.lstrip(chr(10))}"
    else:
        new_text = text.rstrip("\n") + "\n" + line + "\n"
    path.write_text(new_text, encoding="utf-8")


def _append_csv_row(path: Path, row: Mapping[str, str]) -> None:
    """Tambahkan baris ke CSV log (buat header bila perlu)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})
