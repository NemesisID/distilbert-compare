"""Validasi dataset intent.

Memeriksa kolom, nilai kosong, intent tak dikenal, duplikat, konflik label,
jumlah contoh per intent, dan panjang teks ekstrem. Menghasilkan laporan
Markdown di ``reports/data_validation.md``.

Contoh:
    python -m src.validate_data --input data/sandbox/intents_sandbox.csv
    python -m src.validate_data            # memakai data/raw/intents_raw.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src import data_io
from src.data_io import INTENT_COL, SOURCE_COL, TEXT_COL

SANDBOX_SOURCE = "synthetic"


@dataclass
class ValidationReport:
    """Hasil validasi dataset."""

    source_label: str
    n_rows: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    intent_counts: dict[str, int] = field(default_factory=dict)
    is_sandbox: bool = False

    @property
    def ok(self) -> bool:
        """True bila tidak ada error (peringatan diperbolehkan)."""
        return not self.errors


def _norm(text: str) -> str:
    """Normalisasi ringan: huruf kecil, rapikan spasi, buang spasi tepi."""
    return " ".join(str(text).lower().split())


def validate_df(
    df: pd.DataFrame,
    intents: list[str],
    min_counts: dict[str, int],
    min_chars: int,
    max_chars: int,
    source_label: str = "(tidak diketahui)",
) -> ValidationReport:
    """Validasi DataFrame intent dan kembalikan laporan.

    Args:
        df: Tabel dengan kolom ``text,intent[,source]``.
        intents: Daftar intent yang sah.
        min_counts: Ambang minimum per intent (kunci ``default`` + override).
        min_chars: Panjang teks minimum yang wajar.
        max_chars: Panjang teks maksimum yang wajar.
        source_label: Label sumber data untuk laporan.
    """
    report = ValidationReport(source_label=source_label, n_rows=len(df))
    known = set(intents)

    missing = data_io.validate_columns(df, (TEXT_COL, INTENT_COL))
    if missing:
        report.errors.append(f"Kolom wajib hilang: {missing}")
        return report

    text = df[TEXT_COL].astype(str)
    intent = df[INTENT_COL].astype(str)

    if SOURCE_COL in df.columns:
        report.is_sandbox = (df[SOURCE_COL].astype(str) == SANDBOX_SOURCE).any()

    # 1. Nilai kosong
    blank = (text.str.strip() == "") | (intent.str.strip() == "")
    if blank.any():
        report.errors.append(
            f"Ada {int(blank.sum())} baris dengan text/intent kosong."
        )

    # 2. Intent tak dikenal
    unknown = ~intent.isin(known)
    if unknown.any():
        bad = sorted(intent[unknown].unique().tolist())
        report.errors.append(
            f"Ada {int(unknown.sum())} baris dengan intent tak dikenal: {bad}"
        )

    # 3. Duplikat teks (persis, setelah trim)
    trimmed = text.str.strip()
    dup_mask = trimmed.duplicated(keep=False)
    n_dup = int(trimmed[trimmed.duplicated(keep=False)].nunique())
    if dup_mask.any():
        dup_examples = trimmed[dup_mask].value_counts().head(5).to_dict()
        report.warnings.append(
            f"Ada {n_dup} teks duplikat (total {int(dup_mask.sum())} baris). "
            f"Contoh: {list(dup_examples.items())}"
        )

    # 4. Teks sama dengan label berbeda (konflik)
    grouped = df.assign(_t=trimmed).groupby("_t")[INTENT_COL].nunique()
    conflicts = grouped[grouped > 1]
    if len(conflicts) > 0:
        report.errors.append(
            f"Ada {len(conflicts)} teks dengan lebih dari satu label "
            f"(mis. {list(conflicts.index[:3])}). Perlu ditinjau manual."
        )

    # 5. Jumlah per intent
    counts = intent[intent.isin(known)].value_counts().to_dict()
    report.intent_counts = {k: int(counts.get(k, 0)) for k in intents}
    default_min = int(min_counts.get("default", 0))
    for name in intents:
        threshold = int(min_counts.get(name, default_min))
        got = report.intent_counts.get(name, 0)
        if got < threshold:
            report.warnings.append(
                f"Intent '{name}' hanya punya {got} contoh (target >= {threshold})."
            )

    # 6. Panjang ekstrem
    lengths = text.str.len()
    too_short = int(((lengths < min_chars) & (text.str.strip() != "")).sum())
    too_long = int((lengths > max_chars).sum())
    if too_short:
        report.warnings.append(
            f"Ada {too_short} teks lebih pendek dari {min_chars} karakter."
        )
    if too_long:
        report.warnings.append(
            f"Ada {too_long} teks lebih panjang dari {max_chars} karakter."
        )

    return report


def render_markdown(report: ValidationReport) -> str:
    """Susun laporan validasi dalam format Markdown."""
    lines: list[str] = []
    tag = " (SANDBOX - hanya uji pipeline)" if report.is_sandbox else ""
    lines.append("# Laporan Validasi Data")
    lines.append("")
    lines.append(f"sumber: `{report.source_label}`{tag}")
    lines.append(f"jumlah baris: {report.n_rows}")
    status = "LULUS (tanpa error)" if report.ok else "GAGAL"
    lines.append(f"status: **{status}**")
    lines.append("")
    lines.append("## Error")
    lines.append("")
    lines.extend([f"- {e}" for e in report.errors] or ["- (tidak ada)"])
    lines.append("")
    lines.append("## Peringatan")
    lines.append("")
    lines.extend([f"- {w}" for w in report.warnings] or ["- (tidak ada)"])
    lines.append("")
    lines.append("## Jumlah per intent")
    lines.append("")
    lines.append("| intent | jumlah |")
    lines.append("|--------|--------|")
    for name, cnt in report.intent_counts.items():
        lines.append(f"| {name} | {cnt} |")
    lines.append("")
    return "\n".join(lines)


def write_report(report: ValidationReport, out_path: str | Path) -> Path:
    """Tulis laporan Markdown ke disk (UTF-8)."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_markdown(report), encoding="utf-8")
    return p


def main(argv: list[str] | None = None) -> int:
    """Titik masuk CLI."""
    parser = argparse.ArgumentParser(description="Validasi dataset intent.")
    parser.add_argument(
        "--input",
        default=None,
        help="Berkas CSV masukan. Default: data/raw/intents_raw.csv dari config.",
    )
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="Pakai dataset sandbox sintetis.",
    )
    parser.add_argument("--out", default=None, help="Berkas laporan Markdown.")
    args = parser.parse_args(argv)

    cfg = data_io.load_config()
    intents = data_io.get_intents(cfg)
    vcfg = cfg["validation"]

    if args.input is not None:
        path = data_io.resolve_path(args.input)
    elif args.sandbox:
        path = data_io.resolve_path(cfg["paths"]["sandbox"])
    else:
        path = data_io.resolve_path(cfg["paths"]["raw"])

    df = data_io.read_csv(path, data_io.INTENT_TABLE)
    report = validate_df(
        df,
        intents=intents,
        min_counts=cfg["intent_min_counts"],
        min_chars=int(vcfg["min_chars"]),
        max_chars=int(vcfg["max_chars"]),
        source_label=str(path.relative_to(data_io.PROJECT_ROOT))
        if path.is_relative_to(data_io.PROJECT_ROOT)
        else str(path),
    )
    out = (
        data_io.resolve_path(args.out)
        if args.out
        else data_io.resolve_path(cfg["paths"]["reports_dir"]) / "data_validation.md"
    )
    write_report(report, out)
    print(render_markdown(report))
    print(f"\nLaporan ditulis ke: {out}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
