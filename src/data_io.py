"""Pemuat data tunggal untuk seluruh proyek.

Modul ini adalah SATU-SATUNYA pintu masuk untuk membaca data tabular.
Perannya yang paling penting adalah **penjaga test set**: berkas test
(`test.csv`, `mixed_test.csv`, dst.) hanya boleh dibaca bila variabel
lingkungan ``ALLOW_TEST=1`` diset, dan hanya ``src/final_eval.py`` yang
boleh menyetelnya.

Seluruh modul lain WAJIB memuat data melalui fungsi di sini.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Konstanta
# ---------------------------------------------------------------------------

TEXT_COL = "text"
INTENT_COL = "intent"
SOURCE_COL = "source"

#: Berkas yang namanya mengandung token ini dianggap "test set" terlindungi.
PROTECTED_TOKENS: frozenset[str] = frozenset({"test"})

#: Nama variabel lingkungan yang membuka akses ke test set.
ALLOW_TEST_ENV = "ALLOW_TEST"

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]


class ForbiddenTestSetError(RuntimeError):
    """Dibangkitkan saat kode mencoba membaca test set tanpa izin."""


# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------


def load_config(path: str | Path | None = None) -> dict:
    """Muat ``config.yaml`` sebagai dict.

    Args:
        path: Lokasi config. Default ``<root>/config.yaml``.

    Returns:
        Isi konfigurasi.
    """
    cfg_path = Path(path) if path is not None else PROJECT_ROOT / "config.yaml"
    with cfg_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_intents(config: Mapping | None = None) -> list[str]:
    """Ambil daftar intent dari konfigurasi (urutan dipertahankan)."""
    cfg = config if config is not None else load_config()
    return list(cfg["intents"])


def resolve_path(relative: str | Path) -> Path:
    """Ubah jalur relatif (dari config) menjadi absolut terhadap root repo."""
    p = Path(relative)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


# ---------------------------------------------------------------------------
# Penjaga test set
# ---------------------------------------------------------------------------


def _path_tokens(path: Path) -> list[str]:
    """Pecah nama berkas menjadi token huruf kecil (dipisah _ - . dan spasi)."""
    stem = path.stem.lower()
    for ch in ("-", ".", " "):
        stem = stem.replace(ch, "_")
    return [t for t in stem.split("_") if t]


def is_test_path(path: str | Path) -> bool:
    """True bila berkas dianggap bagian dari test set terlindungi."""
    return bool(set(_path_tokens(Path(path))) & PROTECTED_TOKENS)


def test_access_allowed() -> bool:
    """True bila ``ALLOW_TEST=1`` diset (hanya final_eval.py yang boleh)."""
    return os.environ.get(ALLOW_TEST_ENV) == "1"


def guard_path(path: str | Path) -> Path:
    """Tolak akses ke test set kecuali diizinkan.

    Args:
        path: Berkas yang akan dibaca/ditulis.

    Returns:
        ``Path`` yang sama bila diizinkan.

    Raises:
        ForbiddenTestSetError: Bila berkas adalah test set dan izin tidak ada.
    """
    p = Path(path)
    if is_test_path(p) and not test_access_allowed():
        raise ForbiddenTestSetError(
            f"Akses test set ditolak: '{p.name}'. Hanya src/final_eval.py yang "
            f"boleh membaca test set (perlu {ALLOW_TEST_ENV}=1)."
        )
    return p


# ---------------------------------------------------------------------------
# Pembacaan tabel
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TableSpec:
    """Spesifikasi kolom minimal sebuah tabel."""

    required: tuple[str, ...]
    optional: tuple[str, ...] = ()


INTENT_TABLE = TableSpec(required=(TEXT_COL, INTENT_COL), optional=(SOURCE_COL,))
QUESTION_TABLE = TableSpec(required=("question",))
QA_PAIRS_TABLE = TableSpec(required=("question", "doc", "page"))
MIXED_TABLE = TableSpec(required=("question", "expected_route"))


def read_csv(
    path: str | Path,
    spec: TableSpec | None = None,
    *,
    allow_test: bool = False,
) -> pd.DataFrame:
    """Baca CSV UTF-8 dengan penjaga test set.

    Args:
        path: Berkas CSV.
        spec: Kolom minimal yang wajib ada.
        allow_test: Set ``True`` HANYA dari ``final_eval.py``. Sebagai
            tambahan, ``ALLOW_TEST=1`` juga harus diset di lingkungan.

    Returns:
        DataFrame hasil pembacaan.

    Raises:
        ForbiddenTestSetError: Bila membaca test set tanpa izin.
        FileNotFoundError: Bila berkas tidak ada.
        ValueError: Bila kolom wajib tidak ditemukan.
    """
    p = Path(path)
    if not (allow_test and test_access_allowed()):
        guard_path(p)  # akan melempar bila ini test set
    if not p.exists():
        raise FileNotFoundError(f"Berkas tidak ditemukan: {p}")

    df = pd.read_csv(p, encoding="utf-8", dtype=str, keep_default_na=False)
    if spec is not None:
        missing = [c for c in spec.required if c not in df.columns]
        if missing:
            raise ValueError(
                f"Kolom wajib {missing} tidak ada di {p.name}. "
                f"Kolom tersedia: {list(df.columns)}"
            )
    return df


# ---------------------------------------------------------------------------
# Pemuat siap pakai
# ---------------------------------------------------------------------------


def _cfg_paths() -> dict:
    return load_config()["paths"]


def load_intents_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Muat data mentah intent (``data/raw/intents_raw.csv``)."""
    p = resolve_path(path) if path is not None else resolve_path(_cfg_paths()["raw"])
    return read_csv(p, INTENT_TABLE)


def load_sandbox(path: str | Path | None = None) -> pd.DataFrame:
    """Muat dataset sandbox sintetis (hanya untuk uji pipeline)."""
    p = (
        resolve_path(path)
        if path is not None
        else resolve_path(_cfg_paths()["sandbox"])
    )
    return read_csv(p, INTENT_TABLE)


def load_processed(split: str, *, allow_test: bool = False) -> pd.DataFrame:
    """Muat salah satu split hasil ``split.py``.

    Args:
        split: ``"train"``, ``"val"``, atau ``"test"``.
        allow_test: Hanya boleh ``True`` dari ``final_eval.py``.
    """
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split tidak dikenal: {split!r}")
    p = resolve_path(_cfg_paths()["processed_dir"]) / f"{split}.csv"
    if split == "test" and not (allow_test and test_access_allowed()):
        guard_path(p)
    return read_csv(p, INTENT_TABLE)


def load_augmented(path: str | Path | None = None) -> pd.DataFrame:
    """Muat ``data/augmented/train_aug.csv``."""
    merged = resolve_path(_cfg_paths()["augmented_dir"]) / "train_aug.csv"
    p = resolve_path(path) if path is not None else merged
    return read_csv(p, INTENT_TABLE)


def load_mixed(split: str, *, allow_test: bool = False) -> pd.DataFrame:
    """Muat data end-to-end ``data/e2e/mixed_{split}.csv``."""
    if split not in {"val", "test"}:
        raise ValueError(f"split e2e tidak dikenal: {split!r}")
    p = resolve_path(_cfg_paths()["e2e_dir"]) / f"mixed_{split}.csv"
    if split == "test" and not (allow_test and test_access_allowed()):
        guard_path(p)
    return read_csv(p, MIXED_TABLE)


def load_questions(path: str | Path) -> pd.DataFrame:
    """Muat tabel berisi kolom ``question`` (mis. in_doc/out_doc)."""
    return read_csv(resolve_path(path), QUESTION_TABLE)


def validate_columns(df: pd.DataFrame, required: Iterable[str]) -> list[str]:
    """Kembalikan daftar kolom wajib yang hilang dari ``df``."""
    return [c for c in required if c not in df.columns]
