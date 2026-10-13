"""Split dataset menjadi train/val/test dan kunci test set.

Aturan penting:
- Split dilakukan SEBELUM augmentasi (augmentasi hanya pada train).
- ``test.csv`` dikunci: SHA-256 dicatat di ``TEST_SHA256.txt`` dan berkas
  dijadikan read-only. Hanya ``src/final_eval.py`` yang boleh membacanya.

Contoh:
    python -m src.split --sandbox
    python -m src.split                 # dari data/raw/intents_raw.csv
"""

from __future__ import annotations

import argparse
import hashlib
import stat
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src import data_io
from src.data_io import INTENT_COL, SOURCE_COL, TEXT_COL

TEST_SHA_FILENAME = "TEST_SHA256.txt"
DEFAULT_SOURCE = "original"


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Buang baris duplikat berdasarkan teks (tanpa memperhatikan besar-kecil).

    Args:
        df: Tabel intent.

    Returns:
        Tuple (DataFrame bersih, jumlah baris yang dibuang).
    """
    work = df.copy()
    work["_key"] = work[TEXT_COL].astype(str).str.strip().str.lower()
    before = len(work)
    work = work.drop_duplicates(subset=["_key"], keep="first")
    work = work.drop(columns=["_key"]).reset_index(drop=True)
    return work, before - len(work)


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Bagi data secara stratified 70/15/15 (default).

    Args:
        df: Tabel intent yang sudah bersih.
        train_ratio: Proporsi train.
        val_ratio: Proporsi validasi.
        test_ratio: Proporsi test.
        seed: Random state.

    Returns:
        (train, val, test).
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Total rasio train/val/test harus 1.0")

    holdout = val_ratio + test_ratio
    use_stratify = df[INTENT_COL].value_counts().min() >= 3
    strat = df[INTENT_COL] if use_stratify else None

    train_df, temp_df = train_test_split(
        df,
        test_size=holdout,
        random_state=seed,
        stratify=strat,
        shuffle=True,
    )

    temp_strat = temp_df[INTENT_COL] if use_stratify else None
    # Bagi holdout menjadi val dan test dengan proporsi relatif.
    test_share = test_ratio / holdout
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_share,
        random_state=seed,
        stratify=temp_strat,
        shuffle=True,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def sha256_file(path: str | Path) -> str:
    """Hitung SHA-256 sebuah berkas."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def set_readonly(path: str | Path, readonly: bool = True) -> None:
    """Set/buka atribut read-only (di Windows memakai bit read-only)."""
    p = Path(path)
    mode = p.stat().st_mode
    if readonly:
        p.chmod(mode & ~stat.S_IWRITE)
    else:
        p.chmod(mode | stat.S_IWRITE)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    """Tulis CSV UTF-8, hapus read-only bila berkas sudah ada."""
    if path.exists():
        set_readonly(path, readonly=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def run_split(
    df: pd.DataFrame,
    out_dir: str | Path,
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    lock_test: bool = True,
) -> dict[str, Path]:
    """Jalankan deduplikasi + split + penguncian test set.

    Returns:
        Dict berisi jalur ``train``, ``val``, ``test``, ``sha``.
    """
    out = Path(out_dir)
    clean, removed = deduplicate(df)
    if SOURCE_COL not in clean.columns:
        clean = clean.assign(**{SOURCE_COL: DEFAULT_SOURCE})

    train_df, val_df, test_df = stratified_split(
        clean, train_ratio, val_ratio, test_ratio, seed
    )

    paths = {
        "train": out / "train.csv",
        "val": out / "val.csv",
        "test": out / "test.csv",
    }
    _write_csv(train_df, paths["train"])
    _write_csv(val_df, paths["val"])
    _write_csv(test_df, paths["test"])

    digest = sha256_file(paths["test"])
    sha_path = out / TEST_SHA_FILENAME
    sha_path.write_text(f"{digest}  test.csv\n", encoding="utf-8")
    paths["sha"] = sha_path

    if lock_test:
        set_readonly(paths["test"], readonly=True)

    print(
        f"deduplikasi: {removed} baris dibuang | "
        f"train={len(train_df)} val={len(val_df)} test={len(test_df)}"
    )
    return paths


def main(argv: list[str] | None = None) -> int:
    """Titik masuk CLI."""
    parser = argparse.ArgumentParser(description="Split dataset intent.")
    parser.add_argument("--input", default=None, help="Berkas CSV masukan.")
    parser.add_argument("--sandbox", action="store_true", help="Pakai data sandbox.")
    parser.add_argument(
        "--out-dir", default=None, help="Direktori keluaran (default data/processed)."
    )
    parser.add_argument("--seed", type=int, default=None, help="Override seed split.")
    args = parser.parse_args(argv)

    cfg = data_io.load_config()

    if args.input is not None:
        path = data_io.resolve_path(args.input)
    elif args.sandbox:
        path = data_io.resolve_path(cfg["paths"]["sandbox"])
    else:
        path = data_io.resolve_path(cfg["paths"]["raw"])

    df = data_io.read_csv(path, data_io.INTENT_TABLE)
    out_dir = (
        data_io.resolve_path(args.out_dir)
        if args.out_dir
        else data_io.resolve_path(cfg["paths"]["processed_dir"])
    )
    scfg = cfg["split"]
    paths = run_split(
        df,
        out_dir,
        train_ratio=float(scfg["train"]),
        val_ratio=float(scfg["val"]),
        test_ratio=float(scfg["test"]),
        seed=int(args.seed if args.seed is not None else scfg["random_state"]),
    )
    print(f"TEST_SHA256: {paths['sha'].read_text(encoding='utf-8').strip()}")
    print(f"test set dikunci (read-only): {paths['test']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
