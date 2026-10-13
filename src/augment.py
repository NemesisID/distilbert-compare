"""Augmentasi teks (HANYA data train).

Menghasilkan variasi gaya sehari-hari dari kalimat train dengan mengganti
istilah memakai kamus yang dapat diedit (``data/augment_dictionary.yaml``).
Label TIDAK berubah. Maksimum variasi per kalimat diambil dari ``config.yaml``.

Contoh:
    python -m src.augment --sandbox
    python -m src.augment                 # train asli -> data/augmented/train_aug.csv
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import yaml

from src import data_io
from src.data_io import INTENT_COL, SOURCE_COL, TEXT_COL

AUGMENTED_SOURCE = "augmented"


def load_dictionary(path: str | Path) -> list[list[str]]:
    """Muat kamus augmentasi menjadi daftar grup istilah setara.

    Args:
        path: Berkas YAML dengan bagian ``slang`` dan/atau ``sinonim``.

    Returns:
        Daftar grup; tiap grup berisi >= 2 istilah huruf kecil.
    """
    with Path(path).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    groups: list[list[str]] = []
    for section in ("slang", "sinonim"):
        for group in data.get(section) or []:
            terms = [str(t).lower().strip() for t in group if str(t).strip()]
            if len(terms) >= 2:
                groups.append(terms)
    return groups


def _term_pattern(term: str) -> re.Pattern[str]:
    """Regex batas kata untuk sebuah istilah (mendukung frasa)."""
    return re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", re.IGNORECASE)


def _present_terms(text: str, groups: Sequence[Sequence[str]]) -> list[tuple[int, list[str]]]:
    """Cari grup yang istilahnya muncul di ``text``."""
    found: list[tuple[int, list[str]]] = []
    for idx, group in enumerate(groups):
        present = [t for t in group if _term_pattern(t).search(text)]
        if present:
            found.append((idx, present))
    return found


def augment_once(
    text: str,
    groups: Sequence[Sequence[str]],
    rng: random.Random,
    max_replacements: int,
) -> str | None:
    """Buat satu variasi kalimat, atau ``None`` bila tak ada yang bisa diganti."""
    matches = _present_terms(text, groups)
    if not matches:
        return None
    rng.shuffle(matches)
    new_text = text
    changed = False
    for idx, present in matches[: max(1, max_replacements)]:
        group = groups[idx]
        # Ganti istilah yang paling panjang lebih dulu (lebih spesifik).
        source = sorted(present, key=len, reverse=True)[0]
        alternatives = [t for t in group if t.lower() != source.lower()]
        if not alternatives:
            continue
        replacement = rng.choice(alternatives)
        candidate = _term_pattern(source).sub(replacement, new_text, count=1)
        if candidate != new_text:
            new_text = candidate
            changed = True
    return new_text if changed else None


def augment_dataframe(
    df: pd.DataFrame,
    groups: Sequence[Sequence[str]],
    *,
    seed: int,
    max_variants: int,
    min_variants: int,
    max_replacements: int,
) -> pd.DataFrame:
    """Hasilkan baris-baris augmentasi baru (tanpa baris asli).

    Returns:
        DataFrame dengan kolom ``text,intent,source`` (source=``augmented``).
    """
    rng = random.Random(seed)
    out: list[dict[str, str]] = []
    for _, row in df.iterrows():
        text = str(row[TEXT_COL]).strip()
        intent = str(row[INTENT_COL])
        seen = {text.lower()}
        made = 0
        attempts = 0
        while made < max_variants and attempts < max_variants * 8:
            attempts += 1
            variant = augment_once(text, groups, rng, max_replacements)
            if variant is None:
                break
            key = variant.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            out.append({TEXT_COL: variant, INTENT_COL: intent, SOURCE_COL: AUGMENTED_SOURCE})
            made += 1
    df_out = pd.DataFrame(out, columns=[TEXT_COL, INTENT_COL, SOURCE_COL])
    return df_out


def sample_review(rows: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """Ambil sampel acak baris augmentasi untuk ditinjau manual."""
    if rows.empty or fraction <= 0:
        return rows.head(0)
    n = max(1, int(round(len(rows) * fraction)))
    return rows.sample(n=min(n, len(rows)), random_state=seed).reset_index(drop=True)


def run_augment(
    train_df: pd.DataFrame,
    groups: Sequence[Sequence[str]],
    *,
    seed: int = 42,
    max_variants: int = 3,
    min_variants: int = 1,
    max_replacements: int = 2,
    review_fraction: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gabungkan train asli + variasi augmentasi dan buat sampel tinjauan.

    Returns:
        (train_aug, review_sample). ``min_variants`` dicatat untuk laporan;
        kalimat yang tidak punya istilah kamus akan lolos tanpa variasi.
    """
    originals = train_df.copy()
    if SOURCE_COL not in originals.columns:
        originals[SOURCE_COL] = "original"

    variants = augment_dataframe(
        train_df,
        groups,
        seed=seed,
        max_variants=max_variants,
        min_variants=min_variants,
        max_replacements=max_replacements,
    )
    combined = pd.concat([originals, variants], ignore_index=True)
    review = sample_review(variants, review_fraction, seed)
    return combined, review


def main(argv: list[str] | None = None) -> int:
    """Titik masuk CLI."""
    parser = argparse.ArgumentParser(description="Augmentasi data train.")
    parser.add_argument("--train", default=None, help="CSV train masukan.")
    parser.add_argument("--out", default=None, help="CSV keluaran train_aug.")
    parser.add_argument("--dictionary", default=None, help="Kamus augmentasi YAML.")
    parser.add_argument("--review", default=None, help="CSV sampel tinjauan.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sandbox", action="store_true")
    args = parser.parse_args(argv)

    cfg = data_io.load_config()
    acfg = cfg["augmentation"]

    if args.sandbox:
        base = data_io.resolve_path(cfg["paths"]["sandbox"]).parent
        train_path = base / "processed" / "train.csv"
        out_path = base / "augmented" / "train_aug.csv"
        review_path = data_io.resolve_path(cfg["paths"]["reports_dir"]) / "sandbox_augmentation_review.csv"
    else:
        train_path = (
            data_io.resolve_path(args.train)
            if args.train
            else data_io.resolve_path(cfg["paths"]["processed_dir"]) / "train.csv"
        )
        out_path = (
            data_io.resolve_path(args.out)
            if args.out
            else data_io.resolve_path(cfg["paths"]["augmented_dir"]) / "train_aug.csv"
        )
        review_path = (
            data_io.resolve_path(args.review)
            if args.review
            else data_io.resolve_path(cfg["paths"]["reports_dir"]) / "augmentation_review.csv"
        )

    dict_path = (
        data_io.resolve_path(args.dictionary)
        if args.dictionary
        else data_io.resolve_path(acfg["dictionary"])
    )
    seed = int(args.seed if args.seed is not None else data_io.get_primary_seed(cfg))

    train_df = data_io.read_csv(train_path, data_io.INTENT_TABLE)
    groups = load_dictionary(dict_path)
    combined, review = run_augment(
        train_df,
        groups,
        seed=seed,
        max_variants=int(acfg["max_variants_per_sentence"]),
        min_variants=int(acfg["min_variants_per_sentence"]),
        max_replacements=int(acfg["max_token_replacements"]),
        review_fraction=float(acfg["review_fraction"]),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(review_path, index=False, encoding="utf-8", lineterminator="\n")

    print(
        f"kamus: {len(groups)} grup | train asli={len(train_df)} | "
        f"variasi baru={len(combined) - len(train_df)} | total={len(combined)}"
    )
    print(f"train_aug -> {out_path}")
    print(f"sampel tinjauan ({len(review)}) -> {review_path}")
    if args.sandbox:
        print("[SANDBOX] hasil ini hanya untuk uji pipeline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
