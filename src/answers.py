"""Pemuat jawaban baku FAQ dari ``data/intents.yaml``.

Jawaban baku dikelola sebagai data (di-seed ke database), bukan di dalam model.

Sebuah intent boleh punya **satu jawaban** (string) atau **beberapa jawaban**
(daftar). Dengan daftar, chatbot memilih salah satu jawaban secara acak tiap
permintaan sehingga jawaban tidak monoton — tetapi isinya tetap teks baku yang
disetujui (non-generatif).

Contoh:
    - intent: sapaan
      jawaban: "Halo! Ada yang bisa saya bantu?"
      # boleh juga:
      jawaban:
        - "Halo! Ada yang bisa saya bantu?"
        - "Selamat datang! Mau tanya apa hari ini?"
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import yaml


def normalize_variants(value: object) -> list[str]:
    """Normalisasi nilai ``jawaban`` menjadi daftar teks (string ATAU daftar).

    Args:
        value: ``str`` (satu jawaban), daftar teks, atau ``None``.

    Returns:
        Daftar jawaban tidak kosong; urutan dipertahankan.
    """
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            text = "" if item is None else str(item).strip()
            if text:
                out.append(text)
        return out
    text = str(value).strip()
    return [text] if text else []


def primary_answer(entry: dict) -> str:
    """Jawaban utama sebuah entri (variasi pertama; ``""`` bila kosong)."""
    variants = normalize_variants(entry.get("jawaban"))
    return variants[0] if variants else ""


def load_answer_entries(path: str | Path) -> list[dict]:
    """Baca ``intents.yaml`` -> daftar entri (hanya yang punya ``intent``)."""
    p = Path(path)
    if not p.exists():
        return []
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or []
    if not isinstance(data, list):
        return []
    return [e for e in data if isinstance(e, dict) and "intent" in e]


def load_answer_variants(path: str | Path) -> dict[str, list[str]]:
    """Baca ``intents.yaml`` -> peta intent -> daftar variasi jawaban.

    Entri tanpa jawaban valid ikut dikeluarkan (tidak ada variasi).
    """
    out: dict[str, list[str]] = {}
    for entry in load_answer_entries(path):
        variants = normalize_variants(entry.get("jawaban"))
        if not variants:
            continue
        out[str(entry["intent"])] = variants
    return out


def load_answers(path: str | Path) -> dict[str, str]:
    """Baca ``intents.yaml`` -> peta intent -> jawaban baku (variasi pertama)."""
    return {str(e["intent"]): primary_answer(e) for e in load_answer_entries(path)}


def pick_answer(variants: Sequence[str], rng: random.Random | None = None) -> str:
    """Pilih satu jawaban dari ``variants``.

    Deterministik bila ``rng`` tidak diberikan (selalu variasi pertama) atau
    daftar hanya berisi satu jawaban.

    Args:
        variants: Daftar variasi jawaban.
        rng: RNG pilihan; ``None`` = tidak berubah-ubah (untuk laporan/evaluasi).

    Returns:
        Jawaban terpilih, atau ``""`` bila tidak ada variasi valid.
    """
    items = [str(v).strip() for v in variants]
    items = [v for v in items if v]
    if not items:
        return ""
    if len(items) == 1 or rng is None:
        return items[0]
    return rng.choice(items)
