"""Pemuat jawaban baku FAQ dari ``data/intents.yaml``.

Jawaban baku dikelola sebagai data (di-seed ke database), bukan di dalam model.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def load_answer_entries(path: str | Path) -> list[dict]:
    """Baca ``intents.yaml`` -> daftar entri (hanya yang punya ``intent``)."""
    p = Path(path)
    if not p.exists():
        return []
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or []
    if not isinstance(data, list):
        return []
    return [e for e in data if isinstance(e, dict) and "intent" in e]


def load_answers(path: str | Path) -> dict[str, str]:
    """Baca ``intents.yaml`` -> peta intent -> jawaban baku."""
    return {
        str(e["intent"]): str(e.get("jawaban", "")).strip()
        for e in load_answer_entries(path)
    }
