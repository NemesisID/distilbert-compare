"""Konfigurasi layanan API: gabungan ``config.yaml`` + variabel ``.env``.

Rahasia (kunci API) HANYA dibaca dari lingkungan/``.env``, tidak pernah dari
kode atau dari ``config.yaml``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from src import data_io


def load_dotenv(path: Path) -> None:
    """Muat ``.env`` bila ada (tidak menimpa variabel yang sudah diset)."""
    try:
        from dotenv import load_dotenv as _load
    except ImportError:  # pragma: no cover - python-dotenv terpasang di proyek
        return
    if path.exists():
        _load(path, override=False)


def _default_database_url(cfg: dict) -> str:
    """Bangun URL SQLite dari ``paths.database``."""
    db = data_io.resolve_path(cfg["paths"]["database"])
    db.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db.as_posix()}"


@dataclass(frozen=True)
class Settings:
    """Pengaturan runtime layanan."""

    api_key: str
    api_key_header: str
    database_url: str
    cors_origins: tuple[str, ...]
    rate_limit_per_minute: int
    rate_limit_window_seconds: int
    max_input_chars: int


def get_settings(cfg: dict | None = None) -> Settings:
    """Susun :class:`Settings` dari config + ``.env`` (env menang)."""
    cfg = cfg or data_io.load_config()
    load_dotenv(data_io.PROJECT_ROOT / ".env")
    api_cfg = cfg["api"]

    api_key = os.environ.get("CHATBOT_API_KEY", "").strip()
    database_url = os.environ.get("DATABASE_URL", "").strip() or _default_database_url(cfg)

    raw_origins = os.environ.get("CORS_ORIGINS", "").strip()
    if raw_origins:
        cors_origins = tuple(o.strip() for o in raw_origins.split(",") if o.strip())
    else:
        cors_origins = tuple(str(o).strip() for o in (api_cfg.get("cors_origins") or []) if str(o).strip())

    raw_rate = os.environ.get("RATE_LIMIT_PER_MINUTE", "").strip()
    rate_limit = int(raw_rate) if raw_rate else int(api_cfg["rate_limit_per_minute"])

    return Settings(
        api_key=api_key,
        api_key_header=str(api_cfg["api_key_header"]),
        database_url=database_url,
        cors_origins=cors_origins,
        rate_limit_per_minute=rate_limit,
        rate_limit_window_seconds=int(api_cfg["rate_limit_window_seconds"]),
        max_input_chars=int(api_cfg["max_input_chars"]),
    )
