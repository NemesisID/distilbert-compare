"""Verifikasi kunci API dan pembatas laju sederhana (per kunci/IP)."""

from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request


def verify_api_key(request: Request) -> str:
    """Bergantung pada header kunci API (mis. ``X-API-KEY``).

    Raises:
        HTTPException: 503 bila kunci belum dikonfigurasi, 401 bila tidak cocok.
    """
    settings = request.app.state.settings
    if not settings.api_key:
        raise HTTPException(
            status_code=503,
            detail="CHATBOT_API_KEY belum dikonfigurasi di .env.",
        )
    provided = request.headers.get(settings.api_key_header)
    if provided != settings.api_key:
        raise HTTPException(status_code=401, detail="API key tidak valid.")
    return provided


class RateLimiter:
    """Pembatas laju jendela-geser sederhana di memori proses."""

    def __init__(self, limit_per_window: int, window_seconds: float = 60.0):
        self.limit = int(limit_per_window)
        self.window = float(window_seconds)
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str, now: float | None = None) -> None:
        """Catat satu permintaan; lempar 429 bila melewati batas."""
        now = time.monotonic() if now is None else float(now)
        window = self._hits[key]
        while window and now - window[0] > self.window:
            window.popleft()
        if len(window) >= self.limit:
            raise HTTPException(
                status_code=429,
                detail="Terlalu banyak permintaan. Silakan coba lagi nanti.",
            )
        window.append(now)


def rate_limit(request: Request) -> None:
    """Dependensi pembatas laju: kunci per API key, jika tidak per alamat klien."""
    limiter: RateLimiter = request.app.state.rate_limiter
    settings = request.app.state.settings
    key = request.headers.get(settings.api_key_header)
    if not key:
        key = request.client.host if request.client else "unknown"
    limiter.check(key)
