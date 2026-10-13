"""Database (SQLAlchemy) untuk jawaban baku FAQ & log interaksi.

Default: SQLite (``paths.database`` / ``DATABASE_URL``). Tidak ada layanan
permanen; tabel dibuat lewat ``init_db`` saat skrip seed atau API dijalankan.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    """Basis deklaratif SQLAlchemy."""


class ResponseEntry(Base):
    """Jawaban baku untuk satu intent (menggantikan FAQ statis)."""

    __tablename__ = "responses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    intent: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    deskripsi: Mapped[str] = mapped_column(Text, default="")
    jawaban: Mapped[str] = mapped_column(Text)


class QueryLog(Base):
    """Catatan setiap panggilan ``/chat`` (untuk audit & evaluasi)."""

    __tablename__ = "query_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    question: Mapped[str] = mapped_column(Text)
    route: Mapped[str] = mapped_column(String(16), index=True)
    intent: Mapped[str] = mapped_column(String(64), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    answer: Mapped[str] = mapped_column(Text)
    sources_json: Mapped[str] = mapped_column(Text, default="[]")

    def sources(self) -> list[dict]:
        """Kembalikan daftar sumber (hasil ``json.loads``)."""
        return json.loads(self.sources_json or "[]")


def make_engine(database_url: str):
    """Buat engine SQLAlchemy (mode sinkron; SQLite aman lintas thread)."""
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    return create_engine(database_url, connect_args=connect_args, future=True)


def make_session_factory(engine) -> sessionmaker:
    """Buat pabrik sesi SQLAlchemy."""
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def init_db(engine) -> None:
    """Buat tabel yang belum ada."""
    Base.metadata.create_all(engine)
