"""Model Pydantic permintaan/balasan layanan API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Source(BaseModel):
    """Satu sumber kutipan dari dokumen PDF."""

    doc: str
    page: int
    score: float
    dense_score: float | None = None
    bm25_score: float | None = None
    text: str = ""


class ChatRequest(BaseModel):
    """Permintaan ``POST /chat``."""

    question: str = Field(..., min_length=1, description="Pertanyaan pengguna.")
    session_id: str | None = Field(default=None, description="Id sesi (opsional).")


class ChatResponse(BaseModel):
    """Balasan ``POST /chat``."""

    route: str = Field(..., description="faq | dokumen | fallback")
    intent: str
    confidence: float
    answer: str
    sources: list[Source] = Field(default_factory=list)


class IntentItem(BaseModel):
    """Satu entri FAQ (``GET /intents``)."""

    intent: str
    deskripsi: str = ""
    jawaban: str = ""


class HealthResponse(BaseModel):
    """Balasan ``GET /health``."""

    status: str
    router_ready: bool
    n_responses: int
    database: str
