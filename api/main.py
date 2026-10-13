"""Aplikasi FastAPI: ``/health``, ``/intents``, ``/chat``.

Model + indeks dimuat sekali saat startup (lifespan). Jawaban FAQ diambil dari
database (diisi lewat ``scripts/seed_responses.py``); kutipan dokumen dari
indeks retrieval. Tidak ada layanan permanen yang dijalankan dari kode ini.

Contoh:
    uvicorn api.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import replace

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select

from api import database as db
from api.schemas import ChatRequest, ChatResponse, HealthResponse, IntentItem
from api.security import RateLimiter, rate_limit, verify_api_key
from api.settings import Settings, get_settings
from src import data_io


def build_router(cfg: dict):
    """Bangun :class:`src.router.Router` (model + indeks + jawaban) dari config."""
    from src.router import Router
    from src.train import model_slug

    return Router(
        cfg,
        model_dir=data_io.resolve_path(cfg["paths"]["models_dir"])
        / model_slug(cfg["models"]["primary"]),
        vectorstore_dir=data_io.resolve_path(cfg["paths"]["vectorstore_dir"]),
        answers_path=data_io.resolve_path(cfg["paths"]["answers"]),
    )


def get_router(request: Request):
    """Dependensi: ambil Router dari state aplikasi (503 bila belum siap)."""
    router = request.app.state.router
    if router is None:
        raise HTTPException(
            status_code=503, detail="Router belum siap (model/indeks belum dimuat)."
        )
    return router


def get_session(request: Request):
    """Dependensi: buka sesi database per permintaan."""
    factory = request.app.state.session_factory
    session = factory()
    try:
        yield session
    finally:
        session.close()


def create_app(
    *,
    load_router: bool = True,
    settings: Settings | None = None,
    cfg: dict | None = None,
    database_url: str | None = None,
) -> FastAPI:
    """Rakit aplikasi FastAPI.

    Args:
        load_router: Muat model + indeks saat startup. ``False`` untuk pengujian.
        settings: Pengaturan; default dari config + ``.env``.
        cfg: Config yang sudah dimuat (opsional).
        database_url: Override URL database (mis. untuk pengujian).
    """
    cfg = cfg or data_io.load_config()
    settings = settings or get_settings(cfg)
    if database_url:
        settings = replace(settings, database_url=database_url)
    engine = db.make_engine(settings.database_url)
    session_factory = db.make_session_factory(engine)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db.init_db(engine)
        if load_router:
            app.state.router = build_router(cfg)
        yield

    app = FastAPI(title=str(cfg["project"]["name"]), version="0.1.0", lifespan=lifespan)
    app.state.cfg = cfg
    app.state.settings = settings
    app.state.session_factory = session_factory
    app.state.rate_limiter = RateLimiter(
        settings.rate_limit_per_minute, settings.rate_limit_window_seconds
    )
    app.state.router = None

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request, session=Depends(get_session)) -> HealthResponse:
        n = session.scalar(select(func.count()).select_from(db.ResponseEntry)) or 0
        return HealthResponse(
            status="ok",
            router_ready=request.app.state.router is not None,
            n_responses=int(n),
            database="ok",
        )

    @app.get(
        "/intents",
        response_model=list[IntentItem],
        dependencies=[Depends(verify_api_key)],
    )
    def list_intents(session=Depends(get_session)) -> list[IntentItem]:
        rows = session.scalars(
            select(db.ResponseEntry).order_by(db.ResponseEntry.intent)
        ).all()
        return [
            IntentItem(intent=r.intent, deskripsi=r.deskripsi, jawaban=r.jawaban)
            for r in rows
        ]

    @app.post(
        "/chat",
        response_model=ChatResponse,
        dependencies=[Depends(verify_api_key), Depends(rate_limit)],
    )
    def chat(
        payload: ChatRequest,
        router=Depends(get_router),
        session=Depends(get_session),
    ) -> ChatResponse:
        question = payload.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="Pertanyaan kosong.")
        if len(question) > settings.max_input_chars:
            raise HTTPException(
                status_code=422,
                detail=f"Pertanyaan terlalu panjang (maks {settings.max_input_chars} karakter).",
            )
        decision = router.route(question)
        session.add(
            db.QueryLog(
                question=question,
                route=decision.route,
                intent=decision.intent,
                confidence=float(decision.confidence),
                answer=decision.answer,
                sources_json=json.dumps(decision.sources, ensure_ascii=False),
            )
        )
        session.commit()
        return ChatResponse(
            route=decision.route,
            intent=decision.intent,
            confidence=float(decision.confidence),
            answer=decision.answer,
            sources=decision.sources,
        )

    return app


app = create_app()
