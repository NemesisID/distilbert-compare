"""Uji layanan FastAPI memakai TestClient + router palsu (tanpa model)."""

from __future__ import annotations

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from api import database as db
from api.main import create_app
from api.settings import get_settings
from src.router import RouteDecision

API_KEY = "test-key"
HEADERS = {"X-API-KEY": API_KEY}


class FakeRouter:
    """Router palsu: pertanyaan berisi 'dokumen' -> route dokumen, selain itu faq."""

    def route(self, text: str) -> RouteDecision:
        if "dokumen" in text:
            return RouteDecision(
                route="dokumen",
                intent="tanya_dokumen",
                confidence=0.9,
                answer="Isi dokumen (Sumber: a.pdf, hal. 1).",
                reason="retrieval_hit",
                sources=[
                    {"doc": "a.pdf", "page": 1, "score": 0.5, "dense_score": 0.5,
                     "bm25_score": 1.0, "text": "isi"}
                ],
            )
        return RouteDecision(
            route="faq", intent="sapaan", confidence=0.8, answer="Halo!", reason="faq_answer"
        )


def _settings(tmp_path, **over):
    base = get_settings()
    kw = dict(api_key=API_KEY, database_url=f"sqlite:///{(tmp_path / 't.db').as_posix()}")
    kw.update(over)
    return replace(base, **kw)


@pytest.fixture
def app(tmp_path):
    return create_app(load_router=False, settings=_settings(tmp_path))


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        app.state.router = FakeRouter()
        yield c


def _add_response(app, intent, jawaban, deskripsi=""):
    with app.state.session_factory() as session:
        session.add(db.ResponseEntry(intent=intent, jawaban=jawaban, deskripsi=deskripsi))
        session.commit()


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["router_ready"] is True
    assert body["n_responses"] == 0


# ---------------------------------------------------------------------------
# /chat
# ---------------------------------------------------------------------------


def test_chat_requires_api_key(client):
    assert client.post("/chat", json={"question": "halo"}).status_code == 401


def test_chat_wrong_api_key(client):
    r = client.post("/chat", json={"question": "halo"}, headers={"X-API-KEY": "salah"})
    assert r.status_code == 401


def test_chat_faq_route(client):
    r = client.post("/chat", json={"question": "halo"}, headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["route"] == "faq"
    assert body["answer"] == "Halo!"
    assert body["sources"] == []


def test_chat_document_returns_sources(client):
    r = client.post("/chat", json={"question": "carikan dokumen"}, headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["route"] == "dokumen"
    assert body["sources"][0]["doc"] == "a.pdf"
    assert body["sources"][0]["page"] == 1


def test_chat_blank_question_422(client):
    r = client.post("/chat", json={"question": "   "}, headers=HEADERS)
    assert r.status_code == 422


def test_chat_too_long_422(client):
    r = client.post("/chat", json={"question": "x" * 501}, headers=HEADERS)
    assert r.status_code == 422


def test_chat_logs_to_database(client, app):
    client.post("/chat", json={"question": "halo"}, headers=HEADERS)
    with app.state.session_factory() as session:
        n = session.scalar(select(func.count()).select_from(db.QueryLog))
        row = session.scalars(select(db.QueryLog)).one()
    assert n == 1
    assert row.route == "faq" and row.intent == "sapaan"


def test_chat_503_when_router_not_ready(app):
    with TestClient(app) as c:  # router tidak diset
        r = c.post("/chat", json={"question": "halo"}, headers=HEADERS)
    assert r.status_code == 503


def test_missing_api_key_config_503(tmp_path):
    app = create_app(load_router=False, settings=_settings(tmp_path, api_key=""))
    with TestClient(app) as c:
        r = c.post("/chat", json={"question": "halo"}, headers={"X-API-KEY": "apa"})
    assert r.status_code == 503


def test_rate_limit_429(tmp_path):
    app = create_app(
        load_router=False,
        settings=_settings(tmp_path, api_key="k", rate_limit_per_minute=2),
    )
    with TestClient(app) as c:
        app.state.router = FakeRouter()
        h = {"X-API-KEY": "k"}
        assert c.post("/chat", json={"question": "halo"}, headers=h).status_code == 200
        assert c.post("/chat", json={"question": "halo"}, headers=h).status_code == 200
        assert c.post("/chat", json={"question": "halo"}, headers=h).status_code == 429


# ---------------------------------------------------------------------------
# /intents
# ---------------------------------------------------------------------------


def test_intents_requires_api_key(client):
    assert client.get("/intents").status_code == 401


def test_intents_lists_seeded(client, app):
    _add_response(app, "sapaan", "Halo!", "Sapaan pembuka")
    r = client.get("/intents", headers=HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert [d["intent"] for d in data] == ["sapaan"]
    assert data[0]["jawaban"] == "Halo!"


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------


def test_cors_origin_allowed(tmp_path):
    app = create_app(
        load_router=False,
        settings=_settings(tmp_path, api_key="k", cors_origins=("http://example.com",)),
    )
    with TestClient(app) as c:
        r = c.get("/health", headers={"Origin": "http://example.com"})
    assert r.headers.get("access-control-allow-origin") == "http://example.com"
