"""Uji seed jawaban baku ke database & pemuat jawaban."""

from __future__ import annotations

from sqlalchemy import select

from api import database as db
from scripts.seed_responses import seed_responses
from src.answers import load_answer_entries, load_answers


def _factory(tmp_path):
    engine = db.make_engine(f"sqlite:///{(tmp_path / 's.db').as_posix()}")
    db.init_db(engine)
    return db.make_session_factory(engine)


def test_seed_upsert_is_idempotent(tmp_path):
    factory = _factory(tmp_path)
    entries = [
        {"intent": "sapaan", "deskripsi": "d1", "jawaban": "Halo"},
        {"intent": "sapaan", "deskripsi": "d2", "jawaban": "Hai"},
        {"intent": "terima_kasih", "jawaban": "Sama-sama"},
    ]
    with factory() as session:
        processed = seed_responses(session, entries)
    assert processed == 3

    with factory() as session:
        rows = session.scalars(
            select(db.ResponseEntry).order_by(db.ResponseEntry.intent)
        ).all()
    assert [r.intent for r in rows] == ["sapaan", "terima_kasih"]
    assert rows[0].jawaban == "Hai"  # entri terakhir menang (upsert)
    assert rows[0].deskripsi == "d2"


def test_load_answer_entries_and_answers(tmp_path):
    p = tmp_path / "intents.yaml"
    p.write_text(
        "- intent: sapaan\n  deskripsi: s\n  jawaban: Halo\n"
        "- intent: terima_kasih\n  jawaban: Sama-sama\n",
        encoding="utf-8",
    )
    entries = load_answer_entries(p)
    assert len(entries) == 2
    assert load_answers(p) == {"sapaan": "Halo", "terima_kasih": "Sama-sama"}


def test_load_answer_entries_missing_returns_empty(tmp_path):
    assert load_answer_entries(tmp_path / "tidak_ada.yaml") == []
    assert load_answers(tmp_path / "tidak_ada.yaml") == {}
