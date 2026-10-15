"""Isi tabel ``responses`` dari ``data/intents.yaml`` (jawaban baku FAQ).

Idempoten (upsert berdasarkan ``intent``). ``jawaban`` boleh berupa satu teks
atau daftar teks; daftar disimpan penuh sebagai variasi jawaban
(``jawaban_variants``) sehingga chatbot bisa memilih variasi secara acak.

Contoh:
    .venv\\Scripts\\python.exe -m scripts.seed_responses
    .venv\\Scripts\\python.exe -m scripts.seed_responses --sandbox
"""

from __future__ import annotations

import argparse
import json

from sqlalchemy import func, select

from api import database as db
from api.settings import get_settings
from src import data_io
from src.answers import load_answer_entries, normalize_variants


def seed_responses(session, entries: list[dict]) -> int:
    """Upsert entri jawaban baku berdasarkan ``intent``; kembalikan jumlah diproses.

    ``jawaban`` boleh berupa satu teks atau daftar teks. Daftar disimpan penuh di
    ``jawaban_variants`` (JSON list), sedangkan ``jawaban`` = variasi utama.
    """
    processed = 0
    for entry in entries:
        intent = str(entry["intent"])
        variants = normalize_variants(entry.get("jawaban"))
        row = session.scalar(
            select(db.ResponseEntry).where(db.ResponseEntry.intent == intent)
        )
        if row is None:
            row = db.ResponseEntry(intent=intent)
            session.add(row)
        row.deskripsi = str(entry.get("deskripsi", ""))
        row.jawaban = variants[0] if variants else ""
        row.jawaban_variants = json.dumps(variants, ensure_ascii=False)
        processed += 1
    session.commit()
    return processed


def main(argv: list[str] | None = None) -> int:
    """Titik masuk CLI seed jawaban baku."""
    parser = argparse.ArgumentParser(description="Seed jawaban baku FAQ ke database.")
    parser.add_argument("--answers", default=None, help="Berkas intents.yaml (default dari config).")
    parser.add_argument("--db", default=None, help="URL database (default dari config/.env).")
    parser.add_argument("--sandbox", action="store_true", help="Pakai jawaban & DB sandbox.")
    args = parser.parse_args(argv)

    cfg = data_io.load_config()
    if args.sandbox:
        base = data_io.resolve_path(cfg["paths"]["sandbox"]).parent
        answers = data_io.resolve_path(args.answers) if args.answers else base / "intents_sandbox.yaml"
        db_url = args.db or f"sqlite:///{(base / 'chatbot_sandbox.db').as_posix()}"
    else:
        answers = (
            data_io.resolve_path(args.answers)
            if args.answers
            else data_io.resolve_path(cfg["paths"]["answers"])
        )
        db_url = args.db or get_settings(cfg).database_url

    entries = load_answer_entries(answers)
    engine = db.make_engine(db_url)
    db.init_db(engine)
    factory = db.make_session_factory(engine)
    with factory() as session:
        seeded = seed_responses(session, entries)
        total = int(session.scalar(select(func.count()).select_from(db.ResponseEntry)) or 0)
        rows = session.scalars(select(db.ResponseEntry)).all()
        total_variants = sum(len(r.variants()) for r in rows)

    print(
        json.dumps(
            {
                "answers_file": str(answers),
                "seeded": seeded,
                "total_rows": total,
                "total_variants": total_variants,
                "database": db_url,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
