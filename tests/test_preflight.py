"""Uji pemeriksa pra-terbang deployment (``scripts/preflight.py``)."""

from __future__ import annotations

import sqlite3

from scripts.preflight import (
    check_answers_file,
    check_api_key,
    check_database,
    check_env_file,
    check_model_dir,
    check_vectorstore,
    render_report,
    run_checks,
)


def _make_env(path, value: str) -> None:
    path.write_text(
        "# contoh\nCHATBOT_API_KEY=" + value + "\nCORS_ORIGINS=http://x\n",
        encoding="utf-8",
    )


def _make_db(path, *, rows: int = 1, with_table: bool = True) -> None:
    connection = sqlite3.connect(path)
    try:
        if with_table:
            connection.execute("CREATE TABLE responses (intent TEXT, jawaban TEXT)")
            for i in range(rows):
                connection.execute(
                    "INSERT INTO responses (intent, jawaban) VALUES (?, ?)",
                    (f"i{i}", "j"),
                )
        connection.commit()
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# env & api key
# ---------------------------------------------------------------------------


def test_check_env_file(tmp_path):
    assert check_env_file(tmp_path / ".env").ok is False
    (tmp_path / ".env").write_text("A=1\n", encoding="utf-8")
    assert check_env_file(tmp_path / ".env").ok is True


def test_check_api_key_states(tmp_path):
    env = tmp_path / ".env"
    assert check_api_key(env).ok is False  # belum ada
    _make_env(env, "")
    assert check_api_key(env).ok is False  # kosong
    _make_env(env, "ganti-dengan-token-acak-panjang")
    assert check_api_key(env).ok is False  # masih placeholder
    _make_env(env, "rahasia-panjang-123")
    assert check_api_key(env).ok is True


# ---------------------------------------------------------------------------
# answers / model / vectorstore
# ---------------------------------------------------------------------------


def test_check_answers_file(tmp_path):
    assert check_answers_file(tmp_path / "intents.yaml").ok is False
    empty = tmp_path / "kosong.yaml"
    empty.write_text("[]\n", encoding="utf-8")
    assert check_answers_file(empty).ok is False
    good = tmp_path / "intents.yaml"
    good.write_text(
        "- intent: sapaan\n  jawaban: Halo\n- intent: terima_kasih\n  jawaban: Sama-sama\n",
        encoding="utf-8",
    )
    assert check_answers_file(good).ok is True


def test_check_model_dir(tmp_path):
    assert check_model_dir(tmp_path / "models" / "x").ok is False
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    assert check_model_dir(model).ok is False  # tokenizer kurang
    (model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    assert check_model_dir(model).ok is True
    (model / "label_map.json").write_text("{}", encoding="utf-8")
    assert "label_map.json" in check_model_dir(model).detail


def test_check_vectorstore(tmp_path):
    assert check_vectorstore(tmp_path / "vs").ok is False
    vs = tmp_path / "vs"
    vs.mkdir()
    (vs / "chunks.json").write_text("[]", encoding="utf-8")
    assert check_vectorstore(vs).ok is False  # meta.json kurang
    (vs / "meta.json").write_text("{}", encoding="utf-8")
    assert check_vectorstore(vs).detail == "bm25 saja"
    (vs / "embeddings.npy").write_bytes(b"x")
    assert check_vectorstore(vs).detail == "dense+bm25"


# ---------------------------------------------------------------------------
# database
# ---------------------------------------------------------------------------


def test_check_database(tmp_path):
    assert check_database(tmp_path / "tidak_ada.db").ok is False
    empty = tmp_path / "kosong.db"
    _make_db(empty, rows=0)
    assert check_database(empty).ok is False  # tabel kosong
    no_table = tmp_path / "tanpa_tabel.db"
    _make_db(no_table, with_table=False)
    assert check_database(no_table).ok is False  # tanpa tabel responses
    good = tmp_path / "chatbot.db"
    _make_db(good, rows=3)
    assert check_database(good).ok is True


# ---------------------------------------------------------------------------
# agregasi & laporan
# ---------------------------------------------------------------------------


def _ready_tree(tmp_path):
    env = tmp_path / ".env"
    _make_env(env, "rahasia")
    answers = tmp_path / "intents.yaml"
    answers.write_text("- intent: sapaan\n  jawaban: Halo\n", encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    vs = tmp_path / "vs"
    vs.mkdir()
    (vs / "chunks.json").write_text("[]", encoding="utf-8")
    (vs / "meta.json").write_text("{}", encoding="utf-8")
    db = tmp_path / "chatbot.db"
    _make_db(db, rows=2)
    return dict(
        env_path=env,
        answers_path=answers,
        model_dir=model,
        vectorstore_dir=vs,
        db_path=db,
    )


def test_run_checks_all_ready(tmp_path):
    checks = run_checks(**_ready_tree(tmp_path), require_env=True)
    assert all(c.ok for c in checks)
    assert "SIAP" in render_report(checks)


def test_run_checks_reports_failures(tmp_path):
    tree = _ready_tree(tmp_path)
    tree["db_path"] = tmp_path / "hilang.db"
    checks = run_checks(**tree, require_env=True)
    assert not all(c.ok for c in checks)
    report = render_report(checks)
    assert "[GAGAL] database" in report
    assert "BELUM SIAP" in report


def test_run_checks_skip_env_in_sandbox(tmp_path):
    tree = _ready_tree(tmp_path)
    (tmp_path / ".env").unlink()
    checks = run_checks(**tree, require_env=False)
    assert all(c.name not in {"env_file", "api_key"} for c in checks)
    assert all(c.ok for c in checks)  # artefak lengkap -> SIAP
    assert "SIAP" in render_report(checks)
