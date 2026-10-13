"""Pra-terbang (preflight) deployment: cek kesiapan berkas tanpa menjalankan layanan.

Tujuan: memastikan artefak yang dibutuhkan API sudah ada SEBELUM layanan
dinyalakan (kunci API, berkas jawaban, model, indeks retrieval, database).

Contoh:
    .venv\\Scripts\\python.exe -m scripts.preflight
    .venv\\Scripts\\python.exe -m scripts.preflight --sandbox
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from src import data_io

# Nilai contoh di .env.example; dianggap belum dikonfigurasi.
PLACEHOLDER_API_KEY = "ganti-dengan-token-acak-panjang"


@dataclass(frozen=True)
class Check:
    """Hasil satu pemeriksaan kesiapan."""

    name: str
    ok: bool
    detail: str


def _read_env(env_path: str | Path) -> dict[str, str]:
    """Baca ``.env`` sederhana menjadi peta kunci->nilai (tanpa efek samping)."""
    p = Path(env_path)
    if not p.exists():
        return {}
    values: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def check_env_file(env_path: str | Path) -> Check:
    """``.env`` ada?"""
    p = Path(env_path)
    if p.exists():
        return Check("env_file", True, str(p))
    return Check("env_file", False, f"tidak ada: {p} (salin dari .env.example)")


def check_api_key(env_path: str | Path) -> Check:
    """``CHATBOT_API_KEY`` terisi dan bukan nilai contoh?"""
    key = _read_env(env_path).get("CHATBOT_API_KEY", "")
    if not key:
        return Check("api_key", False, "CHATBOT_API_KEY belum diisi di .env")
    if key == PLACEHOLDER_API_KEY:
        return Check("api_key", False, "CHATBOT_API_KEY masih nilai contoh")
    return Check("api_key", True, f"terisi ({len(key)} karakter)")


def check_answers_file(path: str | Path) -> Check:
    """Berkas jawaban baku (``data/intents.yaml``) berisi minimal satu intent?"""
    p = Path(path)
    if not p.exists():
        return Check("answers_file", False, f"tidak ada: {p}")
    try:
        from src.answers import load_answers

        n = len(load_answers(p))
    except Exception as exc:  # pragma: no cover - jalur galat tak terduga
        return Check("answers_file", False, f"gagal dibaca: {exc}")
    if n <= 0:
        return Check("answers_file", False, "0 intent")
    return Check("answers_file", True, f"{n} intent")


def check_model_dir(model_dir: str | Path) -> Check:
    """Direktori model hasil fine-tuning lengkap (config + tokenizer)?"""
    p = Path(model_dir)
    if not p.exists():
        return Check("model_dir", False, f"tidak ada: {p}")
    required = ["config.json", "tokenizer_config.json"]
    missing = [f for f in required if not (p / f).exists()]
    if missing:
        return Check("model_dir", False, f"kurang: {', '.join(missing)}")
    extras = [f for f in ("label_map.json", "train_meta.json") if (p / f).exists()]
    note = ", ".join(extras) if extras else "tanpa label_map.json"
    return Check("model_dir", True, f"ok ({note})")


def check_vectorstore(vectorstore_dir: str | Path) -> Check:
    """Indeks retrieval (``chunks.json`` + ``meta.json``) tersedia?"""
    p = Path(vectorstore_dir)
    if not p.exists():
        return Check("vectorstore", False, f"tidak ada: {p}")
    required = ["chunks.json", "meta.json"]
    missing = [f for f in required if not (p / f).exists()]
    if missing:
        return Check("vectorstore", False, f"kurang: {', '.join(missing)}")
    mode = "dense+bm25" if (p / "embeddings.npy").exists() else "bm25 saja"
    return Check("vectorstore", True, mode)


def check_database(db_path: str | Path) -> Check:
    """Database ada dan tabel ``responses`` sudah terisi?"""
    p = Path(db_path)
    if not p.exists():
        return Check("database", False, f"tidak ada: {p} (jalankan scripts.seed_responses)")
    try:
        uri = f"file:{p.as_posix()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        try:
            count = connection.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
        finally:
            connection.close()
    except Exception as exc:
        return Check("database", False, f"gagal dibaca: {exc}")
    if count <= 0:
        return Check("database", False, "tabel responses kosong")
    return Check("database", True, f"{count} jawaban")


def run_checks(
    *,
    env_path: str | Path,
    answers_path: str | Path,
    model_dir: str | Path,
    vectorstore_dir: str | Path,
    db_path: str | Path,
    require_env: bool = True,
) -> list[Check]:
    """Jalankan semua pemeriksaan kesiapan (urutan tetap, mudah dibaca).

    ``require_env=False`` (mode sandbox) melewati pemeriksaan ``.env`` dan kunci
    API, karena tujuannya hanya memverifikasi artefak pipeline.
    """
    checks = [
        check_answers_file(answers_path),
        check_model_dir(model_dir),
        check_vectorstore(vectorstore_dir),
        check_database(db_path),
    ]
    if require_env:
        checks.insert(0, check_env_file(env_path))
        checks.insert(1, check_api_key(env_path))
    return checks


def render_report(checks: list[Check]) -> str:
    """Susun laporan teks dari hasil pemeriksaan."""
    lines = ["# Pra-terbang deployment", ""]
    for check in checks:
        mark = "OK" if check.ok else "GAGAL"
        lines.append(f"- [{mark}] {check.name}: {check.detail}")
    ready = all(check.ok for check in checks)
    lines += ["", f"Status: {'SIAP' if ready else 'BELUM SIAP'}"]
    return "\n".join(lines)


def _paths(cfg: dict, sandbox: bool) -> dict[str, Path]:
    """Tentukan jalur artefak yang diperiksa (asli atau sandbox)."""
    if sandbox:
        base = data_io.resolve_path(cfg["paths"]["sandbox"]).parent
        return {
            "answers_path": base / "intents_sandbox.yaml",
            "model_dir": data_io.resolve_path(cfg["paths"]["models_dir"]) / "smoke",
            "vectorstore_dir": base / "vectorstore_dense",
            "db_path": base / "chatbot_sandbox.db",
        }

    from src.train import model_slug

    return {
        "answers_path": data_io.resolve_path(cfg["paths"]["answers"]),
        "model_dir": data_io.resolve_path(cfg["paths"]["models_dir"])
        / model_slug(cfg["models"]["primary"]),
        "vectorstore_dir": data_io.resolve_path(cfg["paths"]["vectorstore_dir"]),
        "db_path": data_io.resolve_path(cfg["paths"]["database"]),
    }


def main(argv: list[str] | None = None) -> int:
    """Titik masuk CLI: cetak laporan; keluar kode != 0 bila belum siap."""
    parser = argparse.ArgumentParser(
        description="Cek kesiapan deployment (tanpa menjalankan layanan)."
    )
    parser.add_argument("--sandbox", action="store_true", help="Periksa artefak sandbox.")
    parser.add_argument("--env", default=None, help="Berkas .env (default: <root>/.env).")
    args = parser.parse_args(argv)

    cfg = data_io.load_config()
    env_path = Path(args.env) if args.env else data_io.PROJECT_ROOT / ".env"
    paths = _paths(cfg, args.sandbox)
    checks = run_checks(
        env_path=env_path,
        require_env=not args.sandbox,
        **paths,
    )
    print(render_report(checks))
    return 0 if all(check.ok for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
