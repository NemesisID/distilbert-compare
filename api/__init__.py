"""Layanan REST (FastAPI) untuk chatbot inkubator.

Modul:
- ``settings`` : gabungan ``config.yaml`` + ``.env`` (kunci API, DB, CORS, rate limit).
- ``database`` : model SQLAlchemy + mesin/sesi SQLite.
- ``security`` : verifikasi kunci API (``X-API-KEY``) + pembatas laju sederhana.
- ``schemas``  : model Pydantic permintaan/balasan.
- ``main``     : aplikasi FastAPI (``/health``, ``/intents``, ``/chat``).
"""
