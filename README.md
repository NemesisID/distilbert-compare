# chatbot-intent

Modul chatbot **pengganti FAQ** untuk website Sistem Informasi Inkubator Bisnis
UPN "Veteran" Jawa Timur. Terdiri dari:

1. **Intent classification** — fine-tuning DistilBERT untuk mengklasifikasi
   pertanyaan bahasa Indonesia ke salah satu intent; jawaban baku diambil dari
   database (bukan hasil generasi model).
2. **Retrieval dokumen PDF mitra** — menjawab pertanyaan isi dokumen dengan
   kutipan teks + sumber (nama dokumen, halaman). Tidak ada teks generatif.
3. **Layanan REST (FastAPI)** yang dipanggil oleh backend website utama.

> Chatbot ini **bukan** chatbot generatif. Dilarang memakai LLM/IndoBERT/
> arsitektur lain di luar lingkup yang disepakati.

---

## Prasyarat

| Komponen | Wajib |
|----------|-------|
| OS | Windows 10/11 (64-bit) |
| Python | **3.10 – 3.12 (64-bit)** |
| Git | ada |
| GPU | opsional (CPU pun jalan) |

> **PENTING — kenapa bukan Python 3.13+:** PyTorch dan PyYAML belum
> menerbitkan wheel untuk Python yang lebih baru (diuji: Python 3.15 gagal,
> PyTorch tidak punya wheel sama sekali dan PyYAML butuh compiler C).
> Gunakan Python 3.12.

---

## Setup

```powershell
# 1. Buat virtual environment (pakai Python 3.12)
py -3.12 -m venv .venv

# 2. Aktifkan. Bila diblokir ExecutionPolicy, minta izin dulu ke pemilik proyek.
.\.venv\Scripts\Activate.ps1
#   Alternatif tanpa mengubah policy:
#   & .\.venv\Scripts\python.exe -m pip ...

# 3. Pastikan UTF-8 (penting untuk bahasa Indonesia)
$env:PYTHONUTF8 = 1

# 4. Pasang dependensi
pip install -r requirements.txt

# 5. Pasang PyTorch (pilih salah satu sesuai GPU)
#    CPU (tidak ada GPU NVIDIA):
pip install torch --index-url https://download.pytorch.org/whl/cpu
#    CUDA (ada GPU NVIDIA):
# pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Salin `.env.example` menjadi `.env` lalu isi nilainya. **`.env` tidak pernah
di-commit.**

### Versi library terpasang

Daftar lengkap (`pip freeze`) ada di `reports/library_versions.txt`. Versi kunci:

| Paket | Versi |
|-------|-------|
| Python | 3.12.10 (64-bit) |
| torch | 2.14.1+cpu (**CPU-only**, `cuda.is_available()=False`) |
| transformers | 5.19.0 |
| sentence-transformers | 6.1.0 |
| datasets / evaluate / accelerate | 5.1.0 / 0.4.6 / 1.15.0 |
| scikit-learn | 1.9.1 |
| pandas / numpy | 3.0.6 / 2.5.4 |
| PyMuPDF (pymupdf) | 1.28.2 |
| FastAPI / SQLAlchemy / pydantic | 0.143.0 / 2.1.4 / 2.14.1 |
| rank-bm25 | 0.2.2 |
| python-dotenv | 1.2.4 |
| pytest | 9.1.1 |

> Catatan: `transformers` sudah versi **5.x** — nama argumen Trainer berbeda
> dari contoh tutorial lama. Fase 3 sudah menyesuaikan: `eval_strategy`,
> `processing_class` (bukan `tokenizer`), `warmup_steps` (karena `warmup_ratio`
> sudah dihapus), serta `EvalPrediction.predictions` yang dapat berupa tuple.

---

## Struktur repositori

```
chatbot-intent/
├── config.yaml            # semua parameter (tanpa angka ajaib di kode)
├── EXPERIMENTS.md         # log eksperimen
├── requirements.txt
├── .env.example
├── conftest.py
├── data/
│   ├── sandbox/           # data sintetis (hanya uji pipeline)
│   ├── raw/               # intents_raw.csv (dari pemilik proyek)
│   ├── processed/         # train/val/test.csv + TEST_SHA256.txt
│   ├── augmented/         # train_aug.csv
│   ├── pdf/               # PDF mitra (rahasia)
│   ├── rag_eval/          # qa_pairs / in_doc / out_doc
│   └── e2e/               # mixed_val.csv / mixed_test.csv
├── src/                   # kode inti (data_io, split, augment, train, calibrate, router, ...)
├── rag/                   # pipeline PDF: pdf_extract, chunking, index, retrieve, eval_retrieval
├── api/                   # FastAPI (settings, database, security, schemas, main)
├── scripts/               # seed_responses.py, make_sandbox_rag.py
├── deploy/                # contoh PS1 + pra-terbang + templat layanan Windows (belum dijalankan)
├── models/                # hasil fine-tuning (git-ignored)
├── vectorstore/           # indeks vektor (git-ignored)
├── reports/               # laporan
├── notebooks/             # notebook eksplorasi (memanggil src/)
└── tests/
```

Berkas `*.template.csv` / `*.template.yaml` adalah **templat**; salin ke nama
tanpa `.template` lalu isi dengan data asli.

---

## Perintah umum

```powershell
$env:PYTHONUTF8 = 1

# Validasi data (default: data/raw/intents_raw.csv)
python -m src.validate_data
python -m src.validate_data --sandbox --out reports/sandbox_data_validation.md

# Split 70/15/15 + kunci test set
python -m src.split
python -m src.split --sandbox --out-dir data/sandbox/processed

# Fase 2: augmentasi (train), cek kebocoran, baseline
python -m src.augment            # -> data/augmented/train_aug.csv
python -m src.leakage_check      # -> reports/leakage_report.md (exit!=0 bila bocor)
python -m src.baseline           # -> reports/baseline_val.md + EXPERIMENTS.md
#   (tambahkan --sandbox untuk menguji alur memakai data sintetis)

# Fase 3: fine-tuning DistilBERT (CPU-only -> pakai --grid reduced)
python -m src.train --mode single --grid reduced --model cahya/distilbert-base-indonesian
python -m src.train --mode search --grid reduced                      # grid diperkecil di val
python -m src.train --mode multiseed --grid reduced --lr 2e-5 --batch 16  # 3 seed (42,43,44)
#   (--grid full = grid penuh dari config; --sandbox = uji pipeline di data sintetis)
#   (--output models/<nama> ; model terbaik disalin ke sana + selected.json)
#   (grid "reduced" diatur di config.yaml -> training.reduced_grid)

# Fase 4: kalibrasi suhu & ambang tau1 (HANYA val)
python -m src.calibrate --model-dir models/distilbert-base-indonesian
#   (--sandbox untuk uji pipeline -> reports/sandbox_calibration_val.md)
#   (menghasilkan reports/calibration_val.md + models/<nama>/calibration.json)

# Fase 5: pipeline PDF/retrieval
python -m rag.index --pdf-dir data/pdf --out vectorstore          # bangun indeks (dense + BM25)
python -m rag.index --pdf-dir data/pdf --out vectorstore --no-dense   # BM25 saja (cepat)
python -m rag.eval_retrieval --method rrf                         # evaluasi + pilih tau2 (data rag_eval)
#   (--sandbox untuk uji pipeline di data sintetis)
#   (uji sandbox: python -m scripts.make_sandbox_rag lalu arahkan --pdf-dir/--vectorstore ke data/sandbox/)

# Fase 6: routing intent + retrieval -> faq | dokumen | fallback (evaluasi val)
python -m src.router                                         # evaluasi data/e2e/mixed_val.csv
python -m src.router --ask "berapa kuota peserta angkatan"   # cek satu pertanyaan
#   (--sandbox untuk uji pipeline; --tau1/--tau2 hanya untuk diagnostik)
#   (menghasilkan reports/e2e_val.md dan mencatat ke EXPERIMENTS.md)

# Fase 7: API & database (FastAPI + SQLite)
python -m scripts.seed_responses                         # isi jawaban baku dari data/intents.yaml
python -m scripts.seed_responses --sandbox               # uji pipeline (jawaban + DB sandbox)
#   Jalankan layanan (butuh .env -> CHATBOT_API_KEY). Hentikan manual; JANGAN jadikan
#   layanan permanen:
#   uvicorn api.main:app --host 127.0.0.1 --port 8000
#   Endpoint: GET /health (publik) | GET /intents | POST /chat  (header X-API-KEY)

# Fase 8: pra-terbang & deployment (berkas contoh; layanan TIDAK dijalankan otomatis)
python -m scripts.preflight              # cek kesiapan artefak (keluar != 0 bila belum siap)
python -m scripts.preflight --sandbox    # uji pipeline pada artefak sandbox
#   Uji layanan (latar depan, hentikan manual): .\deploy\run_api.ps1
#   Cek sehat:          .\deploy\healthcheck.ps1
#   Siapkan artefak:    .\deploy\build_artifacts.ps1   (data asli)
#   Cadangkan DB:       .\deploy\backup_db.ps1
#   Templat layanan Windows: deploy\windows-service\  (JANGAN dipasang tanpa izin)

# Tes
python -m pytest -q
```

---

## Penjaga test set

- Test set (`test.csv`, `mixed_test.csv`) **hanya** boleh dibaca oleh
  `src/final_eval.py`.
- Semua modul memuat data lewat `src/data_io.py`, yang **menolak** membaca test
  set kecuali variabel lingkungan `ALLOW_TEST=1` diset (hanya `final_eval.py`
  yang menyetelnya).
- Test set dibuat **read-only** (`attrib +R`) dan SHA-256-nya dicatat di
  `TEST_SHA256.txt`.

---

## Catatan lingkungan

- Selalu gunakan `encoding='utf-8'` dan `$env:PYTHONUTF8 = 1`.
- `dataloader_num_workers=0` (multiprocessing Windows memakai *spawn*).
- Batasi utas dengan `torch.set_num_threads` sesuai `config.yaml`.
- Di server lab bersama: jangan pakai seluruh core/VRAM, jangan jalankan
  layanan permanen, jangan ubah firewall/IIS tanpa izin.

---

## Status fase

| Fase | Status |
|------|--------|
| 0 – Persiapan | ✅ selesai (Python 3.12.10, torch 2.14.1+cpu, CPU-only) |
| 1 – Kerangka & data | ✅ selesai (22 tes lulus) |
| 2 – Augmentasi/leakage/baseline | ✅ kode & uji sandbox (39 tes lulus) |
| 3 – Fine-tuning DistilBERT | ✅ kode & uji sandbox (66 tes lulus; single/search/multiseed) |
| 4 – Kalibrasi & tau1 | ✅ kode & uji sandbox (temperature scaling + tau1 pada val; 77 tes lulus) |
| 5 – Pipeline PDF | ✅ kode & uji sandbox (pdf_extract/chunking/index/retrieve/eval_retrieval; 109 tes lulus) |
| 6 – Routing & e2e | ✅ kode & uji sandbox (src/router.py: faq/dokumen/fallback; 123 tes lulus) |
| 7 – API & database | ✅ kode & uji sandbox (FastAPI /health,/intents,/chat + SQLite + kunci X-API-KEY; 140 tes lulus) |
| 8 – Deployment (berkas saja) | ✅ kode & uji sandbox (contoh PS1 + preflight + templat layanan; 156 tes lulus) |
| 9 – Evaluasi final | hanya atas perintah eksplisit |
