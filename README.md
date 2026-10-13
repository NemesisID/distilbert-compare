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
├── src/                   # kode inti
├── rag/                   # pipeline PDF
├── api/                   # FastAPI
├── scripts/               # seed_responses.py
├── deploy/                # contoh berkas layanan Windows (belum dijalankan)
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
| 0 – Persiapan | ⏳ menunggu Python 3.12 dipasang |
| 1 – Kerangka & data | ✅ kode + tes siap (22 tes lulus) |
| 2 – Augmentasi/leakage/baseline | belum dimulai |
| 3 – Fine-tuning DistilBERT | belum dimulai |
| 4 – Kalibrasi & tau1 | belum dimulai |
| 5 – Pipeline PDF | belum dimulai |
| 6 – Routing & e2e | belum dimulai |
| 7 – API & database | belum dimulai |
| 8 – Deployment (berkas saja) | belum dimulai |
| 9 – Evaluasi final | hanya atas perintah eksplisit |
