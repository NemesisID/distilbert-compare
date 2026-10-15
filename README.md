# chatbot-intent

Modul **chatbot pengganti FAQ** untuk website Sistem Informasi **Inkubator
Bisnis UPN "Veteran" Jawa Timur**.

Chatbot ini menjawab pertanyaan pengguna dengan salah satu dari tiga cara:
**jawaban baku FAQ**, **kutipan dari dokumen PDF mitra**, atau **menyerah dengan
sopan (fallback)** bila tidak yakin. Semua keputusan diambil oleh model
klasifikasi + pencarian dokumen — **tidak ada teks yang dikarang AI**.

> **Chatbot ini BUKAN chatbot generatif.** Dilarang memakai LLM / IndoBERT /
> arsitektur lain di luar lingkup yang disepakati (lihat [Lingkup model](#3-lingkup--batasan-model)).

**Baru di proyek ini?** Mulai dari **[`PANDUAN.md`](PANDUAN.md)**
(langkah-demi-langkah, ramah pemula) dan **[`PANDUAN_ISI_DATA.md`](PANDUAN_ISI_DATA.md)**
(petunjuk detail mengisi & mengoreksi berkas data).

---

## Daftar isi

1. [Tentang proyek](#1-tentang-proyek)
2. [Cara kerja (alur sistem)](#2-cara-kerja-alur-sistem)
3. [Lingkup & batasan model](#3-lingkup--batasan-model)
4. [Fitur utama](#4-fitur-utama)
5. [Teknologi yang dipakai](#5-teknologi-yang-dipakai)
6. [Struktur repositori](#6-struktur-repositori)
7. [Prasyarat](#7-prasyarat)
8. [Setup](#8-setup)
9. [Konfigurasi](#9-konfigurasi)
10. [Data yang dibutuhkan](#10-data-yang-dibutuhkan)
11. [Menjalankan pipeline (per fase)](#11-menjalankan-pipeline-per-fase)
12. [Metodologi evaluasi](#12-metodologi-evaluasi)
13. [Layanan API](#13-layanan-api)
14. [Pengujian](#14-pengujian)
15. [Dokumentasi](#15-dokumentasi)
16. [Catatan lingkungan](#16-catatan-lingkungan)
17. [Status fase](#17-status-fase)

---

## 1. Tentang proyek

FAQ (daftar pertanyaan-jawaban) pada website inkubator sulit dipakai: pengunjung
harus membaca daftar panjang dan menebak kata kunci. Proyek ini menggantinya
dengan chatbot yang:

- memahami **maksud** pertanyaan pengguna (bukan sekadar cocok kata kunci),
  lewat model **DistilBERT** yang dilatih khusus dengan data inkubator;
- menjawab dengan **jawaban baku resmi** untuk pertanyaan umum — satu intent
  boleh punya **beberapa variasi jawaban** supaya tidak monoton;
- mengambil **kutipan dari dokumen PDF** (panduan, jadwal, ketentuan) beserta
  sumber nama dokumen + halaman, untuk pertanyaan isi dokumen;
- **tidak asal menjawab**: kalau tidak yakin atau pertanyaan di luar cakupan, ia
  menjawab *fallback* dan menyarankan menghubungi admin;
- disediakan sebagai **layanan REST (FastAPI)** yang dipanggil oleh backend
  website utama.

**Tujuan akademik:** membandingkan beberapa pendekatan (DistilBERT Indonesia vs
DistilBERT multilingual vs baseline TF-IDF; pencarian vektor/dense vs BM25)
secara **jujur dan dapat diulang** — dengan test set yang dikunci dan dilaporkan
sekali saja.

---

## 2. Cara kerja (alur sistem)

### Saat pengguna bertanya (online)

```
Backend website utama
        │  POST /chat  (header: X-API-KEY)
        ▼
┌──────────────────────────────────────────────────────────────┐
│ 1. Klasifikasi intent (DistilBERT)                            │
│    → probabilitas TERKALIBRASI (temperature scaling)          │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. Routing (aturan)                                           │
│                                                               │
│   intent = out_of_scope  ──────────────► fallback             │
│   keyakinan < tau1        ──────────────► fallback            │
│   intent = tanya_dokumen ──► cari di PDF (dense/BM25/RRF)     │
│         └ skor < tau2    ──────────────► fallback             │
│         └ skor ≥ tau2    ──────────────► dokumen (+ kutipan)  │
│   intent lain            ──────────────► faq (baku + variasi) │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
  Jawaban: { route: faq | dokumen | fallback, jawaban/kutipan, sumber, skor }
```

- **tau1** = ambang keyakinan intent (dihitung dari data validasi).
- **tau2** = ambang kecocokan dokumen (dihitung dari data evaluasi retrieval).
- Jawaban baku FAQ disimpan di database (di-seed dari `data/intents.yaml`),
  **bukan** disimpan di dalam model. Satu intent boleh punya **banyak variasi
  jawaban** yang dipilih acak tiap permintaan (`routing.answer_variants`).

### Saat menyiapkan model (offline, per fase)

```
data mentah → validasi → split (kunci test) → augmentasi(train) → cek kebocoran
   → baseline (TF-IDF)  →  fine-tuning DistilBERT  →  kalibrasi (T & tau1)
   → indeks PDF (dense+BM25)  →  evaluasi retrieval (pilih tau2)
   → routing e2e (val)  →  seed jawaban ke DB  →  preflight
   → [JALANKAN EVALUASI FINAL]  (buka test set SEKALI)
```

---

## 3. Lingkup & batasan model

Lingkup ini **keras** — jangan diubah tanpa persetujuan pemilik proyek.

| Peran | Model / metode | Catatan |
|-------|----------------|---------|
| Klasifikasi intent (primer) | `cahya/distilbert-base-indonesian` | uncased; bahasa Indonesia |
| Klasifikasi intent (pembanding) | `distilbert/distilbert-base-multilingual-cased` | cased; multilingual |
| Embedding retrieval | `sentence-transformers/distiluse-base-multilingual-cased-v2` | maks 128 token |
| Retrieval kata kunci | **BM25** (`rank_bm25`) | pembanding embedding |
| Baseline klasik | **TF-IDF + LinearSVC** | pembanding utama saat ini |

Aturan:
- **Tidak** memakai LLM, IndoBERT, GPT, atau arsitektur generatif lain.
- Jawaban FAQ berasal dari **data** (`data/intents.yaml` → database), bukan
  dihasilkan model.
- **Banyak variasi jawaban bukan berarti generatif**: daftar variasi tetap teks
  baku yang ditulis manusia; model hanya memilih salah satu secara acak.
- Semua parameter ada di **`config.yaml`** (tidak ada angka ajaib di kode).

---

## 4. Fitur utama

- **Deteksi percakapan di luar cakupan** (`out_of_scope` → fallback).
- **Jawaban baku yang luwes**: satu intent boleh punya beberapa variasi jawaban
  (dipilih acak tiap permintaan) — tetap non-generatif, tidak monoton.
- **Ambang keyakinan** (tau1) + **kalibrasi suhu** agar probabilitas model jujur.
- **Pencarian dokumen PDF** dengan dua metode + penggabungan **RRF**.
- **Kutipan bersumber**: jawaban dokumen menyertakan nama dokumen & halaman.
- **Ambang kecocokan dokumen** (tau2) supaya tidak memaksa mengutip dokumen yang
  tidak relevan.
- **API terkunci**: wajib header `X-API-KEY`, ada pembatasan laju (rate limit).
- **Reproducible**: seed tetap (42, 43, 44), log otomatis ke `EXPERIMENTS.md`.
- **Test set terkunci**: hanya boleh dibaca sekali saat evaluasi final.
- **Data sandbox sintetis** untuk menguji seluruh pipeline tanpa data asli.

---

## 5. Teknologi yang dipakai

| Lapisan | Teknologi |
|---------|-----------|
| Klasifikasi teks | PyTorch + Hugging Face Transformers (DistilBERT) |
| Embedding / retrieval | sentence-transformers, rank-bm25, numpy |
| Ekstraksi PDF | PyMuPDF (`import pymupdf`) |
| API | FastAPI + Uvicorn |
| Database | SQLAlchemy + SQLite |
| Keamanan | header `X-API-KEY` + rate limit sederhana |
| Data & metrik | pandas, scikit-learn |
| Konfigurasi | `config.yaml` + `.env` (python-dotenv) |
| Pengujian | pytest (177 tes) |

Versi terpasang (ringkas) — daftar lengkap di `reports/library_versions.txt`:

| Paket | Versi |
|-------|-------|
| Python | 3.12.10 (64-bit) |
| torch | 2.14.1+cpu (**CPU-only**, `cuda.is_available()=False`) |
| transformers | 5.19.0 |
| sentence-transformers | 6.1.0 |
| scikit-learn | 1.9.1 |
| pandas / numpy | 3.0.6 / 2.5.4 |
| PyMuPDF (pymupdf) | 1.28.2 |
| FastAPI / SQLAlchemy / pydantic | 0.143.0 / 2.1.4 / 2.14.1 |
| rank-bm25 | 0.2.2 |
| python-dotenv / pytest | 1.2.4 / 9.1.1 |

> **Catatan `transformers` 5.x:** nama argumen Trainer berbeda dari tutorial
> lama. Kode sudah menyesuaikan: `eval_strategy`, `processing_class` (bukan
> `tokenizer`), `warmup_steps` (karena `warmup_ratio` dihapus), dan
> `EvalPrediction.predictions` yang dapat berupa tuple.

---

## 6. Struktur repositori

```
chatbot-intent/
├── config.yaml            # SEMUA parameter (tanpa angka ajaib di kode)
├── PANDUAN.md             # panduan langkah-demi-langkah (ramah pemula)
├── PANDUAN_ISI_DATA.md    # petunjuk detail pengisian & koreksi berkas data
├── EXPERIMENTS.md         # log eksperimen (otomatis + catatan per fase)
├── README.md
├── requirements.txt
├── .env.example           # salin jadi .env lalu isi
├── conftest.py
├── data/
│   ├── raw/               # intents_raw.csv  (data asli dari pemilik proyek)
│   ├── processed/         # train/val/test.csv + TEST_SHA256.txt  (hasil split)
│   ├── augmented/         # train_aug.csv  (hasil augmentasi train saja)
│   ├── pdf/               # PDF mitra (rahasia)
│   ├── rag_eval/          # qa_pairs / in_doc / out_doc
│   ├── e2e/               # mixed_val.csv / mixed_test.csv
│   ├── intents.yaml       # jawaban baku chatbot (dari template)
│   ├── augment_dictionary.yaml
│   └── sandbox/           # data sintetis untuk uji pipeline (BUKAN hasil penelitian)
├── src/                   # kode inti
│   ├── data_io.py         # SATU-SATUNYA pemuat data + penjaga test set
│   ├── validate_data.py   # validasi data latih
│   ├── split.py           # split 70/15/15 + kunci test set
│   ├── augment.py         # augmentasi (train saja)
│   ├── leakage_check.py   # cek kebocoran train vs val
│   ├── baseline.py        # baseline TF-IDF + LinearSVC
│   ├── train.py           # fine-tuning (single/search/multiseed)
│   ├── calibrate.py       # temperature scaling + tau1 (val)
│   ├── router.py          # routing faq/dokumen/fallback
│   ├── answers.py         # pemuat jawaban baku (boleh banyak variasi)
│   ├── final_eval.py      # evaluasi FINAL (satu-satunya pembaca test set)
│   └── experiment_log.py  # pencatat otomatis ke EXPERIMENTS.md
├── rag/                   # pipeline PDF: pdf_extract, chunking, index, retrieve, eval_retrieval
├── api/                   # FastAPI: settings, database, security, schemas, main
├── scripts/               # seed_responses.py, preflight.py, make_sandbox_rag.py
├── deploy/                # contoh PS1 + templat layanan Windows (belum dipasang)
├── models/                # hasil fine-tuning (git-ignored)
├── vectorstore/           # indeks vektor (git-ignored)
├── reports/               # semua laporan Markdown
├── notebooks/             # notebook eksplorasi (memanggil src/)
└── tests/                 # 177 tes (pytest)
```

Berkas `*.template.csv` / `*.template.yaml` adalah **templat**: salin ke nama
tanpa `.template`, lalu isi data asli.

---

## 7. Prasyarat

| Komponen | Wajib |
|----------|-------|
| OS | Windows 10/11 (64-bit) |
| Python | **3.10 – 3.12 (64-bit)** — diuji 3.12.10 |
| Git | ada |
| GPU | opsional (CPU pun jalan; proyek ini diuji **CPU-only**) |
| Internet | hanya untuk unduh model sekali (sekarang sudah ada di lokal) |

> **Kenapa bukan Python 3.13+?** PyTorch & PyYAML belum menerbitkan wheel untuk
> versi lebih baru (diuji: Python 3.15 gagal — PyTorch tanpa wheel, PyYAML butuh
> compiler C). Gunakan **Python 3.12**.

---

## 8. Setup

```powershell
# 1. Buat virtual environment (pakai Python 3.12)
py -3.12 -m venv .venv
#   Bila perintah "py" memanggil versi Python yang salah, pakai jalur penuh, mis.:
#   "C:\Users\<Anda>\AppData\Local\Programs\Python\Python312\python.exe" -m venv .venv

# 2. Aktifkan. Bila diblokir ExecutionPolicy, cukup pakai jalur python.exe langsung.
.\.venv\Scripts\Activate.ps1
#   Alternatif tanpa mengubah policy:
#   & .\.venv\Scripts\python.exe -m pip ...

# 3. Pastikan UTF-8 (penting untuk bahasa Indonesia)
$env:PYTHONUTF8 = 1

# 4. Pasang dependensi
pip install -r requirements.txt

# 5. Pasang PyTorch (pilih salah satu)
#    CPU (tanpa GPU NVIDIA):
pip install torch --index-url https://download.pytorch.org/whl/cpu
#    CUDA (ada GPU NVIDIA):
# pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Salin `.env.example` menjadi `.env`, lalu isi nilainya. **`.env` tidak pernah
di-commit.**

Model AI yang dipakai sudah **terunduh di cache lokal** (tidak perlu unduh lagi):
`cahya/distilbert-base-indonesian`, `distilbert/distilbert-base-multilingual-cased`,
dan `distiluse-base-multilingual-cased-v2`.

---

## 9. Konfigurasi

### `config.yaml` — semua parameter proyek

| Bagian | Isi |
|--------|-----|
| `project` | seed utama (42) dan daftar seed (42, 43, 44) |
| `paths` | lokasi semua berkas (data, model, vectorstore, laporan) |
| `intents` | **11 intent** (urutan dipakai untuk label model) |
| `intent_min_counts` | minimal contoh per intent (default 20; `tanya_dokumen` 40; `out_of_scope` 100) |
| `validation` | panjang teks wajar (min 3, maks 300 karakter) |
| `augmentation` | kamus augmentasi, maks varian, maks penggantian kata |
| `leakage` | ambang cosine pelaporan kebocoran (0.85) |
| `baseline` | TF-IDF + LinearSVC (ngram, C, max_iter) |
| `split` | 70% / 15% / 15%, `random_state: 42` |
| `models` | model primer / pembanding / embedding (lingkup keras) |
| `training` | `max_length`, early stopping, grid hyperparameter |
| `calibration` | pengaturan suhu & pencarian tau1 |
| `rag` | panjang chunk (90 kata, overlap 20), `top_k`, RRF, ambang tau2 |
| `routing` | aturan route, intent OOS, pesan fallback, variasi jawaban (`answer_variants`, `variant_seed`) |
| `api` | header kunci, rate limit, CORS |

### `.env` — rahasia (tidak di-commit)

| Kunci | Arti |
|-------|------|
| `CHATBOT_API_KEY` | kunci yang dikirim backend website via header `X-API-KEY` |
| `DATABASE_URL` | lokasi database (default SQLite `data/chatbot.db`) |
| `CORS_ORIGINS` | origin website yang diizinkan, dipisah koma |
| `RATE_LIMIT_PER_MINUTE` | batas permintaan per menit per kunci/IP |
| `HF_HUB_OFFLINE` | opsional: `1` = pakai model dari cache lokal saja |

---

## 10. Data yang dibutuhkan

**Belum ada data asli?** Itu wajar — berkas berikut harus disiapkan pemilik
proyek. **Petunjuk detail tiap berkas ada di [`PANDUAN_ISI_DATA.md`](PANDUAN_ISI_DATA.md).**

| Berkas | Untuk apa |
|--------|-----------|
| `data/raw/intents_raw.csv` | data latih: pertanyaan + label intent |
| `data/intents.yaml` | jawaban baku chatbot per intent (boleh banyak variasi) |
| `data/pdf/*.pdf` | dokumen mitra (basis jawaban dokumen) |
| `data/rag_eval/qa_pairs.csv` | uji pencarian: pertanyaan + dokumen + halaman |
| `data/rag_eval/in_doc.csv` | pertanyaan yang jawabannya ada di PDF |
| `data/rag_eval/out_doc.csv` | pertanyaan yang jawabannya tidak ada di PDF |
| `data/e2e/mixed_val.csv` | uji routing pada validasi |
| `data/e2e/mixed_test.csv` | uji routing **final** (test set terkunci) |

Bentuk `data/intents.yaml` — kunci `jawaban` menerima **satu teks** atau
**daftar teks** (variasi yang dipilih acak tiap permintaan):

```yaml
- intent: syarat_pendaftaran
  deskripsi: Pertanyaan tentang syarat pendaftaran.
  jawaban: "Satu jawaban resmi yang selalu sama."

- intent: sapaan
  deskripsi: Sapaan pembuka.
  jawaban:
    - "Halo! Ada yang bisa saya bantu terkait program inkubator?"
    - "Selamat datang! Silakan tanyakan seputar program inkubator."
```

> Tanpa data asli, seluruh pipeline tetap bisa diuji memakai **data sandbox
> sintetis** (`--sandbox`). Hasil sandbox **berlabel `[SANDBOX]`** dan **bukan**
> hasil penelitian.

---

## 11. Menjalankan pipeline (per fase)

```powershell
$env:PYTHONUTF8 = 1
```

**Fase 1 — validasi & split**
```powershell
& ".\.venv\Scripts\python.exe" -m src.validate_data          # -> reports/data_validation.md
& ".\.venv\Scripts\python.exe" -m src.split                  # -> data/processed/* + kunci test
```

**Fase 2 — augmentasi, kebocoran, baseline**
```powershell
& ".\.venv\Scripts\python.exe" -m src.augment                # -> data/augmented/train_aug.csv
& ".\.venv\Scripts\python.exe" -m src.leakage_check          # exit != 0 bila bocor
& ".\.venv\Scripts\python.exe" -m src.baseline               # -> reports/baseline_val.md
```

**Fase 3 — fine-tuning DistilBERT**
```powershell
& ".\.venv\Scripts\python.exe" -m src.train --mode search    --grid reduced
& ".\.venv\Scripts\python.exe" -m src.train --mode multiseed --grid reduced --lr 2e-5 --batch 16
#   --grid full = grid penuh; --sandbox = uji pipeline
#   --model <nama-hf> untuk model pembanding
#   Hasil: models/distilbert-base-indonesian/ (+ selected.json)
```

**Fase 4 — kalibrasi (val)**
```powershell
& ".\.venv\Scripts\python.exe" -m src.calibrate --model-dir models/distilbert-base-indonesian
#   -> reports/calibration_val.md + models/<nama>/calibration.json
```

**Fase 5 — indeks PDF & evaluasi retrieval**
```powershell
& ".\.venv\Scripts\python.exe" -m rag.index --pdf-dir data/pdf --out vectorstore
& ".\.venv\Scripts\python.exe" -m rag.eval_retrieval --method rrf     # -> pilih tau2
#   --no-dense untuk BM25 saja (lebih cepat)
```

**Fase 6 — routing e2e (val)**
```powershell
& ".\.venv\Scripts\python.exe" -m src.router                 # -> reports/e2e_val.md
& ".\.venv\Scripts\python.exe" -m src.router --ask "berapa kuota peserta angkatan"
```

**Fase 7 — jawaban ke database & layanan API**
```powershell
& ".\.venv\Scripts\python.exe" -m scripts.seed_responses     # isi jawaban baku ke DB
#   Jalankan layanan (butuh .env) — hentikan manual, jangan jadikan permanen:
#   uvicorn api.main:app --host 127.0.0.1 --port 8000
```

**Fase 8 — pra-terbang & deployment (berkas saja)**
```powershell
& ".\.venv\Scripts\python.exe" -m scripts.preflight          # harus "SIAP"
.\deploy\run_api.ps1              # jalankan layanan uji
.\deploy\build_artifacts.ps1      # jalankan urutan Fase 1–7 otomatis (data asli)
.\deploy\backup_db.ps1            # cadangkan database
```

**Fase 9 — evaluasi FINAL (test set terkunci, sekali saja)**
```powershell
& ".\.venv\Scripts\python.exe" -m src.final_eval --confirm "JALANKAN EVALUASI FINAL"
#   tanpa frasa persis -> keluar kode 2 dan test set tidak disentuh
#   -> reports/final_eval.md
```

> Untuk menguji seluruh alur **tanpa data asli**, tambahkan `--sandbox` pada
> perintah yang mendukung. Hasilnya diberi tag `[SANDBOX]`.

---

## 12. Metodologi evaluasi

- **Split 70/15/15** dilakukan **sebelum** augmentasi (train / val / test).
- **Augmentasi hanya pada data train**; `leakage_check` memastikan tidak ada
  kebocoran train↔val.
- **Test set dikunci**: dibuat read-only (`attrib +R`), SHA-256 dicatat di
  `TEST_SHA256.txt`, dan hanya `src/final_eval.py` yang boleh membacanya.
- **Penyetelan hyperparameter & ambang** (tau1, tau2) **hanya memakai val**
  / data eval retrieval — tidak pernah menyentuh test.
- **Kalibrasi**: temperature scaling + pemilihan tau1 (val).
- **Seed tetap** (42, 43, 44) untuk hasil yang dapat diulang.
- **Jawaban deterministik bila diperlukan**: `routing.answer_variants: false`
  membuat setiap intent selalu memakai jawaban utama (variasi pertama).
- **Model pembanding**: DistilBERT multilingual dan baseline TF-IDF + LinearSVC.
- **Semua percobaan dicatat otomatis** ke `EXPERIMENTS.md` (dan `reports/`).

Laporan yang dihasilkan: `data_validation.md`, `leakage_report.md`,
`baseline_val.md`, `calibration_val.md`, `rag_eval.md`, `e2e_val.md`,
`final_eval.md` (semuanya di `reports/`).

### Penjaga test set (detail)

- Semua modul memuat data lewat `src/data_io.py`.
- `data_io` **menolak** membaca berkas yang dianggap test set (`test.csv`,
  `mixed_test.csv`) kecuali variabel `ALLOW_TEST=1` diset.
- Hanya `src/final_eval.py` yang menyetel `ALLOW_TEST`, dan hanya saat diberi
  frasa konfirmasi persis.

---

## 13. Layanan API

| Endpoint | Metode | Kunci `X-API-KEY`? | Guna |
|----------|--------|--------------------|------|
| `/health` | GET | tidak (publik) | status layanan + jumlah jawaban |
| `/intents` | GET | ya | daftar intent yang dikenali |
| `/chat` | POST | ya | kirim pertanyaan → dapat jawaban + route |

- Kunci dibaca dari `.env` (`CHATBOT_API_KEY`). Jika tidak ada → layanan
  mengembalikan **503**; kunci salah → **401**.
- Ada **rate limit** per kunci/IP (`RATE_LIMIT_PER_MINUTE`, default 30).
- Jawaban pada route `faq` bisa **berbeda tiap permintaan** bila intent punya
  lebih dari satu variasi jawaban (`routing.answer_variants`).
- Layanan ini **untuk uji coba**. Jangan dipasang sebagai layanan permanen /
  service Windows tanpa izin pengelola server (templat NSSM/WinSW sengaja
  dinonaktifkan).

---

## 14. Pengujian

```powershell
& ".\.venv\Scripts\python.exe" -m pytest -q
```

Saat ini **177 tes lulus**. Tes memakai data sintetis sementara (tidak menyentuh
test set proyek).

---

## 15. Dokumentasi

| Berkas | Isi |
|--------|-----|
| [`PANDUAN.md`](PANDUAN.md) | panduan langkah-demi-langkah untuk pemula (termasuk kamus istilah & solusi masalah umum) |
| [`PANDUAN_ISI_DATA.md`](PANDUAN_ISI_DATA.md) | format & cara mengoreksi tiap berkas data |
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | catatan tiap fase + tabel eksperimen otomatis |
| [`deploy/README.md`](deploy/README.md) | cara menjalankan & men-deploy layanan (berkas contoh) |
| `src/`, `rag/`, `api/` | docstring di tiap modul menjelaskan alur & contoh pemakaian |

---

## 16. Catatan lingkungan

- Selalu `encoding='utf-8'` dan `$env:PYTHONUTF8 = 1`.
- `dataloader_num_workers=0` (multiprocessing Windows memakai *spawn*).
- Utas dibatasi lewat `torch.set_num_threads` sesuai `config.yaml` (server lab
  bersama).
- Proyek ini **CPU-only** (tanpa GPU NVIDIA).
- Tidak mengubah firewall / IIS / registry; tidak menjalankan layanan permanen.
- Peringatan `StarletteDeprecationWarning` (saran `httpx2`) hanya kosmetik.

---

## 17. Status fase

| Fase | Status |
|------|--------|
| 0 – Persiapan | ✅ selesai (Python 3.12.10, torch 2.14.1+cpu, CPU-only) |
| 1 – Kerangka & data | ✅ selesai (22 tes lulus) |
| 2 – Augmentasi/leakage/baseline | ✅ kode & uji sandbox (39 tes lulus) |
| 3 – Fine-tuning DistilBERT | ✅ kode & uji sandbox (66 tes lulus; single/search/multiseed) |
| 4 – Kalibrasi & tau1 | ✅ kode & uji sandbox (77 tes lulus) |
| 5 – Pipeline PDF | ✅ kode & uji sandbox (109 tes lulus) |
| 6 – Routing & e2e | ✅ kode & uji sandbox (123 tes lulus) |
| 7 – API & database | ✅ kode & uji sandbox (140 tes lulus) |
| 8 – Deployment (berkas saja) | ✅ kode & uji sandbox (156 tes lulus) |
| 9 – Evaluasi final | 🟡 siap; menunggu perintah `JALANKAN EVALUASI FINAL` (test set belum dibaca) |

> **Total tes saat ini: 177.** Angka pada tabel di atas adalah riwayat akhir
> tiap fase; sejak itu ditambahkan **12 tes** untuk fitur variasi jawaban per
> intent (2026-10-15).

> **Catatan penting:** saat ini **belum ada data asli**, jadi seluruh angka yang
> pernah dihasilkan bertanda **`[SANDBOX]`** (data sintetis) dan **bukan hasil
> penelitian**. Begitu data asli tersedia, jalankan pipeline data asli lalu
> evaluasi final untuk mendapatkan angka yang sah.
