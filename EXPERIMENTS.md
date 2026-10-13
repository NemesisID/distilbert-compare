# EXPERIMENTS.md

Log eksperimen. Setiap eksperimen **ditambahkan otomatis** oleh skrip
(lihat `src/experiment_log.py`) dengan kolom: tanggal, skrip/fase, model,
konfigurasi, seed, metrik, durasi.

> **Aturan:** Angka yang berasal dari `data/sandbox/` adalah UJI PIPELINE
> saja dan **bukan** hasil penelitian. Angka tersebut wajib diberi tanda
> `[SANDBOX]`.

---

## Ringkasan fase

| Fase | Status | Catatan |
|------|--------|---------|
| 0 - Persiapan | ✅ selesai | Python 3.12.10 + .venv + semua library; torch 2.14.1+cpu; CPU-only |
| 1 - Kerangka & data | ✅ selesai | 22 tes pytest lulus (PyYAML asli) |
| 2 - Augmentasi/leakage/baseline | ✅ kode & uji sandbox | 39 tes lulus; angka sandbox ditandai [SANDBOX] |
| 3 - Fine-tuning DistilBERT | ✅ kode & uji sandbox | 66 tes lulus; train.py (single/search/multiseed); grid CPU disetujui (--grid reduced) |
| 4 - Kalibrasi & tau1 | ✅ kode & uji sandbox | calibrate.py: temperature scaling (grid) + tau1 pada val; 77 tes lulus |
| 5 - Pipeline PDF/retrieval | ✅ kode & uji sandbox | rag/: pdf_extract, chunking, index (dense+BM25), retrieve (RRF), eval_retrieval (tau2); 109 tes lulus |
| 6 - Routing & e2e | ✅ kode & uji sandbox | src/router.py: intent+tau1+retrieval/tau2 -> faq/dokumen/fallback; 123 tes lulus |
| 7 - API & database | ✅ kode & uji sandbox | api/: FastAPI (/health, /intents, /chat) + SQLite (SQLAlchemy) + kunci X-API-KEY + rate limit + CORS; scripts/seed_responses.py; 140 tes lulus |
| 8 - Deployment | ✅ kode & uji sandbox | deploy/ (contoh PS1 + templat layanan Windows, berkas saja) + scripts/preflight.py; 156 tes lulus |
| 9 - Evaluasi final | 🟡 siap, belum dijalankan | src/final_eval.py + gerbang frasa; test set BELUM dibaca; 165 tes lulus |

### 2026-10-13 - Fase 0 (lingkungan)

- Python 3.12.10 dipasang user-level (tanpa Administrator); .venv dibuat ulang.
- `pip freeze` -> `reports/library_versions.txt` (83 paket).
- `torch.cuda.is_available()` = **False** -> CPU-only; `nvidia-smi` tidak ada.
- Tesseract tidak terpasang (relevan untuk Fase 5 bila PDF hasil scan).

---

## Catatan eksekusi

### 2026-10-13 - Fase 1 (uji pipeline, SANDBOX)

- Perintah: `python -m src.split --sandbox --out-dir data/sandbox/processed`
- Hasil **[SANDBOX]**: train=115, val=25, test=25 (total 165, 11 intent x 15).
- Deduplikasi: 0 baris dibuang.
- Test set sandbox dikunci (read-only), SHA-256:
  `c4f5d2824e1debbb9b76b59aeccf88937f5f8331ecc5f5f38b59842318aead95`
- `python -m pytest -q` -> 22 passed (bertambah menjadi 39 setelah Fase 2).
- Catatan: angka ini dari data sintetis, BUKAN hasil penelitian.

### 2026-10-13 - Fase 2 (uji pipeline, SANDBOX)

- Perintah: `python -m src.augment --sandbox`, `python -m src.leakage_check --sandbox`,
  `python -m src.baseline --sandbox`.
- Augmentasi **[SANDBOX]**: 115 kalimat asli -> 254 variasi (kamus 37 grup), total 369.
- Leakage **[SANDBOX]**: 0 teks identik vs val, 0 duplikat internal -> BERSIH.
- Baseline **[SANDBOX]**: accuracy=0.7200, f1_macro=0.6641 (val 25 baris).
- Catatan: semua angka di atas dari data sintetis, BUKAN hasil penelitian.

### 2026-10-13 - Fase 3 (uji pipeline, SANDBOX)

- Perintah: `python -m src.train --mode single --sandbox --model cahya/distilbert-base-indonesian --epochs 1 --batch 16 --lr 3e-5`
- Fine-tuning **[SANDBOX]**: 1 epoch, val 25 baris -> accuracy=0.4400, f1_macro=0.3919
  (belum konvergen; hanya untuk memvalidasi pipeline, BUKAN hasil penelitian).
- Catatan teknis transformers 5.19: `eval_strategy`, `processing_class`, `warmup_steps`
  (bukan `warmup_ratio`), `save_only_model=True`.
- `torch.cuda.is_available()` = **False** (device_count=0, tanpa `nvidia-smi`) -> CPU-only.
- `python -m pytest -q` -> 66 passed.
- **Grid diperkecil DISETUJUI user** (CPU-only): model primer, lr {2e-5, 3e-5} x
  batch {16}, maks 6 epoch, early-stopping patience 2 (`training.reduced_grid`),
  dijalankan lewat `--grid reduced` saat data asli tersedia.
- Model pembanding (distilbert-base-multilingual-cased) menyusul bila hasil primer wajar.

### 2026-10-13 - Fase 4 (uji pipeline, SANDBOX)

- Perintah: `python -m src.calibrate --sandbox --model-dir models/smoke`
- Kalibrasi **[SANDBOX]** (val 25 baris, model smoke 1 epoch): T=0.3254,
  ECE 0.3026 -> 0.1826; tau1=0.40 (status `ok`) dengan coverage 0.04
  (model belum konvergen sehingga sangat sedikit yang yakin). BUKAN hasil penelitian.
- Suhu dicari lewat grid log-spaced (bukan LBFGS) demi kestabilan pada data kecil.
- `python -m pytest -q` -> 77 passed.

### 2026-10-13 - Fase 5 (uji pipeline, SANDBOX)

- Perintah:
  `python -m scripts.make_sandbox_rag` (buat PDF+eval sintetis) ->
  `python -m rag.index --pdf-dir data/sandbox/pdf --out data/sandbox/vectorstore --no-dense` ->
  `python -m rag.index --pdf-dir data/sandbox/pdf --out data/sandbox/vectorstore_dense` ->
  `python -m rag.eval_retrieval --sandbox --method rrf --vectorstore data/sandbox/vectorstore_dense`.
- Retrieval **[SANDBOX]** (2 dokumen, 4 chunk): MRR=1.0000, recall@1=recall@2=recall@3=1.0000.
  Data sintetis, BUKAN hasil penelitian.
- **Temuan skala tau2:** model `distiluse-base-multilingual-cased-v2` sangat menurunkan
  kemiripan saat jawaban tertanam dalam chunk panjang (mis. 0.66 -> 0.19 pada chunk ~47 kata).
  Akibatnya skor top-1 in-doc hanya ~0.15-0.22, di bawah `rag.tau2_min` (0.30), sehingga
  grid tau2 default [0.30, 0.90] tidak menemukan ambang valid (status `relaxed`).
  Laporan menandai ini lewat "Peringatan skala". Kandidat perbaikan (menunggu keputusan
  pemilik proyek): turunkan `rag.tau2_min` (mis. 0.05) atau pakai skor gabungan BM25/RRF.
- Tesseract belum terpasang -> PDF hasil scan hanya memicu peringatan (tanpa OCR).
- `python -m pytest -q` -> 109 passed.

### 2026-10-13 - Fase 6 (uji pipeline, SANDBOX)

- Perintah: `python -m src.router --sandbox` (evaluasi `mixed_val`) dan
  `python -m src.router --sandbox --ask "<pertanyaan>" [--tau1 .. --tau2 ..]`.
- Routing e2e **[SANDBOX]** (12 baris, model smoke 1 epoch): akurasi route = 0.3333;
  hampir semua jatuh ke `fallback` karena model belum konvergen (keyakinan ~0.17-0.39
  < tau1=0.40). **BUKAN hasil penelitian**; hanya validasi alur
  model -> kalibrasi -> retrieval -> jawaban.
- Ketiga route terbukti jalan lewat `--ask`: `faq` (terima_kasih -> jawaban baku),
  `dokumen` (kutipan `jadwal_program.pdf` hal. 1 + daftar sumber), dan `fallback`
  (OOS / keyakinan rendah / dokumen tak ditemukan).
- Definisi route (default; menunggu konfirmasi pemilik proyek): OOS => `fallback`;
  keyakinan < tau1 => `fallback`; `tanya_dokumen` & skor < tau2 => `fallback`;
  selain itu => `faq`.
- `python -m pytest -q` -> 123 passed.

### 2026-10-13 - Fase 7 (uji pipeline, SANDBOX)

- Perintah: `python -m scripts.seed_responses --sandbox`, lalu smoke `TestClient`
  (FastAPI) dengan Router sandbox (`models/smoke` + `data/sandbox/vectorstore_dense`).
- API: `GET /health` (publik), `GET /intents` + `POST /chat` (wajib header
  `X-API-KEY`). Jawaban baku dibaca dari tabel `responses` (SQLite via SQLAlchemy);
  setiap panggilan `/chat` dicatat ke `query_log` (audit).
- Seed **[SANDBOX]**: `data/sandbox/intents_sandbox.yaml` -> 11 intent ke
  `data/sandbox/chatbot_sandbox.db`.
- Smoke **[SANDBOX]**: `/health` -> `n_responses=11`, `router_ready=True`;
  `/intents` -> 11 entri; `/chat` tanpa kunci -> **401**; semua pertanyaan ->
  `fallback` karena model smoke (1 epoch) belum konvergen (keyakinan ~0.22-0.29
  < tau1=0.40). **BUKAN hasil penelitian**, hanya validasi alur
  API -> Router -> DB.
- Keamanan/operasional: kunci API HANYA dari `.env` (`CHATBOT_API_KEY`); bila
  belum diisi, `/chat` & `/intents` -> **503**. CORS dari `CORS_ORIGINS`;
  pembatas laju jendela-geser per kunci/IP (`RATE_LIMIT_PER_MINUTE`, default 30/menit).
  URL database dari `DATABASE_URL` (fallback `paths.database`). Tidak ada layanan
  permanen yang dijalankan.
- Catatan: `TestClient` (starlette 1.8 + httpx 0.28) memunculkan
  `StarletteDeprecationWarning` (saran `httpx2`) — hanya peringatan, semua tes lulus.
- `python -m pytest -q` -> 140 passed.

### 2026-10-13 - Fase 8 (berkas deployment, belum dijalankan)

- Ditambahkan `scripts/preflight.py`: pemeriksaan kesiapan tanpa menjalankan
  layanan (`.env` + kunci API, berkas jawaban, direktori model, indeks
  retrieval, database `responses`). `python -m scripts.preflight --sandbox`
  **[SANDBOX]** -> **SIAP** (11 intent, model smoke, vectorstore dense+bm25,
  11 jawaban); keluar kode != 0 bila belum siap.
- Berkas contoh di `deploy/` (**tidak** dijalankan otomatis): `run_api.ps1`
  (uvicorn latar depan + preflight), `healthcheck.ps1`, `build_artifacts.ps1`,
  `backup_db.ps1`, `env.production.example`, dan templat layanan Windows
  (`windows-service/`: NSSM + WinSW) yang **dinonaktifkan** (butuh izin).
- Tidak mengubah firewall/IIS/registry; tidak menjalankan layanan permanen.
- `python -m pytest -q` -> 156 passed.

### 2026-10-13 - Fase 9 (siap, BELUM dijalankan)

- Dibuat `src/final_eval.py`: SATU-SATUNYA modul yang menyetel `ALLOW_TEST=1`
  untuk membaca test set. Dilindungi **gerbang frasa konfirmasi**
  (`--confirm "JALANKAN EVALUASI FINAL"`); tanpa frasa persis -> keluar kode 2
  dan tidak menyentuh apa pun.
- Mengimpor modul ini **tidak** membuka test set (kunci dibuka hanya di dalam
  `run_final_eval`, lalu dikembalikan ke keadaan semula) — diuji eksplisit.
- Yang dihitung bila dijalankan: metrik klasifikasi intent (accuracy, P/R/F1
  macro & weighted, F1 per kelas, confusion), ECE + coverage/accepted-accuracy
  pada tau1, dan akurasi route e2e dari `mixed_test.csv`. T & tau1 dibaca dari
  `models/<slug>/calibration.json` (hasil val, tidak dipilih ulang).
- **Test set proyek BELUM dibaca.** Alur diuji lewat fake (logits & router)
  memakai CSV sintetis sementara di `tmp_path`, tanpa model dan tanpa test set.
- `python -m pytest -q` -> 165 passed (9 tes final_eval baru).

---

## Log eksperimen

<!-- Baris baru ditambahkan otomatis di bawah ini -->

| Tanggal | Fase/Skrip | Model | Konfigurasi | Seed | Metrik | Durasi |
|---------|------------|-------|-------------|------|--------|--------|
<!-- APPEND_ROWS_HERE -->
| 2026-10-13 | Fase 8 - preflight deployment | smoke | [SANDBOX] require_env=False checks=answers,model,vectorstore,db | 42 | [SANDBOX] status=SIAP \| answers=11 \| vs=dense+bm25 \| db=11 | 0.00s |
| 2026-10-13 | Fase 7 - API e2e (smoke) | smoke | [SANDBOX] FastAPI+SQLite auth=X-API-KEY seed=11 taut1=0.40 | 42 | [SANDBOX] health=ok \| intents=11 \| chat_no_key=401 \| routes=fallback(3) | 0.00s |
| 2026-10-13 | Fase 6 - routing e2e (val) | smoke | [SANDBOX] method=rrf tau1=0.40 tau2=0.30 n=12 | 42 | [SANDBOX] accuracy=0.3333 | 0.00s |
| 2026-10-13 | Fase 5 - retrieval PDF (eval) | rrf | [SANDBOX] method=rrf top_k=3 rrf_k=60 tau2=0.3 status=relaxed | 42 | [SANDBOX] mrr=1.0000 \| recall_at_1=1.0000 \| recall_at_k=1.0000 \| tau=0.3000 \| in_doc_recall=0.0000 \| out_doc_false_accept=0.0000 \| n_in_doc=4 \| n_out_doc=4 | 0.00s |
| 2026-10-13 | Fase 5 - retrieval PDF (eval) | rrf | [SANDBOX] method=rrf top_k=3 rrf_k=60 tau2=0.3 status=relaxed | 42 | [SANDBOX] mrr=1.0000 \| recall_at_1=1.0000 \| recall_at_k=1.0000 \| tau=0.3000 \| in_doc_recall=0.0000 \| out_doc_false_accept=0.0000 \| n_in_doc=4 \| n_out_doc=4 | 0.00s |
| 2026-10-13 | Fase 5 - retrieval PDF (eval) | bm25 | [SANDBOX] method=bm25 top_k=3 rrf_k=60 tau2=None status=skipped | 42 | [SANDBOX] mrr=1.0000 \| recall_at_1=1.0000 \| recall_at_k=1.0000 | 0.00s |
| 2026-10-13 | Fase 4 - kalibrasi (val) | smoke | [SANDBOX] temperature=0.3254 tau1=0.40 status=ok val=val.csv | 42 | [SANDBOX] ece_before=0.3026 \| ece_after=0.1826 \| coverage=0.0400 \| accepted_accuracy=1.0000 \| oos_false_accept=0.0000 | 0.00s |
| 2026-10-13 | Fase 3 - fine-tuning | cahya/distilbert-base-indonesian | [SANDBOX] lr=3e-05 batch=16 epochs=1 max_len=64 class_weight=False train=train_aug.csv | 42 | [SANDBOX] accuracy=0.4400 \| f1_macro=0.3919 \| eval_loss=2.2171 \| best_metric=0.3919 | 44.34s |
| 2026-10-13 | Fase 3 - fine-tuning | cahya/distilbert-base-indonesian | [SANDBOX] lr=3e-05 batch=16 epochs=1 max_len=64 class_weight=False train=train_aug.csv | 42 | [SANDBOX] accuracy=0.4400 \| f1_macro=0.3919 \| eval_loss=2.2171 \| best_metric=0.3919 | 45.43s |
| 2026-10-13 | Fase 2 - baseline | TF-IDF + LinearSVC | [SANDBOX] ngram=(1,2) sublinear_tf=True C=1.0 max_iter=2000 train=train_aug.csv | 42 | [SANDBOX] accuracy=0.7200 \| f1_macro=0.6641 \| f1_weighted=0.6670 \| precision_macro=0.6424 \| recall_macro=0.7273 | 0.02s |
