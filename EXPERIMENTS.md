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

---

## Log eksperimen

<!-- Baris baru ditambahkan otomatis di bawah ini -->

| Tanggal | Fase/Skrip | Model | Konfigurasi | Seed | Metrik | Durasi |
|---------|------------|-------|-------------|------|--------|--------|
<!-- APPEND_ROWS_HERE -->
| 2026-10-13 | Fase 5 - retrieval PDF (eval) | rrf | [SANDBOX] method=rrf top_k=3 rrf_k=60 tau2=0.3 status=relaxed | 42 | [SANDBOX] mrr=1.0000 \| recall_at_1=1.0000 \| recall_at_k=1.0000 \| tau=0.3000 \| in_doc_recall=0.0000 \| out_doc_false_accept=0.0000 \| n_in_doc=4 \| n_out_doc=4 | 0.00s |
| 2026-10-13 | Fase 5 - retrieval PDF (eval) | rrf | [SANDBOX] method=rrf top_k=3 rrf_k=60 tau2=0.3 status=relaxed | 42 | [SANDBOX] mrr=1.0000 \| recall_at_1=1.0000 \| recall_at_k=1.0000 \| tau=0.3000 \| in_doc_recall=0.0000 \| out_doc_false_accept=0.0000 \| n_in_doc=4 \| n_out_doc=4 | 0.00s |
| 2026-10-13 | Fase 5 - retrieval PDF (eval) | bm25 | [SANDBOX] method=bm25 top_k=3 rrf_k=60 tau2=None status=skipped | 42 | [SANDBOX] mrr=1.0000 \| recall_at_1=1.0000 \| recall_at_k=1.0000 | 0.00s |
| 2026-10-13 | Fase 4 - kalibrasi (val) | smoke | [SANDBOX] temperature=0.3254 tau1=0.40 status=ok val=val.csv | 42 | [SANDBOX] ece_before=0.3026 \| ece_after=0.1826 \| coverage=0.0400 \| accepted_accuracy=1.0000 \| oos_false_accept=0.0000 | 0.00s |
| 2026-10-13 | Fase 3 - fine-tuning | cahya/distilbert-base-indonesian | [SANDBOX] lr=3e-05 batch=16 epochs=1 max_len=64 class_weight=False train=train_aug.csv | 42 | [SANDBOX] accuracy=0.4400 \| f1_macro=0.3919 \| eval_loss=2.2171 \| best_metric=0.3919 | 44.34s |
| 2026-10-13 | Fase 3 - fine-tuning | cahya/distilbert-base-indonesian | [SANDBOX] lr=3e-05 batch=16 epochs=1 max_len=64 class_weight=False train=train_aug.csv | 42 | [SANDBOX] accuracy=0.4400 \| f1_macro=0.3919 \| eval_loss=2.2171 \| best_metric=0.3919 | 45.43s |
| 2026-10-13 | Fase 2 - baseline | TF-IDF + LinearSVC | [SANDBOX] ngram=(1,2) sublinear_tf=True C=1.0 max_iter=2000 train=train_aug.csv | 42 | [SANDBOX] accuracy=0.7200 \| f1_macro=0.6641 \| f1_weighted=0.6670 \| precision_macro=0.6424 \| recall_macro=0.7273 | 0.02s |
