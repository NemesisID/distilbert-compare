# Panduan Pemilik Proyek — Apa yang Perlu Anda Buat & Lakukan

Berkas ini adalah **daftar tugas** untuk pemilik proyek (Inkubator Bisnis UPN
"Veteran" Jawa Timur). Isinya: berkas yang harus **Anda buat** dan langkah yang
harus **Anda lakukan** agar chatbot bisa dilatih dengan **data asli** dan
dioperasikan.

> Ringkas: **kode Fase 0–9 sudah selesai dan teruji (165 tes lulus).** Yang
> belum ada hanyalah **data asli** dan berkas konfigurasi rahasia (`.env`).
> Selama data asli belum ada, semua angka hanya `[SANDBOX]` (sintetis) dan
> **bukan hasil penelitian**.

---

## 0. Status saat ini (apa yang sudah ada)

| Sudah ada ✅ | Belum ada ❌ (tugas Anda) |
|--------------|---------------------------|
| Seluruh kode: `src/`, `rag/`, `api/`, `scripts/`, `deploy/` | `data/raw/intents_raw.csv` (data latih asli) |
| `config.yaml` (semua parameter) | `data/intents.yaml` (jawaban baku) |
| Templat: `*.template.csv` / `*.template.yaml` | `data/pdf/*.pdf` (dokumen mitra) |
| Data **sintetis sandbox** untuk uji pipeline | `data/rag_eval/{qa_pairs,in_doc,out_doc}.csv` |
| Model pratinjau `models/smoke` (uji alur, 1 epoch) | `data/e2e/{mixed_val,mixed_test}.csv` |
| Model `cahya/distilbert-base-indonesian` terunduh ✅ | Berkas `.env` (kunci API rahasia) |
| Embedder `distiluse-base-multilingual-cased-v2` terunduh ✅ | Model pembanding `distilbert-base-multilingual-cased` (belum diunduh) |

Daftar **11 intent** (tidak boleh diubah tanpa persetujuan):
`syarat_pendaftaran`, `alur_seleksi`, `jadwal_kegiatan`, `ketentuan_luaran`,
`pengeluaran_keuangan`, `review_revisi_berkas`, `kontak_admin`, `sapaan`,
`terima_kasih`, `tanya_dokumen`, `out_of_scope`.

---

## A. Yang perlu Anda BUAT

Semua berkas target berada di dalam `C:\Nitip Ragil\Chatbot\`. Templat sudah
disediakan — **salin** templat, buang akhiran `.template`, lalu isi.

### A1. `.env` (rahasia — jangan di-commit)

```powershell
Copy-Item .env.example .env
```
Lalu isi minimal:

| Kunci | Arti | Contoh |
|-------|------|--------|
| `CHATBOT_API_KEY` | token rahasia yang dikirim backend website via header `X-API-KEY` | string acak panjang |
| `DATABASE_URL` | lokasi database (default SQLite) | `sqlite:///./data/chatbot.db` |
| `CORS_ORIGINS` | origin website utama, dipisah koma | `http://localhost:3000` |
| `RATE_LIMIT_PER_MINUTE` | batas permintaan per menit per IP/kunci | `30` |
| `HF_HUB_OFFLINE` | **opsional** (hanya untuk `huggingface_hub`). `0` = boleh cek internet saat memuat model; `1` = pakai cache lokal saja. Model sudah ada di cache lokal, jadi `0` pun tetap jalan. | `0` |

> Tentang unduhan model: `cahya/distilbert-base-indonesian` dan
> `distiluse-base-multilingual-cased-v2` **sudah ada di cache lokal**, jadi tidak
> perlu internet untuk memuatnya. `HF_HUB_OFFLINE` hanya mengatur apakah Hugging
> Face boleh mengecek internet; **bukan** syarat agar model bisa dipakai.

### A2. `data/raw/intents_raw.csv` — data latih asli (PALING PENTING)

Salin `data/raw/intents_raw.template.csv` → `data/raw/intents_raw.csv`.

- **Kolom wajib:** `text`, `intent` (opsional: `source`, isi `asli`).
- **Isi:** pertanyaan nyata pengguna dalam bahasa Indonesia, satu baris satu
  pertanyaan. Boleh variasi ejaan/gaya agar model tangguh.
- **Jumlah minimal per intent** (di bawah ini hanya **peringatan**, bukan error):

| Intent | Minimal contoh |
|--------|----------------|
| umum (default) | 20 |
| `tanya_dokumen` | 40 |
| `out_of_scope` | 100 |

- **Aturan:** hindari kalimat yang sama persis di dua intent berbeda (konflik
  label), dan hindari duplikat. Panjang teks wajar (3–300 karakter).

Contoh format (baris contoh — **ganti dengan kalimat asli**):

```csv
text,intent,source
apa saja syarat mendaftar tenant inkubator,syarat_pendaftaran,asli
bagaimana tahapan seleksi calon tenant,alur_seleksi,asli
kapan jadwal pelatihan angkatan berikutnya,jadwal_kegiatan,asli
berapa lama masa inkubasi yang diberikan,ketentuan_luaran,asli
```

### A3. `data/intents.yaml` — jawaban baku per intent

Salin `data/intents.template.yaml` → `data/intents.yaml`. Satu entri per intent
(termasuk `sapaan`, `terima_kasih`, `out_of_scope`), berisi `intent`,
`deskripsi`, dan `jawaban` (teks baku yang ditampilkan chatbot — **bukan**
hasil generasi model).

```yaml
- intent: syarat_pendaftaran
  deskripsi: Pertanyaan tentang syarat dan berkas pendaftaran tenant.
  jawaban: "Syarat pendaftaran: ... (isi sesuai kebijakan inkubator)."
```

### A4. `data/pdf/*.pdf` — dokumen mitra

Letakkan berkas PDF (panduan, jadwal, ketentuan, dsb.) di `data/pdf/`.
Catatan: PDF hasil pindai (gambar) tidak dibaca karena OCR tidak dipasang —
gunakan PDF berbasis teks.

### A5. `data/rag_eval/` — berkas uji retrieval

Salin templat berikut (buang `.template`):

| Berkas | Kolom | Isi |
|--------|-------|-----|
| `qa_pairs.csv` | `question,doc,page` | pertanyaan yang jawabannya ada di PDF + nama file + halaman |
| `in_doc.csv` | `question` | pertanyaan yang jawabannya **ADA** di PDF |
| `out_doc.csv` | `question` | pertanyaan yang jawabannya **TIDAK ADA** di PDF (termasuk pengecoh sulit) |

`doc` harus sama dengan nama berkas di `data/pdf/`.

### A6. `data/e2e/` — berkas uji routing

Salin templat:

| Berkas | Kolom | Isi |
|--------|-------|-----|
| `mixed_val.csv` | `question,expected_route` | untuk menyetel ambang (val) |
| `mixed_test.csv` | `question,expected_route` | **test set terkunci** (dibaca hanya saat evaluasi final) |

Nilai `expected_route` salah satu dari: `faq`, `dokumen`, `fallback`.

### A7. (Opsional) Unduh model pembanding

Untuk membandingkan model primer vs pembanding:

```powershell
.\.venv\Scripts\python.exe -c "from transformers import AutoModelForSequenceClassification as M, AutoTokenizer as T; T.from_pretrained('distilbert/distilbert-base-multilingual-cased'); M.from_pretrained('distilbert/distilbert-base-multilingual-cased')"
```
(±520 MB, butuh internet. Tidak mengubah sistem apa pun.)

---

## B. Yang perlu Anda LAKUKAN (urutan langkah)

Semua perintah dijalankan dari `C:\Nitip Ragil\Chatbot` memakai Python venv.
`$env:PYTHONUTF8 = 1` disarankan agar teks Indonesia benar.

```powershell
cd "C:\Nitip Ragil\Chatbot"
$env:PYTHONUTF8 = 1
```

> Alternatif: buka di VS Code dengan `code "C:\Nitip Ragil\Chatbot"` lalu
> gunakan terminal VS Code.

### Langkah 1 — Validasi data
```powershell
.\.venv\Scripts\python.exe -m src.validate_data
```
Cek `reports/data_validation.md`. Wajib **tanpa error** sebelum lanjut
(peringatan jumlah minimal boleh ada, tapi sebaiknya diperbaiki).

### Langkah 2 — Split & kunci test set
```powershell
.\.venv\Scripts\python.exe -m src.split
```
Menghasilkan `data/processed/{train,val,test}.csv` + `TEST_SHA256.txt`.
**Test set dikunci** (read-only) dan sejak ini **tidak boleh dibaca** oleh
proses apa pun kecuali evaluasi final. Augmentasi hanya untuk data train.

### Langkah 3 — Augmentasi, cek kebocoran, baseline
```powershell
.\.venv\Scripts\python.exe -m src.augment        # -> data/augmented/train_aug.csv
.\.venv\Scripts\python.exe -m src.leakage_check  # exit != 0 bila bocor
.\.venv\Scripts\python.exe -m src.baseline       # baseline TF-IDF + LinearSVC (val)
```

### Langkah 4 — Fine-tuning DistilBERT (CPU-only) 
```powershell
# 4a. Cari hyperparameter terbaik di val (grid diperkecil untuk CPU)
.\.venv\Scripts\python.exe -m src.train --mode search --grid reduced

# 4b. Latih ulang dengan 3 seed (42, 43, 44) memakai setelan terbaik
.\.venv\Scripts\python.exe -m src.train --mode multiseed --grid reduced --lr 2e-5 --batch 16
```
Model tersimpan di `models/distilbert-base-indonesian/` (+ `selected.json`).
> Untuk membandingkan dengan model pembanding:
> `... -m src.train --mode search --grid reduced --model distilbert/distilbert-base-multilingual-cased`

### Langkah 5 — Kalibrasi & ambang tau1 (hanya val)
```powershell
.\.venv\Scripts\python.exe -m src.calibrate --model-dir models/distilbert-base-indonesian
```
Menghasilkan `reports/calibration_val.md` + `models/distilbert-base-indonesian/calibration.json`.

### Langkah 6 — Indeks PDF & evaluasi retrieval
```powershell
.\.venv\Scripts\python.exe -m rag.index --pdf-dir data/pdf --out vectorstore
.\.venv\Scripts\python.exe -m rag.eval_retrieval --method rrf
```
Indeks di `vectorstore/`; ambang `tau2` disimpan di `vectorstore/tau2.json`.

### Langkah 7 — Routing end-to-end (val)
```powershell
.\.venv\Scripts\python.exe -m src.router
```
Cek `reports/e2e_val.md` (akurasi route `faq`/`dokumen`/`fallback`).

### Langkah 8 — Siapkan database jawaban, cek kesiapan, jalankan API
```powershell
.\.venv\Scripts\python.exe -m scripts.seed_responses   # isi jawaban baku dari data/intents.yaml
.\.venv\Scripts\python.exe -m scripts.preflight        # harus "SIAP"; keluar != 0 bila belum
.\deploy\run_api.ps1                                   # jalankan layanan (latar depan; hentikan manual)
```
Endpoint: `GET /health` (publik), `GET /intents`, `POST /chat` (header `X-API-KEY`).

### Langkah 9 — Evaluasi FINAL (test set terkunci)
Jalankan **hanya sekali** dan **hanya** atas perintah eksplisit, dengan frasa persis:
```powershell
.\.venv\Scripts\python.exe -m src.final_eval --confirm "JALANKAN EVALUASI FINAL"
```
Menghasilkan `reports/final_eval.md` + catatan di `EXPERIMENTS.md`.
Tanpa frasa persis, perintah **ditolak** (keluar kode 2) dan test set tidak disentuh.

### Opsional — Otomatis
```powershell
.\deploy\build_artifacts.ps1   # menjalankan urutan Langkah 1–8 untuk data asli
python -m pytest -q            # cek seluruh kode (saat ini 165 tes lulus)
```

---

## C. Checklist

- [ ] `.env` dibuat & `CHATBOT_API_KEY` diisi
- [ ] `data/raw/intents_raw.csv` diisi (memenuhi jumlah minimal per intent)
- [ ] `data/intents.yaml` diisi (11 intent)
- [ ] `data/pdf/` diisi PDF berbasis teks
- [ ] `data/rag_eval/{qa_pairs,in_doc,out_doc}.csv` diisi
- [ ] `data/e2e/{mixed_val,mixed_test}.csv` diisi
- [ ] (opsional) model pembanding diunduh
- [ ] Langkah 1–9 dijalankan berurutan
- [ ] Evaluasi final dijalankan **sekali** dengan frasa persis
- [ ] Hasil (non-sandbox) dicatat & dilaporkan

---

## D. Aturan penting & keputusan yang menunggu Anda

**Aturan yang tidak boleh dilanggar**
- **Test set terkunci.** Hanya `src/final_eval.py` yang boleh membacanya, dan
  hanya dengan frasa `JALANKAN EVALUASI FINAL`. Semua data dimuat lewat
  `src/data_io.py` (menolak test set kecuali `ALLOW_TEST=1`).
- **Split sebelum augmentasi**; augmentasi **hanya** pada data train; cek kebocoran.
- **Tanpa LLM / IndoBERT / arsitektur lain.** Lingkup: DistilBERT (klasifikasi),
  `distiluse` + BM25 (retrieval), TF-IDF + LinearSVC (baseline). Jawaban FAQ
  berasal dari database, **bukan** hasil generasi.
- **Jangan mengarang data/hasil.** Data sintetis wajib ditandai `source=synthetic`
  dan tidak boleh dicampur dengan data asli atau dilaporkan sebagai hasil penelitian.
- Semua parameter di `config.yaml` (tanpa angka ajaib di kode); seed 42/43/44.

**Dua keputusan yang masih perlu persetujuan Anda**
1. **Definisi route OOS vs ambang tau1** — apakah "diterima" = keyakinan ≥ tau1,
   atau menolak khusus saat prediksi teratas = `out_of_scope`.
2. **Skala ambang tau2 untuk `distiluse`** — skor kemiripan model ini kecil
   (±0,08–0,15 pada potongan 90 kata), sedangkan `rag.tau2_min` = 0,30.
   Perlu menurunkan `rag.tau2_min` (mis. 0,05) atau menggerbang memakai skor BM25/RRF.

---

## E. Tabel perintah ringkas

| Tujuan | Perintah |
|--------|----------|
| Validasi data | `.\.venv\Scripts\python.exe -m src.validate_data` |
| Split + kunci test | `.\.venv\Scripts\python.exe -m src.split` |
| Augmentasi | `.\.venv\Scripts\python.exe -m src.augment` |
| Cek kebocoran | `.\.venv\Scripts\python.exe -m src.leakage_check` |
| Baseline | `.\.venv\Scripts\python.exe -m src.baseline` |
| Cari hyperparameter | `.\.venv\Scripts\python.exe -m src.train --mode search --grid reduced` |
| Latih 3 seed | `.\.venv\Scripts\python.exe -m src.train --mode multiseed --grid reduced --lr 2e-5 --batch 16` |
| Kalibrasi | `.\.venv\Scripts\python.exe -m src.calibrate --model-dir models/distilbert-base-indonesian` |
| Indeks PDF | `.\.venv\Scripts\python.exe -m rag.index --pdf-dir data/pdf --out vectorstore` |
| Evaluasi retrieval | `.\.venv\Scripts\python.exe -m rag.eval_retrieval --method rrf` |
| Evaluasi routing | `.\.venv\Scripts\python.exe -m src.router` |
| Isi jawaban DB | `.\.venv\Scripts\python.exe -m scripts.seed_responses` |
| Cek kesiapan | `.\.venv\Scripts\python.exe -m scripts.preflight` |
| Evaluasi FINAL | `.\.venv\Scripts\python.exe -m src.final_eval --confirm "JALANKAN EVALUASI FINAL"` |
| Uji kode | `.\.venv\Scripts\python.exe -m pytest -q` |

---

> Catatan: seluruh perintah di atas sudah dipakai pada uji **sandbox**
> (`--sandbox`) dan lulus. Untuk hasil penelitian yang sah, ganti ke **data
> asli** sesuai Bagian A. Setiap fase akan dilaporkan dan di-commit setelah
> diselesaikan.
