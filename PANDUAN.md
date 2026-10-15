# Panduan Praktis — untuk Pemula

**Selamat!** Kode chatbot ini sudah **selesai dan teruji**. Yang tersisa hanya
dua hal: **mengisi data asli** dan **menjalankan beberapa perintah**. Panduan
ini ditulis untuk orang yang belum terbiasa dengan Python atau terminal —
cukup ikuti urutannya, tidak perlu paham pemrograman.

---

## Cara memakai panduan ini

1. Kerjakan **Bagian A** lebih dulu (membuat berkas data).
2. Lanjut **Bagian B** **berurutan** — jangan melompat.
3. Di setiap perintah ada kotak **"Tanda berhasil"**. Pastikan tanda itu muncul
   sebelum lanjut ke langkah berikutnya.
4. Kalau ada yang aneh/error, buka **Bagian E (Masalah umum)**.
5. **Aturan nomor satu: jangan pernah mengarang data atau hasil.** Semua isian
   harus data sungguhan dari inkubator.

> Semua perintah ditulis satu baris dan bisa langsung **disalin-tempel** ke
> terminal. Jangan diketik ulang secara manual agar tidak salah.

---

## Kamus istilah (biar tidak bingung)

| Istilah | Arti sederhana |
|---------|----------------|
| **Terminal / PowerShell** | Jendela hitam/biru tempat mengetik perintah. |
| **Folder proyek** | `C:\Nitip Ragil\Chatbot` — tempat semua berkas proyek ini. |
| **venv** | "Kotak" berisi Python + semua alat proyek. Selalu pakai Python dari dalam `.venv`. |
| **`.csv`** | Berkas tabel sederhana (seperti Excel). Pemisah kolom = **koma**. |
| **`.yaml`** | Berkas teks terstruktur. **Spasi/indentasi penting** (lihat A3). |
| **Intent** | "Maksud" pertanyaan, mis. `syarat_pendaftaran` = pertanyaan soal syarat daftar. |
| **Dataset** | Kumpulan pertanyaan + intent-nya, dipakai untuk melatih model. |
| **Latih / fine-tuning** | Proses model "belajar" dari data Anda. |
| **Model** | Hasil latihan; yang menebak intent dari pertanyaan baru. |
| **Split** | Membagi data jadi 3: **train** (belajar), **val** (uji sambil menyetel), **test** (ujian akhir). |
| **test set (terkunci)** | Bagian data untuk ujian akhir. **Tidak boleh dilihat** sampai evaluasi final. |
| **Augmentasi** | Menambah variasi kalimat latih otomatis (hanya dari data train). |
| **Kalibrasi / tau1** | Menyetel seberapa "yakin" model harus sebelum menjawab, agar tidak asal jawab. |
| **Retrieval / PDF** | Mencari jawaban di dalam dokumen PDF mitra. |
| **Embedding / BM25 / RRF** | Cara mencocokkan pertanyaan dengan isi dokumen (dua metode + gabungannya). |
| **tau2** | Ambang "kecocokan dokumen" — kalau kurang cocok, tidak dipaksa jawab. |
| **Route** | Hasil akhir: `faq` (jawaban baku), `dokumen` (kutipan PDF), atau `fallback` (menyerah sopan). |
| **API** | "Pintu" agar website utama bisa menanyakan ke chatbot ini. |
| **`.env`** | Berkas rahasia berisi kunci/kata sandi. **Jangan dibagikan/di-commit.** |
| **Commit** | Menyimpan catatan perubahan (dilakukan oleh pengembang, bukan tugas Anda). |

---

## 0. Status sekarang

| Sudah siap ✅ | Belum ada ❌ (tugas Anda di Bagian A) |
|----------------|--------------------------------------|
| Semua kode program | `data/raw/intents_raw.csv` — pertanyaan latih asli |
| `config.yaml` (semua setelan) | `data/intents.yaml` — jawaban baku chatbot |
| Templat `*.template.csv` / `*.template.yaml` | `data/pdf/*.pdf` — dokumen mitra |
| Data **sintetis** untuk uji coba | `data/rag_eval/` — berkas uji pencarian dokumen |
| Model pratinjau `models/smoke` | `data/e2e/` — berkas uji routing |
| 3 model AI sudah terunduh ✅ | Berkas `.env` — kunci API |

Daftar **11 intent** (nama-nama ini **tidak boleh diubah** tanpa persetujuan):
`syarat_pendaftaran`, `alur_seleksi`, `jadwal_kegiatan`, `ketentuan_luaran`,
`pengeluaran_keuangan`, `review_revisi_berkas`, `kontak_admin`, `sapaan`,
`terima_kasih`, `tanya_dokumen`, `out_of_scope`.

---

## A. Membuat berkas (tugas Anda)

> 📄 **Rincian lengkap tiap berkas** (arti tiap kolom, contoh siap salin-tempel,
> dan cara mengoreksi bila ada kesalahan) ada di dokumen terpisah:
> **[`PANDUAN_ISI_DATA.md`](PANDUAN_ISI_DATA.md)**. Bagian di bawah ini adalah
> ringkasannya.

### Dulu: 3 tips penting agar tidak gagal

1. **Tampilkan ekstensi berkas di Windows.** Buka File Explorer → tab **View** →
   centang **File name extensions**. Kalau tidak, berkas bisa tersimpan sebagai
   `intents_raw.csv.txt` dan tidak terbaca. (Nama harus berakhiran `.csv` / `.yaml`.)
2. **Simpan sebagai UTF-8.** Pakai **VS Code** untuk mengetik. Kalau pakai
   **Excel**, pilih *Save As* → **CSV UTF-8 (Comma delimited) (\*.csv)** — jangan
   yang "CSV (Comma delimited)" biasa.
3. **Jangan ubah nama kolom.** Baris pertama (header) harus persis seperti yang
   tertulis di panduan ini.

> Cara termudah menyunting berkas: buka VS Code (`code "C:\Nitip Ragil\Chatbot"`),
> klik berkasnya di panel kiri, ketik, lalu **Ctrl+S** untuk simpan.

### A1. `.env` — berkas rahasia (jangan dibagikan)

1. Di VS Code, buka berkas `.env.example`.
2. **File → Save As…**, ketik nama `.env` (tanpa `.example`), simpan di folder proyek.
3. Buat kunci rahasia acak dengan perintah ini (salin-tempel ke terminal):

   ```powershell
   -join ((1..40) | ForEach-Object { '{0:x}' -f (Get-Random -Maximum 16) })
   ```
   Salin hasilnya (40 huruf/angka) → tempel sebagai nilai `CHATBOT_API_KEY`.
4. Isinya minimal seperti ini:

   ```dotenv
   CHATBOT_API_KEY=hasil-acak-tadi-tempel-di-sini
   DATABASE_URL=sqlite:///./data/chatbot.db
   CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
   RATE_LIMIT_PER_MINUTE=30
   HF_HUB_OFFLINE=0
   ```

| Kunci | Artinya |
|-------|---------|
| `CHATBOT_API_KEY` | Kata sandi yang dikirim website utama (header `X-API-KEY`). |
| `DATABASE_URL` | Lokasi database jawaban baku (default: SQLite). Biarkan apa adanya. |
| `CORS_ORIGINS` | Alamat website yang boleh memakai API ini. |
| `RATE_LIMIT_PER_MINUTE` | Batas jumlah permintaan per menit (anti-spam). |
| `HF_HUB_OFFLINE` | **Opsional.** `0` = boleh cek internet; `1` = pakai lokal saja. Tiga model sudah ada di lokal, jadi aman `0`. |

> ✅ Tanda berhasil: sampai langkah ini belum ada yang dijalankan — cukup
> pastikan berkas bernama tepat `.env` sudah ada di folder proyek.

### A2. `data/raw/intents_raw.csv` — pertanyaan latih asli (PALING PENTING)

**Ini isi yang paling menentukan kualitas chatbot.** Isinya: daftar pertanyaan
nyata pengguna, masing-masing diberi label intent.

Cara membuat:
1. Buat berkas baru `data/raw/intents_raw.csv` (VS Code, atau Excel → Save As CSV UTF-8).
2. Baris pertama (header) **wajib** persis:

   ```
   text,intent,source
   ```
3. Baris berikutnya: satu pertanyaan per baris. Contoh:

   ```csv
   text,intent,source
   apa saja syarat mendaftar tenant inkubator,syarat_pendaftaran,asli
   dokumen apa yang harus saya siapkan untuk pendaftaran,syarat_pendaftaran,asli
   bagaimana alur seleksi calon tenant,alur_seleksi,asli
   kapan jadwal pelatihan angkatan berikutnya,jadwal_kegiatan,asli
   berapa lama masa inkubasi yang diberikan,ketentuan_luaran,asli
   selamat pagi,sapaan,asli
   terima kasih,terima_kasih,asli
   ```

**Aturan mengisi (penting):**
- **Kalau kalimat mengandung koma**, bungkus seluruh kalimat dengan tanda kutip
  ganda. Contoh: `"apa syaratnya, tolong jelaskan",syarat_pendaftaran,asli`
  (di Excel ini otomatis).
- Kolom `source` sebaiknya diisi `asli` (menandai data sungguhan).
- **Hindari** kalimat yang sama persis diberi intent berbeda, dan hindari duplikat.
- Usahakan kalimat bervariasi (gaya formal, santai, singkat, panjang) supaya model tangguh.

**Jumlah minimal per intent** (kurang dari ini hanya jadi *peringatan*, bukan error):

| Intent | Minimal contoh |
|--------|----------------|
| Sebagian besar intent | 20 |
| `tanya_dokumen` | 40 |
| `out_of_scope` | 100 |

> ✅ Tanda berhasil: berkas `data/raw/intents_raw.csv` ada, header-nya
> `text,intent,source`, dan setiap intent punya cukup baris.

### A3. `data/intents.yaml` — jawaban baku chatbot

Ini jawaban yang akan ditampilkan chatbot untuk tiap intent (teks baku, **bukan**
buatan AI). Cara termudah: salin `data/intents.template.yaml` → `data/intents.yaml`,
lalu **ganti teks `jawaban`** dengan jawaban asli kebijakan inkubator.

Kalau mau membuat dari nol, tempel kerangka berikut di VS Code lalu isi tiap
`jawaban`. **Perhatikan spasi** — gunakan **2 spasi** untuk indentasi, **jangan
pakai tombol Tab**, dan pertahankan tanda `-` di awal tiap entri:

```yaml
- intent: syarat_pendaftaran
  deskripsi: Pertanyaan tentang syarat dan berkas pendaftaran tenant.
  jawaban: "Tulis jawaban resmi di sini."

- intent: alur_seleksi
  deskripsi: Pertanyaan tentang tahapan seleksi calon tenant.
  jawaban: "Tulis jawaban resmi di sini."

- intent: jadwal_kegiatan
  deskripsi: Pertanyaan tentang jadwal kegiatan dan pelatihan.
  jawaban: "Tulis jawaban resmi di sini."

- intent: ketentuan_luaran
  deskripsi: Pertanyaan tentang luaran yang harus dihasilkan peserta.
  jawaban: "Tulis jawaban resmi di sini."

- intent: pengeluaran_keuangan
  deskripsi: Pertanyaan tentang aturan pengeluaran dan reimbursement.
  jawaban: "Tulis jawaban resmi di sini."

- intent: review_revisi_berkas
  deskripsi: Pertanyaan tentang revisi dan peninjauan berkas.
  jawaban: "Tulis jawaban resmi di sini."

- intent: kontak_admin
  deskripsi: Pertanyaan tentang cara menghubungi admin/pengelola.
  jawaban: "Tulis jawaban resmi di sini."

- intent: sapaan
  deskripsi: Sapaan pembuka.
  jawaban: "Halo! Ada yang bisa saya bantu terkait program inkubator?"

- intent: terima_kasih
  deskripsi: Ucapan terima kasih.
  jawaban: "Sama-sama, semoga membantu!"

- intent: tanya_dokumen
  deskripsi: Permintaan dokumen/PDF mitra.
  jawaban: "Silakan sebutkan dokumen yang Anda cari, saya akan bantu temukan."

- intent: out_of_scope
  deskripsi: Pertanyaan di luar cakupan layanan FAQ inkubator.
  jawaban: "Maaf, pertanyaan itu di luar cakupan layanan ini. Silakan hubungi admin."
```

> Nama setelah `intent:` **harus sama** dengan 11 daftar di Bagian 0.

> ✅ Tanda berhasil: berkas `data/intents.yaml` ada dan berisi **11 entri**
> (masing-masing diawali `- intent:`).

### A4. `data/pdf/` — dokumen mitra

Salin berkas PDF kebijakan/panduan/jadwal ke folder `data/pdf/`.
Syarat: PDF harus **berisi teks** (bisa di-cari dengan Ctrl+F). PDF hasil
**pindai** (gambar) **tidak terbaca** karena fitur OCR tidak dipasang.

> ✅ Tanda berhasil: ada satu atau lebih berkas `.pdf` di `data/pdf/`.

### A5. `data/rag_eval/` — berkas uji pencarian dokumen

Salin 3 templat ini (buang akhiran `.template`), lalu isi:

| Berkas tujuan | Header (baris pertama) | Isinya |
|---------------|------------------------|--------|
| `data/rag_eval/qa_pairs.csv` | `question,doc,page` | Pertanyaan yang jawabannya **ada** di PDF + **nama berkas PDF** + nomor halaman |
| `data/rag_eval/in_doc.csv` | `question` | Pertanyaan yang jawabannya **ada** di PDF |
| `data/rag_eval/out_doc.csv` | `question` | Pertanyaan yang jawabannya **tidak ada** di PDF (termasuk pengecoh) |

Contoh `qa_pairs.csv`:

```csv
question,doc,page
berapa biaya pendaftaran program,panduan_pendaftaran.pdf,2
apa saja jadwal pelatihan tahun ini,jadwal_program.pdf,1
```

> `doc` harus **sama persis** dengan nama berkas di `data/pdf/` (termasuk `.pdf`).

### A6. `data/e2e/` — berkas uji routing (faq/dokumen/fallback)

Salin 2 templat ini (buang `.template`), lalu isi:

| Berkas tujuan | Header | Isinya |
|---------------|--------|--------|
| `data/e2e/mixed_val.csv` | `question,expected_route` | Untuk menyetel ambang (dipakai program) |
| `data/e2e/mixed_test.csv` | `question,expected_route` | **TEST SET — jangan dibuka/diubah sampai evaluasi final** |

Nilai `expected_route` **hanya** salah satu dari: `faq`, `dokumen`, `fallback`.

Contoh isi:

```csv
question,expected_route
apa saja syarat mendaftar tenant,faq
berapa biaya pada dokumen panduan,dokumen
siapa presiden Indonesia,fallback
```

### A7. ✅ Model AI — sudah diunduh (tidak perlu Anda lakukan)

Ketiga model sudah ada di komputer ini: klasifier primer, klasifier pembanding
(`distilbert-base-multilingual-cased`), dan embedder. Jadi **tidak ada** yang
perlu diunduh lagi.

---

## B. Menjalankan perintah (langkah demi langkah)

### Langkah 0 — Buka terminal yang benar (sekali saja)

1. Buka VS Code: jalankan perintah ini di PowerShell, atau buka VS Code lalu
   **File → Open Folder** → pilih `C:\Nitip Ragil\Chatbot`:

   ```powershell
   code "C:\Nitip Ragil\Chatbot"
   ```
2. Di VS Code buka terminal: menu **Terminal → New Terminal** (pintasan:
   **Ctrl + \`** — tombol di bawah Esc).
3. Pastikan terminal berada di folder proyek, lalu aktifkan "mode teks
   Indonesia" dan uji Python:

   ```powershell
   cd "C:\Nitip Ragil\Chatbot"
   $env:PYTHONUTF8 = 1
   & ".\.venv\Scripts\python.exe" --version
   ```

> **Penting:** selalu mulai perintah dengan `& ".\.venv\Scripts\python.exe" -m ...`.
> **Jangan** pakai `python` saja (bisa memanggil Python versi salah).

> ✅ Tanda berhasil: muncul tulisan seperti `Python 3.12.10`.

---

Berikut langkahnya. Di setiap perintah ada **Tanda berhasil** dan **Kalau gagal**.

### Langkah 1 — Periksa kelengkapan data

**Tujuan:** memastikan berkas data sudah benar sebelum dipakai.

```powershell
& ".\.venv\Scripts\python.exe" -m src.validate_data
```

- **Hasilnya:** laporan dibuat di `reports/data_validation.md`.
- **Tanda berhasil:** perintah selesai tanpa kata `ERROR` di laporan. Untuk
  melihat laporan: klik berkas itu di VS Code, lalu **Ctrl+Shift+V** (pratinjau).
- **Kalau gagal:** perbaiki berkas data sesuai pesan (biasanya kolom salah,
  ada baris kosong, atau intent tidak dikenal). Jalankan ulang perintah ini.

### Langkah 2 — Bagi data & kunci test set

**Tujuan:** membagi data jadi train/val/test (70%/15%/15%) dan **mengunci**
bagian test.

```powershell
& ".\.venv\Scripts\python.exe" -m src.split
```

- **Hasilnya:** `data/processed/train.csv`, `val.csv`, `test.csv`, + `TEST_SHA256.txt`.
- **Tanda berhasil:** ketiga berkas itu muncul.
- ⚠️ **Setelah langkah ini, JANGAN membuka `test.csv`.** Itu "soal ujian" yang
  hanya boleh dibuka saat evaluasi final.

### Langkah 3 — Tambah variasi, cek kebocoran, dan uji pembanding sederhana

**Tujuan:** (a) menambah variasi kalimat latih, (b) memastikan tidak ada
kebocoran data, (c) mengukur pembanding klasik (TF-IDF + LinearSVC).

```powershell
& ".\.venv\Scripts\python.exe" -m src.augment
& ".\.venv\Scripts\python.exe" -m src.leakage_check
& ".\.venv\Scripts\python.exe" -m src.baseline
```

- **Hasilnya:** `data/augmented/train_aug.csv`, `reports/leakage_report.md`,
  `reports/baseline_val.md`.
- **Tanda berhasil:** `leakage_check` selesai **tanpa** menandai kebocoran
  (kalau ada, akan diminta diperiksa dulu).
- **Kalau gagal:** kalau kebocoran terdeteksi, ada kalimat yang terlalu mirip
  antara train dan val — perbaiki data lalu ulangi dari Langkah 2.

### Langkah 4 — Latih model AI (bagian paling lama)

**Tujuan:** melatih model untuk mengenali intent. Ada 2 tahap.

```powershell
& ".\.venv\Scripts\python.exe" -m src.train --mode search --grid reduced
& ".\.venv\Scripts\python.exe" -m src.train --mode multiseed --grid reduced --lr 2e-5 --batch 16
```

- **Hasilnya:** folder model di `models/distilbert-base-indonesian/`.
- **Tanda berhasil:** di akhir muncul angka `accuracy` dan `f1_macro`, dan ada
  baris `model disimpan di: ...`.
- **Catatan:** tahap ini **paling lama** (bisa puluhan menit) karena komputer ini
  tidak punya GPU. **Biarkan jendela terbuka** sampai selesai; jangan diclose.
- **Kalau gagal:** bila keluar peringatan berat, catat pesan terakhirnya dan
  beri tahu pengembang.

> Ingin membandingkan dengan model lain? Tambahkan di akhir perintah:
> `--model distilbert/distilbert-base-multilingual-cased`

### Langkah 5 — Setel ambang keyakinan (kalibrasi)

**Tujuan:** menentukan seberapa yakin model harus menjawab, agar tidak asal.

```powershell
& ".\.venv\Scripts\python.exe" -m src.calibrate --model-dir models/distilbert-base-indonesian
```

- **Hasilnya:** `reports/calibration_val.md` + `models/distilbert-base-indonesian/calibration.json`.
- **Tanda berhasil:** angka `tau1` muncul di ringkasan/berkas laporan.

### Langkah 6 — Bangun "indeks" dokumen PDF

**Tujuan:** menyiapkan mesin pencari untuk isi PDF + mengujinya.

```powershell
& ".\.venv\Scripts\python.exe" -m rag.index --pdf-dir data/pdf --out vectorstore
& ".\.venv\Scripts\python.exe" -m rag.eval_retrieval --method rrf
```

- **Hasilnya:** folder `vectorstore/` + `vectorstore/tau2.json` + laporan evaluasi.
- **Tanda berhasil:** angka `MRR`/kecocokan muncul.
- **Kalau gagal:** pastikan `data/pdf/` berisi PDF **berbasis teks** (bukan hasil
  pindai).

### Langkah 7 — Uji routing lengkap (faq/dokumen/fallback)

**Tujuan:** menguji apakah chatbot memilih jawaban yang tepat.

```powershell
& ".\.venv\Scripts\python.exe" -m src.router
```

- **Hasilnya:** `reports/e2e_val.md`.
- **Tanda berhasil:** muncul `akurasi route` (angka 0–1, makin besar makin baik).
- **Mau coba satu pertanyaan?** Gunakan:
  ```powershell
  & ".\.venv\Scripts\python.exe" -m src.router --ask "berapa kuota peserta angkatan"
  ```

### Langkah 8 — Isi database jawaban & cek kesiapan

**Tujuan:** memasukkan jawaban baku ke database, lalu memeriksa semua siap.

```powershell
& ".\.venv\Scripts\python.exe" -m scripts.seed_responses
& ".\.venv\Scripts\python.exe" -m scripts.preflight
```

- **Hasilnya:** database `data/chatbot.db` terisi; laporan kesiapan.
- **Tanda berhasil:** `preflight` menulis **SIAP** (kalau ada `GAGAL`, perbaiki
  item yang ditandai — biasanya `.env` belum ada atau model belum dilatih).

### Langkah 9 — EVALUASI FINAL (ujian akhir — hanya sekali!)

**Tujuan:** mengukur performa akhir pada test set. **Hanya dijalankan sekali**
dan hanya saat Anda benar-benar siap.

```powershell
& ".\.venv\Scripts\python.exe" -m src.final_eval --confirm "JALANKAN EVALUASI FINAL"
```

- **Hasilnya:** `reports/final_eval.md` + catatan di `EXPERIMENTS.md`.
- **Tanda berhasil:** muncul angka `accuracy`, `f1_macro`, dll.
- ⚠️ Tanpa frasa persis di atas, perintah **ditolak** dan test set tidak tersentuh.

> **Kenapa hanya sekali?** Kalau test set dipakai berulang-ulang untuk menyetel,
> hasilnya jadi "bocor" dan tidak jujur. Jadi ia dikunci sampai benar-benar final.

### Alternatif otomatis (Langkah 1–8 sekaligus)

Bila sudah yakin data benar, Anda boleh memakai satu skrip yang menjalankan
urutan di atas:

```powershell
.\deploy\build_artifacts.ps1
```

---

## C. Menjalankan layanan chatbot (opsional)

Setelah Langkah 8 berhasil, jalankan layanan uji:

```powershell
.\deploy\run_api.ps1
```

- Layanan berjalan di `http://127.0.0.1:8000`. **Biarkan jendela terbuka.**
- Untuk menghentikan: tekan **Ctrl+C** di jendela itu.
- Cek "sehat": jalankan `.\deploy\healthcheck.ps1`.
- Pintu yang tersedia: `GET /health` (bebas), `GET /intents`, dan `POST /chat`
  (perlu header `X-API-KEY` berisi nilai `CHATBOT_API_KEY` dari `.env`).

> Layanan ini **untuk uji coba saja**. Jangan dijadikan layanan permanen /
> dipasang sebagai service Windows tanpa izin pengelola server.

---

## D. Checklist (centang saat selesai)

- [ ] `.env` dibuat & `CHATBOT_API_KEY` diisi
- [ ] `data/raw/intents_raw.csv` diisi (cukup baris per intent)
- [ ] `data/intents.yaml` diisi (11 intent)
- [ ] `data/pdf/` diisi PDF berbasis teks
- [ ] `data/rag_eval/{qa_pairs,in_doc,out_doc}.csv` diisi
- [ ] `data/e2e/{mixed_val,mixed_test}.csv` diisi
- [x] Model AI (3 buah) sudah tersedia di lokal
- [ ] Langkah 1–8 selesai & berhasil
- [ ] Evaluasi final dijalankan **sekali** dengan frasa persis
- [ ] Hasil dilaporkan ke pengembang / dosen pembimbing

---

## E. Masalah umum & solusinya

| Gejala | Penyebab & solusi |
|--------|-------------------|
| `python` tidak dikenal / versi salah | Jangan pakai `python`. Pakai `& ".\.venv\Scripts\python.exe" -m ...` |
| Tidak bisa menjalankan `Activate.ps1` (diblokir) | Tidak perlu aktivasi. Cukup pakai `& ".\.venv\Scripts\python.exe" -m ...` seperti di panduan. |
| Muncul `Exited with code 1` padahal tidak ada error nyata | Kadang peringatan dari pustaka AI dianggap error oleh PowerShell. Cek **GAGAL/SIAP** atau laporan, bukan angka kode. |
| Berkas jadi `intents_raw.csv.txt` | Ekstensi tersembunyi. Aktifkan **File name extensions** di File Explorer (Bagian A, tips 1). |
| Data aneh/berubah setelah dibuka di Excel | Excel menyimpan sebagai ANSI. Simpan ulang sebagai **CSV UTF-8**. |
| Kolom bergeser / `intent` salah baca | Ada koma di dalam kalimat tanpa tanda kutip. Bungkus kalimat dengan `"..."`. |
| `leakage_check` menandai kebocoran | Ada kalimat terlalu mirip di train & val. Perbaiki data, ulangi dari Langkah 2. |
| `preflight` menulis `GAGAL` | Perbaiki item yang ditandai: biasanya `.env` belum ada, atau model/indeks belum dibuat (Langkah 4/6 belum jalan). |
| PDF tidak terbaca | PDF hasil pindai (gambar). Ganti dengan PDF berbasis teks. |
| Perintah lama sekali di Langkah 4 | Normal (tanpa GPU). Biarkan berjalan. |

---

## F. Aturan penting (versi sederhana)

1. **Data asli saja.** Jangan mengarang data atau hasil. Data sintetis hanya
   untuk uji coba, ditandai `synthetic`, dan tidak dilaporkan sebagai hasil penelitian.
2. **Test set terkunci.** Hanya program evaluasi final yang boleh membukanya,
   dan hanya dengan frasa `JALANKAN EVALUASI FINAL`.
3. **Jangan campur** data latih dengan data uji.
4. **Tanpa AI generatif.** Model hanya *menggolongkan* pertanyaan dan *mencari*
   jawaban di dokumen — jawaban FAQ berasal dari `data/intents.yaml`, bukan dikarang.
5. **Jangan ubah** `config.yaml` (terutama `models`, `intents`, `seeds`) tanpa
   persetujuan.

### Dua hal yang masih menunggu keputusan pemilik proyek
1. Aturan pasti kapan pertanyaan dianggap "di luar cakupan" (berkaitan dengan tau1).
2. Penyesuaian ambang kecocokan dokumen (tau2) untuk model embedder — saat ini
   perlu disetel lebih rendah atau memakai skor BM25/RRF.

---

## G. Tabel perintah ringkas (salin-tempel)

| Tujuan | Perintah |
|--------|----------|
| Uji Python | `& ".\.venv\Scripts\python.exe" --version` |
| Periksa data | `& ".\.venv\Scripts\python.exe" -m src.validate_data` |
| Bagi + kunci test | `& ".\.venv\Scripts\python.exe" -m src.split` |
| Tambah variasi | `& ".\.venv\Scripts\python.exe" -m src.augment` |
| Cek kebocoran | `& ".\.venv\Scripts\python.exe" -m src.leakage_check` |
| Uji pembanding klasik | `& ".\.venv\Scripts\python.exe" -m src.baseline` |
| Latih (cari setelan) | `& ".\.venv\Scripts\python.exe" -m src.train --mode search --grid reduced` |
| Latih (3 seed) | `& ".\.venv\Scripts\python.exe" -m src.train --mode multiseed --grid reduced --lr 2e-5 --batch 16` |
| Kalibrasi | `& ".\.venv\Scripts\python.exe" -m src.calibrate --model-dir models/distilbert-base-indonesian` |
| Indeks PDF | `& ".\.venv\Scripts\python.exe" -m rag.index --pdf-dir data/pdf --out vectorstore` |
| Uji pencarian PDF | `& ".\.venv\Scripts\python.exe" -m rag.eval_retrieval --method rrf` |
| Uji routing | `& ".\.venv\Scripts\python.exe" -m src.router` |
| Isi jawaban DB | `& ".\.venv\Scripts\python.exe" -m scripts.seed_responses` |
| Cek kesiapan | `& ".\.venv\Scripts\python.exe" -m scripts.preflight` |
| Evaluasi FINAL | `& ".\.venv\Scripts\python.exe" -m src.final_eval --confirm "JALANKAN EVALUASI FINAL"` |
| Uji kode program | `& ".\.venv\Scripts\python.exe" -m pytest -q` |

---

> Bingung atau macet di langkah mana pun? Catat **nomor langkahnya** dan
> **salin pesan terakhir di layar**, lalu tanyakan ke pengembang. Jangan
> mengubah berkas lain secara acak untuk "mencoba-coba".
