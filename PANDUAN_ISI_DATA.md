# Panduan Mengisi & Mengoreksi Data Asli

Dokumen ini menjelaskan **apa yang harus Anda isi**, **formatnya**, **di mana
menaruhnya**, dan **cara mengoreksi** bila ada kesalahan — khusus untuk 4 hal:

1. `data/raw/intents_raw.csv` — pertanyaan latih (paling penting)
2. `data/intents.yaml` — jawaban baku chatbot
3. `data/pdf/*.pdf` — dokumen mitra
4. `data/rag_eval/` — berkas uji pencarian dokumen (3 berkas)

> Semua berkas disimpan dalam pengodean **UTF-8**. Paling mudah disunting dengan
> **VS Code**. Bila memakai **Excel**, pilih *Save As* → **CSV UTF-8 (Comma
> delimited) (\*.csv)**.

---

## Aturan umum (berlaku untuk semua berkas)

- **Baris pertama (header) harus ditulis persis**, termasuk huruf kecil dan
  tanda koma. Jangan menambah/mengganti nama kolom.
- **Satu baris = satu data.** Jangan ada baris kosong di tengah tabel.
- **Kalau teks mengandung koma (`,`), bungkus dengan tanda kutip ganda (`"`).**
  Contoh: `"apa syaratnya, tolong jelaskan",syarat_pendaftaran,asli`.
  (Excel melakukan ini otomatis.)
- **Jangan menambah kolom** di luar yang diminta (kolom tambahan boleh diabaikan
  tapi bisa membingungkan — lebih baik tidak ada).
- Simpan di **lokasi yang ditentukan** (lihat tiap bagian). Nama berkas harus
  tepat (jangan sampai tertambah `.txt`).
- **Data asli saja.** Jangan mengarang pertanyaan/jawaban.

---

## 1. `data/raw/intents_raw.csv` — pertanyaan latih

### Ada di mana?
Buat berkas baru bernama `intents_raw.csv` di dalam folder `data/raw/`.
(Ada contoh kerangka di `data/raw/intents_raw.template.csv`.)

### Format

| Kolom | Wajib? | Isi |
|-------|--------|-----|
| `text` | **Ya** | Satu pertanyaan pengguna (bahasa Indonesia, kalimat asli/alami). |
| `intent` | **Ya** | Salah satu dari **11 nama intent** (lihat tabel di bawah). |
| `source` | Opsional | Tulis `asli` untuk menandai data sungguhan. |

Header persis:

```
text,intent,source
```

### 11 nama intent yang sah (huruf kecil, pakai garis bawah)

| Intent | Untuk pertanyaan tentang… | Minimal baris |
|--------|--------------------------|---------------|
| `syarat_pendaftaran` | syarat & berkas pendaftaran tenant | 20 |
| `alur_seleksi` | tahapan/proses seleksi calon tenant | 20 |
| `jadwal_kegiatan` | jadwal kegiatan & pelatihan | 20 |
| `ketentuan_luaran` | luaran/hasil yang harus dicapai | 20 |
| `pengeluaran_keuangan` | aturan biaya, reimbursement, klaim dana | 20 |
| `review_revisi_berkas` | revisi/peninjauan berkas & proposal | 20 |
| `kontak_admin` | cara menghubungi admin/pengelola | 20 |
| `sapaan` | ucapan pembuka | 20 |
| `terima_kasih` | ucapan terima kasih | 20 |
| `tanya_dokumen` | minta/bertanya isi dokumen atau PDF | **40** |
| `out_of_scope` | di luar cakupan layanan inkubator | **100** |

> Jumlah minimal di bawah ini hanya menghasilkan **peringatan**, bukan error.
> Tapi makin banyak & makin bervariasi, makin baik akurasi model.

### Aturan isi (dicek otomatis)

| Kondisi | Status | Tindakan |
|---------|--------|----------|
| Kolom `text`/`intent` hilang atau salah tulis | **ERROR** | Perbaiki header. |
| Ada baris dengan `text` atau `intent` kosong | **ERROR** | Hapus/isi baris itu. |
| `intent` bukan salah satu dari 11 nama | **ERROR** | Ganti ke nama yang benar (typo?). |
| Teks sama diberi **dua intent berbeda** | **ERROR** | Pilih satu intent yang benar. |
| Teks duplikat (sama persis) | peringatan | Hapus salah satu. |
| Jumlah per intent kurang dari target | peringatan | Tambah contoh. |
| Panjang teks < 3 atau > 300 karakter | peringatan | Perbaiki. |

### Ide pertanyaan per intent (untuk membuat banyak variasi)

Pakai daftar ini sebagai pemicu ide, lalu tulis dengan kalimat yang berbeda-beda
(gaya formal, santai, singkat, panjang):

- `syarat_pendaftaran`: "apa syarat mendaftar?", "dokumen apa saja yang
  disiapkan?", "perlu proposal tidak?", "syarat untuk tim berapa orang?"
- `alur_seleksi`: "bagaimana tahapan seleksi?", "berapa lama prosesnya?",
  "apa itu presentasi produk?", "urutan administrasi sampai wawancara?"
- `jadwal_kegiatan`: "kapan pelatihan dimulai?", "jadwal mentoring kapan?",
  "tanggal penting yang harus diingat?", "berlangsung berapa lama?"
- `ketentuan_luaran`: "apa luaran yang wajib?", "harus ada produk jadi?",
  "target akhir program?", "format laporan akhir?"
- `pengeluaran_keuangan`: "bagaimana klaim reimbursement?", "transport
  ditanggung?", "bukti apa untuk klaim dana?", "batas maksimal pengeluaran?"
- `review_revisi_berkas`: "cara revisi proposal?", "batas waktu revisi?",
  "siapa yang meninjau berkas?", "format revisi yang benar?"
- `kontak_admin`: "kontak admin?", "nomor panitia?", "email resmi?",
  "jam layanan admin?"
- `sapaan`: "halo", "selamat pagi", "hai", "assalamualaikum"
- `terima_kasih`: "terima kasih", "makasih ya", "thanks", "siap, terima kasih"
- `tanya_dokumen`: "tolong carikan dokumen panduan", "apa isi PDF jadwal?",
  "minta file ketentuan luaran", "lihat dokumen pendaftaran"
- `out_of_scope`: "siapa presiden Indonesia?", "resep nasi goreng",
  "harga tiket kereta", "cuaca besok" (hal di luar layanan inkubator)

### Contoh kerangka (salin-tempel, lalu ganti/tambah sampai cukup)

```csv
text,intent,source
apa saja syarat mendaftar tenant inkubator,syarat_pendaftaran,asli
dokumen apa yang harus saya siapkan untuk pendaftaran,syarat_pendaftaran,asli
bagaimana tahapan seleksi calon tenant,alur_seleksi,asli
berapa lama proses seleksi berlangsung,alur_seleksi,asli
kapan jadwal pelatihan angkatan berikutnya,jadwal_kegiatan,asli
apa luaran yang wajib dihasilkan peserta,ketentuan_luaran,asli
bagaimana cara klaim reimbursement biaya kegiatan,pengeluaran_keuangan,asli
bagaimana cara merevisi proposal yang ditolak,review_revisi_berkas,asli
berapa nomor kontak admin inkubator,kontak_admin,asli
selamat pagi,sapaan,asli
terima kasih,terima_kasih,asli
tampilkan dokumen panduan pendaftaran,tanya_dokumen,asli
siapa juara piala dunia tahun lalu,out_of_scope,asli
```

> Ini baru contoh kecil (13 baris). Lanjutkan menambah baris sampai tiap intent
> memenuhi jumlah minimal pada tabel di atas. Semua baris harus **benar** dan
> konsisten — tidak ada baris yang sengaja dibuat salah.

Cara mengoreksi bila ada yang tidak sesuai: lihat tabel **"Cara mengoreksi
(petakan pesan laporan → tindakan)"** di bawah.

### Cara mengisi
1. **VS Code** (disarankan): buat `data/raw/intents_raw.csv`, tulis header,
   lalu tambah baris demi baris. **Ctrl+S** menyimpan.
2. **Excel**: baris pertama isi `text`, `intent`, `source`; baris berikutnya
   pertanyaan. Selain itu **Save As → CSV UTF-8 (Comma delimited)**.

### Cara memeriksa
```powershell
& ".\.venv\Scripts\python.exe" -m src.validate_data
```
Buka `reports/data_validation.md`. Status harus **LULUS (tanpa error)**.
Peringatan boleh ada, tapi sebaiknya ditindaklanjuti.

### Cara mengoreksi (petakan pesan laporan → tindakan)

| Pesan di laporan | Artinya | Yang harus dilakukan |
|------------------|---------|----------------------|
| `Kolom wajib hilang: ['text', 'intent']` | Header salah/kurang | Pastikan baris pertama persis `text,intent,source`. |
| `Ada N baris dengan text/intent kosong.` | Ada baris kosong | Hapus baris itu atau isi nilainya. |
| `Ada N baris dengan intent tak dikenal: [...]` | Nama intent typo | Ganti dengan salah satu dari 11 nama (perhatikan huruf kecil & `_`). |
| `Ada N teks dengan lebih dari satu label` | Satu kalimat diberi 2 intent | Pilih satu intent yang paling tepat. |
| `Ada N teks duplikat` (peringatan) | Kalimat sama muncul ≥2× | Hapus salah satu. |
| `Intent 'x' hanya punya N contoh` (peringatan) | Belum cukup | Tambah pertanyaan untuk intent itu. |
| `Ada N teks lebih pendek/panjang dari ...` (peringatan) | Teks aneh | Perpendek/perpanjang atau hapus. |

Setelah memperbaiki, **jalankan lagi** `src.validate_data`.

---

## 2. `data/intents.yaml` — jawaban baku chatbot

### Ada di mana?
Buat berkas baru `intents.yaml` di dalam folder `data/`.
(Ada contoh di `data/intents.template.yaml`.)

### Format
Berkas berisi **daftar** (diawali tanda `-`) sebanyak **11 entri**. Setiap entri
punya:

| Kunci | Wajib? | Isi |
|-------|--------|-----|
| `intent` | **Ya** | Salah satu dari 11 nama intent (harus sama). |
| `deskripsi` | Opsional | Penjelasan singkat. |
| `jawaban` | Sangat disarankan | Teks jawaban baku yang ditampilkan chatbot. |

### Kerangka lengkap 11 intent (salin-tempel, ganti `jawaban`)

```yaml
- intent: syarat_pendaftaran
  deskripsi: Pertanyaan tentang syarat dan berkas pendaftaran tenant.
  jawaban: "Tulis di sini jawaban resmi: daftar syarat, berkas, dan ketentuan pendaftaran."

- intent: alur_seleksi
  deskripsi: Pertanyaan tentang tahapan seleksi calon tenant.
  jawaban: "Tulis di sini tahapan seleksi dari awal sampai akhir beserta perkiraan waktunya."

- intent: jadwal_kegiatan
  deskripsi: Pertanyaan tentang jadwal kegiatan dan pelatihan.
  jawaban: "Tulis di sini jadwal/agenda kegiatan atau tautan ke jadwal resminya."

- intent: ketentuan_luaran
  deskripsi: Pertanyaan tentang luaran yang harus dihasilkan peserta.
  jawaban: "Tulis di sini luaran/hasil wajib peserta inkubasi."

- intent: pengeluaran_keuangan
  deskripsi: Pertanyaan tentang aturan pengeluaran dan reimbursement.
  jawaban: "Tulis di sini aturan biaya, batas, dan prosedur reimbursement."

- intent: review_revisi_berkas
  deskripsi: Pertanyaan tentang revisi dan peninjauan berkas.
  jawaban: "Tulis di sini prosedur revisi berkas/proposal dan batas waktunya."

- intent: kontak_admin
  deskripsi: Pertanyaan tentang cara menghubungi admin/pengelola.
  jawaban: "Tulis di sini kontak resmi: email, nomor, jam layanan."

- intent: sapaan
  deskripsi: Sapaan pembuka.
  jawaban: "Halo! Selamat datang. Ada yang bisa saya bantu terkait program inkubator?"

- intent: terima_kasih
  deskripsi: Ucapan terima kasih.
  jawaban: "Sama-sama! Semoga informasi ini membantu."

- intent: tanya_dokumen
  deskripsi: Permintaan dokumen/PDF mitra.
  jawaban: "Sebutkan dokumen yang Anda cari, saya akan bantu temukan kutipannya."

- intent: out_of_scope
  deskripsi: Pertanyaan di luar cakupan layanan FAQ inkubator.
  jawaban: "Maaf, pertanyaan itu di luar cakupan layanan ini. Silakan hubungi admin."
```

### Aturan YAML (penting agar tidak error)
- Indentasi **2 spasi**; **jangan pakai tombol Tab**.
- Setiap entri diawali `- intent:` (tanda minus + spasi).
- Bungkus `jawaban` dengan tanda kutip ganda bila berisi tanda titik dua (`:`)
  atau diawali karakter khusus. Aman: selalu pakai tanda kutip ganda.
- Untuk jawaban **panjang beberapa baris**, gunakan `|`:

  ```yaml
  - intent: syarat_pendaftaran
    deskripsi: Syarat pendaftaran.
    jawaban: |
      Syarat pendaftaran:
      1. Mengisi formulir online.
      2. Melampirkan proposal usaha.
      3. Fotokopi KTP ketua tim.
  ```

- Nama setelah `intent:` **harus identik** dengan 11 daftar (perhatikan
  huruf kecil dan `_`).

### Cara memeriksa
```powershell
& ".\.venv\Scripts\python.exe" -m scripts.seed_responses
```
Kalau berhasil, jawaban masuk ke database `data/chatbot.db`. Bila ada masalah
YAML, akan muncul error saat itu (biasanya soal indentasi).

### Cara mengoreksi

| Gejala | Penyebab | Perbaikan |
|--------|----------|-----------|
| `mapping values are not allowed here` / error YAML | Salah indentasi atau tanda `:` di dalam teks tanpa kutip | Rapikan indentasi 2 spasi; bungkus `jawaban` dengan `"..."` |
| Chatbot menjawab kosong untuk suatu intent | Entri intent itu belum ada / nama salah | Tambahkan entri dengan nama yang benar |
| Jawaban tampil terpotong | Tanda kutip tidak berpasangan | Pastikan kutip buka & tutup lengkap |
| Muncul `tab` error | Memakai Tab | Ganti semua Tab jadi 2 spasi |

---

## 3. `data/pdf/*.pdf` — dokumen mitra

### Ada di mana?
Salin semua PDF ke folder `data/pdf/`.

### Aturan nama berkas (PENTING)
- Nama berkas inilah yang dipakai di `rag_eval/qa_pairs.csv` (kolom `doc`).
  Jadi **jangan mengganti nama PDF** setelah Anda mengisi `qa_pairs.csv`.
- Sebaiknya: huruf kecil, tanpa spasi, pakai garis bawah.
  Contoh yang baik: `panduan_pendaftaran.pdf`, `jadwal_program.pdf`,
  `ketentuan_luaran.pdf`.
- Hindari karakter aneh (`#`, `%`, `&`, tanda petik) di nama berkas.

### Syarat isi
- PDF harus **berisi teks** yang bisa dipilih/di-cari.
- PDF hasil **pindai (scan)** = gambar → **tidak terbaca** (OCR/Tesseract tidak
  dipasang). Ganti dengan PDF digital, atau salin ulang teksnya.

**Cara cepat mengecek PDF berbasis teks:** buka PDF → tekan **Ctrl+F** → cari
satu kata yang ada di halaman → kalau kata itu tersorot, berarti ada teksnya.

### Nomor halaman
- Penomoran halaman **mulai dari 1** (sama seperti penomoran di pembaca PDF).
- Halaman yang dirujuk di `qa_pairs.csv` adalah halaman **tempat jawaban berada**.

### Cara memeriksa
```powershell
& ".\.venv\Scripts\python.exe" -m rag.index --pdf-dir data/pdf --out vectorstore
```
- Hasilnya folder `vectorstore/`.
- Bila muncul peringatan "tidak ada teks terekstrak", berarti ada PDF hasil
  pindai yang perlu diganti.

### Cara mengoreksi
- Peringatan "kemungkinan PDF hasil scan" → ganti PDF itu dengan versi digital.
- Nama berkas berubah setelah `qa_pairs.csv` dibuat → perbarui juga kolom `doc`
  di `qa_pairs.csv` agar cocok.

---

## 4. `data/rag_eval/` — berkas uji pencarian dokumen

Ada **3 berkas** (salin dari templat `*.template.csv`, buang `.template`):

| Berkas | Header (baris pertama) | Gunanya |
|--------|------------------------|---------|
| `qa_pairs.csv` | `question,doc,page` | Mengukur apakah pencarian menemukan dokumen + halaman yang tepat |
| `in_doc.csv` | `question` | Pertanyaan yang jawabannya **ada** di PDF |
| `out_doc.csv` | `question` | Pertanyaan yang jawabannya **tidak ada** di PDF |

### 4a. `qa_pairs.csv` — pertanyaan + lokasi jawaban

| Kolom | Isi |
|-------|-----|
| `question` | Pertanyaan uji (harus bisa dijawab dari salah satu PDF). |
| `doc` | **Nama berkas PDF persis**, termasuk `.pdf` (mis. `panduan_pendaftaran.pdf`). |
| `page` | Nomor halaman (**angka**, mulai dari 1) tempat jawaban berada. |

Aturan penting:
- `doc` harus **sama persis** dengan nama berkas di `data/pdf/`
  (huruf besar/kecil berpengaruh). Salah sedikit → dianggap tidak ketemu.
- `page` harus halaman yang **benar-benar memuat** jawaban.
- `page` = angka tanpa tanda kutip (mis. `2`, bukan `"2"` atau `hal. 2`).

Contoh:

```csv
question,doc,page
berapa biaya pendaftaran program,panduan_pendaftaran.pdf,2
apa saja jadwal pelatihan tahun ini,jadwal_program.pdf,1
ketentuan luaran akhir program apa saja,ketentuan_luaran.pdf,3
```

### 4b. `in_doc.csv` — jawabannya ada di PDF

Header: `question`. Isi: pertanyaan yang **pasti** terjawab oleh isi PDF.
Tujuannya mengukur seberapa besar skor kemiripan untuk pertanyaan yang cocok.

```csv
question
berapa biaya pendaftaran program
syarat apa saja untuk mendaftar
```

### 4c. `out_doc.csv` — jawabannya tidak ada di PDF

Header: `question`. Isi: pertanyaan yang **tidak** terjawab oleh PDF — sebaiknya
topik **dekat tapi tidak ada** (pengecoh/hard negative), bukan sekadar
pertanyaan acak. Tujuannya memastikan chatbot tidak asal mengutip dokumen.

```csv
question
berapa gaji karyawan inkubator tahun lalu
bagaimana cara mendaftar beasiswa luar negeri
prosedur pengajuan kredit bank untuk UMKM
```

### Berapa banyak baris?
Tidak ada batas wajib, tetapi makin banyak makin andal. Saran minimal:

| Berkas | Saran jumlah baris |
|--------|--------------------|
| `qa_pairs.csv` | ≥ 15 |
| `in_doc.csv` | ≥ 15 |
| `out_doc.csv` | ≥ 15 (campur dengan pengecoh sulit) |

### Cara memeriksa
```powershell
& ".\.venv\Scripts\python.exe" -m rag.index --pdf-dir data/pdf --out vectorstore
& ".\.venv\Scripts\python.exe" -m rag.eval_retrieval --method rrf
```
Buka `reports/rag_eval.md`, perhatikan **MRR** dan **recall@k** (makin dekat ke
1,0 makin baik).

### Cara mengoreksi (bila hasil jelek)

| Gejala | Kemungkinan penyebab | Perbaikan |
|--------|----------------------|-----------|
| MRR rendah | `doc` salah tulis / beda huruf besar-kecil | Samakan persis dengan nama berkas di `data/pdf/` |
| MRR rendah | `page` salah (sering meleset 1 halaman) | Buka PDF, cari jawaban, catat nomor halaman yang benar |
| Pertanyaan tidak ketemu | Jawaban sebenarnya tidak ada di PDF | Perbaiki pertanyaan, atau pindahkan ke `out_doc.csv` |
| Semua skor rendah | PDF hasil pindai (tidak ada teks) | Ganti PDF dengan versi digital (lihat bagian 3) |
| tau2 aneh | Data eval terlalu sedikit | Tambah baris `in_doc.csv` & `out_doc.csv` |

> Catatan teknis: tiap halaman dipotong jadi potongan ±90 kata. Jadi jawaban
> sebaiknya berada pada halaman yang Anda tulis — bukan halaman sebelum/sesudah.

---

## 5. Urutan memeriksa setelah mengisi (ringkas)

```powershell
# 1. Periksa data latih
& ".\.venv\Scripts\python.exe" -m src.validate_data

# 2. Periksa PDF + bangun indeks
& ".\.venv\Scripts\python.exe" -m rag.index --pdf-dir data/pdf --out vectorstore

# 3. Periksa rag_eval (MRR/recall)
& ".\.venv\Scripts\python.exe" -m rag.eval_retrieval --method rrf

# 4. Periksa jawaban baku masuk ke database
& ".\.venv\Scripts\python.exe" -m scripts.seed_responses
```

Perbaiki bila ada **ERROR** (di laporan validasi) atau **MRR rendah**
(di laporan retrieval), lalu ulangi.

---

## 6. Ringkasan lokasi & header

| Isi | Lokasi tepat | Header wajib |
|-----|--------------|--------------|
| Pertanyaan latih | `data/raw/intents_raw.csv` | `text,intent,source` |
| Jawaban baku | `data/intents.yaml` | (YAML: `- intent:` / `jawaban:`) |
| Dokumen mitra | `data/pdf/*.pdf` | — |
| Uji pencarian (lokasi jawaban) | `data/rag_eval/qa_pairs.csv` | `question,doc,page` |
| Uji pencarian (ada di PDF) | `data/rag_eval/in_doc.csv` | `question` |
| Uji pencarian (tidak ada di PDF) | `data/rag_eval/out_doc.csv` | `question` |

---

## 7. Checklist keempat bagian

**intents_raw.csv**
- [ ] Berada di `data/raw/intents_raw.csv`
- [ ] Header persis `text,intent,source`
- [ ] Tiap intent memenuhi jumlah minimal (termasuk `out_of_scope` ≥ 100)
- [ ] Tidak ada intent typo, baris kosong, atau teks berlabel ganda
- [ ] `src.validate_data` → **LULUS**

**intents.yaml**
- [ ] Berada di `data/intents.yaml`
- [ ] Berisi 11 entri (nama intent sama persis)
- [ ] Semua `jawaban` sudah diisi jawaban resmi
- [ ] Indentasi 2 spasi, tanpa Tab
- [ ] `scripts.seed_responses` berhasil

**data/pdf**
- [ ] Semua PDF berbasis teks (bisa di-Ctrl+F)
- [ ] Nama berkas huruf kecil & tidak diubah setelah `qa_pairs.csv` dibuat

**data/rag_eval**
- [ ] `qa_pairs.csv` header `question,doc,page`, `doc` sama persis dengan nama PDF
- [ ] `page` benar (mulai dari 1)
- [ ] `in_doc.csv` & `out_doc.csv` header `question`
- [ ] `rag.eval_retrieval` menunjukkan MRR yang baik (mendekati 1,0)
