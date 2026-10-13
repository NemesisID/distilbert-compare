# Deployment (berkas contoh)

Folder ini berisi **berkas contoh** untuk menyiapkan/menjalankan layanan
chatbot di server. **Tidak ada berkas di sini yang dijalankan otomatis** oleh
kode, tes, maupun `scripts/` — semuanya dipanggil manual oleh operator.

> ⚠️ **Aturan penting**
> - Jangan menjalankan **layanan permanen** (Windows Service) tanpa izin pemilik
>   proyek/server. Contoh di `windows-service/` hanya templat.
> - Jangan mengubah **firewall**, IIS, registry, atau kebijakan sistem tanpa izin.
> - Rahasia (kunci API) **hanya** lewat `.env`. `.env` tidak pernah di-commit.
> - Bind API ke `127.0.0.1`; biarkan website utama (reverse proxy) yang
>   meneruskan permintaan.

---

## Berkas

| Berkas | Guna |
|--------|------|
| `run_api.ps1` | Menjalankan uvicorn di latar depan (Ctrl+C untuk berhenti). |
| `healthcheck.ps1` | Memanggil `GET /health`. |
| `build_artifacts.ps1` | Alur penyiapan data -> model -> seed -> indeks (untuk data asli). |
| `backup_db.ps1` | Menyalin database SQLite bertanggal ke `backups/`. |
| `env.production.example` | Contoh nilai `.env` produksi. |
| `windows-service/` | Templat layanan Windows (**contoh**, jangan dijalankan tanpa izin). |

---

## Alur singkat

```powershell
# 0. Sekali saja: siapkan .env dari contoh
Copy-Item .env.example .env
#    lalu isi CHATBOT_API_KEY dengan token acak yang panjang

# 1. Pra-terbang: pastikan artefak lengkap
& .\.venv\Scripts\python.exe -m scripts.preflight

# 2. Siapkan artefak (HANYA saat data asli sudah tersedia)
.\deploy\build_artifacts.ps1

# 3. Jalankan layanan di latar depan untuk uji
.\deploy\run_api.ps1

# 4. Di terminal lain, cek kesehatan
.\deploy\healthcheck.ps1
```

Mode sandbox (uji pipeline, tanpa data asli):

```powershell
& .\.venv\Scripts\python.exe -m scripts.preflight --sandbox
```

---

## Catatan

- API memuat model + indeks sekali saat startup; bila salah satu tidak ada,
  `/chat` mengembalikan **503** sampai artefak siap.
- `DATABASE_URL`, `CORS_ORIGINS`, dan `RATE_LIMIT_PER_MINUTE` dibaca dari `.env`
  (lihat `.env.example`).
- `logs/` dan `backups/` tidak di-commit (lihat `.gitignore`).
