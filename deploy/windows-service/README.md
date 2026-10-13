# Layanan Windows (templat)

> ⚠️ **Hanya contoh. JANGAN dijalankan tanpa izin pemilik proyek/server.**
> Memasang layanan Windows adalah **perubahan sistem** (butuh Administrator)
> dan menjadikan proses **permanen**. Proyek ini **tidak** menjalankannya.

Berkas di sini menunjukkan cara membungkus perintah:

```powershell
.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

menjadi layanan Windows. Ada dua opsi umum:

| Opsi | Berkas | Alat |
|------|--------|------|
| NSSM | `install_service.nssm.ps1.example` | [NSSM](https://nssm.cc) |
| WinSW | `chatbot-api.winsw.xml.example` | [WinSW](https://github.com/winsw/winsw) |

Sebelum memakai layanan permanen, pertimbangkan:

1. **Mockup/uji** cukup pakai `deploy\run_api.ps1` (latar depan).
2. Pastikan `.env` sudah ada dan `scripts.preflight` **SIAP**.
3. Arahkan log ke `logs/` dan aktifkan rotasi.
4. Batasi bind ke `127.0.0.1`; reverse proxy (IIS/nginx) yang menghadap publik.
5. Minta izin tertulis sebelum memasang/mengaktifkan layanan di server lab.
