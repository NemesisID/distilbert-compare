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
| 0 - Persiapan | ⏳ tertunda | Lingkungan terlapor; TERHAMBAT: Python 3.15 (butuh 3.10-3.12) |
| 1 - Kerangka & data | ✅ kode & tes siap | 22 tes pytest lulus (dijalankan dgn shim YAML sementara) |

---

## Catatan eksekusi

### 2026-10-13 - Fase 1 (uji pipeline, SANDBOX)

- Perintah: `python -m src.split --sandbox --out-dir data/sandbox/processed`
- Hasil **[SANDBOX]**: train=115, val=25, test=25 (total 165, 11 intent x 15).
- Deduplikasi: 0 baris dibuang.
- Test set sandbox dikunci (read-only), SHA-256:
  `c4f5d2824e1debbb9b76b59aeccf88937f5f8331ecc5f5f38b59842318aead95`
- `python -m pytest -q` -> 22 passed.
- Catatan: angka ini dari data sintetis, BUKAN hasil penelitian.


---

## Log eksperimen

<!-- Baris baru ditambahkan otomatis di bawah ini -->

| Tanggal | Fase/Skrip | Model | Konfigurasi | Seed | Metrik | Durasi |
|---------|------------|-------|-------------|------|--------|--------|
