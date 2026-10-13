# Laporan Evaluasi Routing End-to-End (val)

> **[SANDBOX] Data: validasi saja.** Test set TIDAK disentuh.

- Folder model: `C:\Nitip Ragil\Chatbot\models\smoke`
- Vectorstore: `C:\Nitip Ragil\Chatbot\data\sandbox\vectorstore_dense`
- Jawaban FAQ: `C:\Nitip Ragil\Chatbot\data\sandbox\intents_sandbox.yaml`
- Metode retrieval: `rrf`
- tau1 (keyakinan) = 0.4000
- tau2 (dokumen) = 0.30

**Akurasi route = 0.3333** (n = 12)

## Metrik per-route

| route | precision | recall | f1 | support |
|-------|-----------|--------|----|---------|
| faq | 0.0000 | 0.0000 | 0.0000 | 4 |
| dokumen | 0.0000 | 0.0000 | 0.0000 | 4 |
| fallback | 0.3333 | 1.0000 | 0.5000 | 4 |

## Confusion matrix (baris = aktual, kolom = prediksi)

| aktual \ prediksi | faq | dokumen | fallback |
|---|---|---|---|
| **faq** | 0 | 0 | 4 |
| **dokumen** | 0 | 0 | 4 |
| **fallback** | 0 | 0 | 4 |

## Salah route (8 dari 12)

| pertanyaan | aktual | prediksi | intent | keyakinan | alasan |
|------------|--------|----------|--------|-----------|--------|
| halo selamat pagi admin | faq | fallback | alur_seleksi | 0.1973 | low_confidence |
| terima kasih atas informasinya | faq | fallback | alur_seleksi | 0.2156 | low_confidence |
| bagaimana cara menghubungi admin inkubator | faq | fallback | kontak_admin | 0.3025 | low_confidence |
| apa saja syarat pendaftaran tenant | faq | fallback | ketentuan_luaran | 0.2345 | low_confidence |
| berapa kuota peserta setiap angkatan | dokumen | fallback | ketentuan_luaran | 0.2180 | low_confidence |
| kapan klinik keuangan dijadwalkan | dokumen | fallback | ketentuan_luaran | 0.1881 | low_confidence |
| bagaimana alur seleksi dan verifikasi berkas | dokumen | fallback | ketentuan_luaran | 0.2471 | low_confidence |
| apa ketentuan luaran akhir program inkubasi | dokumen | fallback | ketentuan_luaran | 0.3104 | low_confidence |

> Catatan: evaluasi dilakukan pada val; test set menunggu `final_eval.py`.
