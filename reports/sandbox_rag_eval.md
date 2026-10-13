# Laporan Evaluasi Retrieval PDF & ambang tau2

> **[SANDBOX] Data: evaluasi retrieval saja.** Test set klasifikasi TIDAK disentuh.

- Metode retrieval: `rrf`
- Jumlah chunk: 4
- Jumlah dokumen: 2
- Dense aktif: True
- Model embedding: `sentence-transformers/distiluse-base-multilingual-cased-v2`

## Metrik qa_pairs (doc + halaman emas)

- n = 4
- MRR = 1.0000

| k | recall@k |
|---|----------|
| 1 | 1.0000 |
| 2 | 1.0000 |
| 3 | 1.0000 |

## Kriteria pemilihan tau2

- `tau2_accept_recall_min` = 0.9
- `tau2_false_accept_max` = 0.05

**Hasil: tau2 = 0.30** (status: `relaxed`)

- in_doc_recall = 0.0000 (n=4)
- out_doc_false_accept = 0.0000 (n=4)

## Tabel telusur tau2

| tau2 | in_doc_recall | out_doc_false_accept |
|------|---------------|----------------------|
| 0.30 | 0.0000 | 0.0000 |
| 0.35 | 0.0000 | 0.0000 |
| 0.40 | 0.0000 | 0.0000 |
| 0.45 | 0.0000 | 0.0000 |
| 0.50 | 0.0000 | 0.0000 |
| 0.55 | 0.0000 | 0.0000 |
| 0.60 | 0.0000 | 0.0000 |
| 0.65 | 0.0000 | 0.0000 |
| 0.70 | 0.0000 | 0.0000 |
| 0.75 | 0.0000 | 0.0000 |
| 0.80 | 0.0000 | 0.0000 |
| 0.85 | 0.0000 | 0.0000 |
| 0.90 | 0.0000 | 0.0000 |

> Catatan: tau2 ditentukan dari data eval retrieval, bukan test set.

> **Peringatan skala:** ambang tau2 terendah pada grid (0.30) pun belum mencapai target in-doc recall (0.0000 < 0.9). Model embedding `distiluse-base-multilingual-cased-v2` menurunkan kemiripan saat jawaban tertanam dalam chunk panjang. Perlu menurunkan `rag.tau2_min` (mis. ke 0.05) atau memakai skor metode lain; keputusan ada pada pemilik proyek.
