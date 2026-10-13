# Laporan Kalibrasi (temperature scaling) & ambang tau1

> **[SANDBOX] Data: validasi saja.** Test set TIDAK disentuh.

- Folder model: `C:\Nitip Ragil\Chatbot\models\smoke`
- Data val: `C:\Nitip Ragil\Chatbot\data\sandbox\processed\val.csv`
- Suhu terkalibrasi **T = 0.3254**
- ECE sebelum kalibrasi: **0.3026**
- ECE sesudah kalibrasi: **0.1826**

## Kriteria pemilihan tau1

- `target_accepted_accuracy_min` = 0.95
- `target_oos_false_accept_max` = 0.05
- Intent OOS: `out_of_scope`

**Hasil: tau1 = 0.40** (status: `ok`)

- coverage = 0.0400
- accepted_accuracy = 1.0000
- oos_false_accept = 0.0000

## Tabel telusur tau1

| tau1 | coverage | accepted_accuracy | oos_false_accept | n_accepted |
|------|----------|-------------------|------------------|------------|
| 0.40 | 0.0400 | 1.0000 | 0.0000 | 1 |
| 0.45 | 0.0000 | 0.0000 | 0.0000 | 0 |
| 0.50 | 0.0000 | 0.0000 | 0.0000 | 0 |
| 0.55 | 0.0000 | 0.0000 | 0.0000 | 0 |
| 0.60 | 0.0000 | 0.0000 | 0.0000 | 0 |
| 0.65 | 0.0000 | 0.0000 | 0.0000 | 0 |
| 0.70 | 0.0000 | 0.0000 | 0.0000 | 0 |
| 0.75 | 0.0000 | 0.0000 | 0.0000 | 0 |
| 0.80 | 0.0000 | 0.0000 | 0.0000 | 0 |
| 0.85 | 0.0000 | 0.0000 | 0.0000 | 0 |
| 0.90 | 0.0000 | 0.0000 | 0.0000 | 0 |
| 0.95 | 0.0000 | 0.0000 | 0.0000 | 0 |

> Catatan: pemilihan ambang dilakukan HANYA pada val. Test set dipakai
> nanti di `src/final_eval.py` atas perintah eksplisit.
