# Baseline TF-IDF + LinearSVC [SANDBOX - uji pipeline, BUKAN hasil penelitian]

- train: train_aug.csv
- val: val.csv
- durasi latih+prediksi: 0.02s

## Metrik (val)

| metrik | nilai |
|--------|-------|
| accuracy | 0.7200 |
| f1_macro | 0.6641 |
| f1_weighted | 0.6670 |
| precision_macro | 0.6424 |
| recall_macro | 0.7273 |

## Laporan per intent

```
                      precision    recall  f1-score   support

        alur_seleksi       1.00      1.00      1.00         2
     jadwal_kegiatan       1.00      1.00      1.00         3
    ketentuan_luaran       1.00      1.00      1.00         2
        kontak_admin       0.50      0.33      0.40         3
        out_of_scope       0.00      0.00      0.00         2
pengeluaran_keuangan       0.50      1.00      0.67         2
review_revisi_berkas       1.00      1.00      1.00         2
              sapaan       0.00      0.00      0.00         2
  syarat_pendaftaran       0.67      0.67      0.67         3
       tanya_dokumen       0.40      1.00      0.57         2
        terima_kasih       1.00      1.00      1.00         2

            accuracy                           0.72        25
           macro avg       0.64      0.73      0.66        25
        weighted avg       0.65      0.72      0.67        25
```
