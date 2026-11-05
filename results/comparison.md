# Laporan eksperimen

## Konfigurasi

| Parameter | Nilai |
|---|---|
| SEED | 42 |
| MODEL_NAME | indobenchmark/indobert-base-p1 |
| MAX_LEN | 128 |
| BATCH_SIZE | 16 |
| MICRO_BATCH | 8 |
| LR | 2e-05 |
| EPOCHS | 3 |
| WEIGHT_DECAY | 0.01 |
| PATIENCE | 1 |
| TEST_SIZE | 0.2 |
| VAL_SIZE | 0.1 |
| DEDUP_SIM | 0.9 |
| TARGET_RATIO | 0.3333 |

## Jumlah data per kelas (0 = negatif, 1 = positif)

| Partisi | Negatif | Positif | Total |
|---|---|---|---|
| train_baseline | 5543 | 1847 | 7390 |
| train_augmented | 5543 | 5543 | 11086 |
| val | 606 | 363 | 969 |
| test | 1556 | 840 | 2396 |

Kelas minoritas: **positif**; data minoritas yang dibuang dari train: 1251

## Metrik per kelas (data test)

| Model | Kelas | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|
| Baseline | negatif | 0.9922 | 0.9801 | 0.9861 | 1556 |
| Baseline | positif | 0.9639 | 0.9857 | 0.9747 | 840 |
| + Augmentasi | negatif | 0.9878 | 0.9916 | 0.9897 | 1556 |
| + Augmentasi | positif | 0.9844 | 0.9774 | 0.9809 | 840 |

## Confusion matrix (positif = kelas 1)

| Model | TP | TN | FP | FN | Total salah |
|---|---|---|---|---|---|
| Baseline | 828 | 1525 | 31 | 12 | 43 |
| + Augmentasi | 821 | 1543 | 13 | 19 | 32 |

## Perbandingan

| Metrik | Baseline | + Augmentasi | Selisih |
|---|---|---|---|
| Accuracy | 0.9821 | 0.9866 | +0.0046 |
| Macro Precision | 0.9781 | 0.9861 | +0.0081 |
| Macro Recall | 0.9829 | 0.9845 | +0.0016 |
| Macro F1 | 0.9804 | 0.9853 | +0.0049 |
| Minoritas (positif) precision | 0.9639 | 0.9844 | +0.0205 |
| Minoritas (positif) recall | 0.9857 | 0.9774 | -0.0083 |
| Minoritas (positif) f1 | 0.9747 | 0.9809 | +0.0062 |

## Uji signifikansi McNemar (exact)

- Baseline benar, augmented salah: 14
- Baseline salah, augmented benar: 25
- p-value = 0.1081 -> TIDAK berbeda signifikan (alpha = 0,05)