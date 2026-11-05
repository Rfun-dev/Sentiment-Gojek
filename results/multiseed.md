# Ringkasan beberapa seed training

Seed selesai: 42, 1, 2

## Per seed

| Seed | Metrik | Baseline | + Augmentasi | Selisih |
|---|---|---|---|---|
| 42 | Accuracy | 0.9821 | 0.9866 | +0.0046 |
| 42 | Macro F1 | 0.9804 | 0.9853 | +0.0049 |
| 42 | Positif precision | 0.9639 | 0.9844 | +0.0205 |
| 42 | Positif recall | 0.9857 | 0.9774 | -0.0083 |
| 42 | Positif F1 | 0.9747 | 0.9809 | +0.0062 |
| 42 | McNemar p | | | 0.1081 |
| 1 | Accuracy | 0.9821 | 0.9829 | +0.0008 |
| 1 | Macro F1 | 0.9803 | 0.9812 | +0.0009 |
| 1 | Positif precision | 0.9738 | 0.9784 | +0.0046 |
| 1 | Positif recall | 0.9750 | 0.9726 | -0.0024 |
| 1 | Positif F1 | 0.9744 | 0.9755 | +0.0011 |
| 1 | McNemar p | | | 0.8450 |
| 2 | Accuracy | 0.9871 | 0.9862 | -0.0008 |
| 2 | Macro F1 | 0.9857 | 0.9849 | -0.0009 |
| 2 | Positif precision | 0.9879 | 0.9798 | -0.0082 |
| 2 | Positif recall | 0.9750 | 0.9810 | +0.0060 |
| 2 | Positif F1 | 0.9814 | 0.9804 | -0.0011 |
| 2 | McNemar p | | | 0.8450 |

## Rata-rata ± simpangan baku (3 seed)

| Metrik | Baseline | + Augmentasi | Selisih rata-rata | Seed augmentasi lebih baik |
|---|---|---|---|---|
| Accuracy | 0.9837 ± 0.0029 | 0.9853 ± 0.0021 | +0.0015 ± 0.0028 | 2/3 |
| Macro F1 | 0.9821 ± 0.0031 | 0.9838 ± 0.0023 | +0.0016 ± 0.0030 | 2/3 |
| Positif precision | 0.9752 ± 0.0121 | 0.9809 ± 0.0031 | +0.0057 ± 0.0144 | 2/3 |
| Positif recall | 0.9786 ± 0.0062 | 0.9770 ± 0.0042 | -0.0016 ± 0.0072 | 1/3 |
| Positif F1 | 0.9768 ± 0.0040 | 0.9789 ± 0.0030 | +0.0021 ± 0.0037 | 2/3 |

Seed dengan perbedaan signifikan (McNemar, alpha 0,05): 0/3.
Simpangan baku dengan sedikit seed masih kasar; laporkan sebagai indikasi kestabilan, bukan uji formal.