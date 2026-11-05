# Evaluasi pada test set terverifikasi manusia

Sudah diverifikasi: 1002 dari 2396 ulasan test (SEBAGIAN).
Ulasan yang dicek BUKAN sampel acak: proporsi positifnya 30%, sedangkan seluruh test 35%. Lihat kolom 'Di test' vs 'Dicek' di bawah.

## 1. Audit label dataset (label dataset vs label terverifikasi)

| Sumber label | Di test | Dicek | Benar | Akurasi label (CI 95%) | Diubah | Dibuang |
|---|---|---|---|---|---|---|
| lexicon+llm | 1682 | 606 | 605 | 100.00% (99.37% - 100.00%) | 0 | 1 |
| llm (leksikon ambigu) | 412 | 94 | 94 | 100.00% (96.07% - 100.00%) | 0 | 0 |
| manual: claude_style | 19 | 19 | 19 | 100.00% (83.18% - 100.00%) | 0 | 0 |
| manual: rule:ikut Claude | 271 | 271 | 271 | 100.00% (98.60% - 100.00%) | 0 | 0 |
| manual: user | 12 | 12 | 12 | 100.00% (75.75% - 100.00%) | 0 | 0 |
| **Semua** | 2396 | 1002 | 1001 | 100.00% (99.62% - 100.00%) | 0 | 1 |

Cohen's kappa label dataset vs terverifikasi: 1.000

## 2a. Seluruh test setelah koreksi verifikasi (1001 dicek manusia): 2395 ulasan, positif 35%

| Model | Kelas | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|
| Baseline | negatif | 0.9922 | 0.9801 | 0.9861 | 1555 |
| Baseline | positif | 0.9639 | 0.9857 | 0.9747 | 840 |
| + Augmentasi | negatif | 0.9878 | 0.9916 | 0.9897 | 1555 |
| + Augmentasi | positif | 0.9844 | 0.9774 | 0.9809 | 840 |

| Model | Accuracy | Macro F1 | Precision positif | Recall positif | F1 positif | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| Baseline | 0.9820 | 0.9804 | 0.9639 | 0.9857 | 0.9747 | 828 | 1524 | 31 | 12 |
| + Augmentasi | 0.9866 | 0.9853 | 0.9844 | 0.9774 | 0.9809 | 821 | 1542 | 13 | 19 |

Uji McNemar (exact): baseline benar & augmented salah = 14, kebalikannya = 25, p-value = 0.1081 -> TIDAK berbeda signifikan (alpha = 0,05)

## 2b. Hanya ulasan yang sudah dicek manusia: 1001 ulasan, positif 30%

| Model | Kelas | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|
| Baseline | negatif | 0.9884 | 0.9743 | 0.9813 | 700 |
| Baseline | positif | 0.9421 | 0.9734 | 0.9575 | 301 |
| + Augmentasi | negatif | 0.9815 | 0.9843 | 0.9829 | 700 |
| + Augmentasi | positif | 0.9632 | 0.9568 | 0.9600 | 301 |

| Model | Accuracy | Macro F1 | Precision positif | Recall positif | F1 positif | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| Baseline | 0.9740 | 0.9694 | 0.9421 | 0.9734 | 0.9575 | 293 | 682 | 18 | 8 |
| + Augmentasi | 0.9760 | 0.9714 | 0.9632 | 0.9568 | 0.9600 | 288 | 689 | 11 | 13 |

Uji McNemar (exact): baseline benar & augmented salah = 9, kebalikannya = 11, p-value = 0.8238 -> TIDAK berbeda signifikan (alpha = 0,05)

## 3. Jumlah salah prediksi per sumber label (seluruh test setelah koreksi)

| Sumber label | Ulasan | Salah baseline | Salah augmented |
|---|---|---|---|
| lexicon+llm | 1681 | 16 | 3 |
| llm (leksikon ambigu) | 412 | 9 | 8 |
| manual: claude_style | 19 | 1 | 1 |
| manual: rule:ikut Claude | 271 | 16 | 19 |
| manual: user | 12 | 1 | 1 |