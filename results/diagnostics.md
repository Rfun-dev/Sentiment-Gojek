# Diagnostik model: kebocoran, tingkat kesulitan, dan pembanding

Test: 2395 ulasan (label terkoreksi verifikasi; 840 positif, 1555 negatif). Selang kepercayaan 95% = bootstrap 1.000 kali.

## 1. Cek kebocoran

| Pemeriksaan | Hasil |
|---|---|
| ID test yang muncul di train/val | 0 |
| Teks test identik dengan train/val | 0 |
| Teks test identik dengan data sintetis | 0 |
| Ulasan test hampir kembar (>= 0.9) dengan train/val | 0 |
| Ulasan test hampir kembar (>= 0.9) dengan data sintetis | 0 |

## 2. Pembanding sederhana

| Model | Accuracy (CI 95%) | Macro F1 (CI 95%) |
|---|---|---|
| Kelas mayoritas (selalu negatif) | 0.6493 (0.6301-0.6668) | 0.3937 (0.3865-0.4001) |
| InSet Lexicon (ambigu -> mayoritas) | 0.7545 (0.7374-0.7708) | 0.6962 (0.6759-0.7156) |
| TF-IDF kata 1-2 gram + regresi logistik | 0.9628 (0.9553-0.9699) | 0.9590 (0.9509-0.9667) |
| TF-IDF karakter 3-5 gram + regresi logistik | 0.9608 (0.9532-0.9687) | 0.9569 (0.9486-0.9657) |
| IndoBERT baseline | 0.9820 (0.9766-0.9875) | 0.9804 (0.9747-0.9862) |
| IndoBERT + augmentasi | 0.9866 (0.9820-0.9916) | 0.9853 (0.9801-0.9906) |

Jika model sederhana sudah tinggi, tugasnya memang mudah; selisih IndoBERT dengan TF-IDF adalah nilai tambah sebenarnya dari model bahasa.

## 3. Skor per lapisan kesulitan

| Lapisan | Ulasan | Model | Accuracy (CI 95%) | Macro F1 (CI 95%) |
|---|---|---|---|---|
| Seluruh test | 2395 | baseline | 0.9820 (0.9766-0.9871) | 0.9804 (0.9746-0.9859) |
| Seluruh test | 2395 | augmented | 0.9866 (0.9820-0.9908) | 0.9853 (0.9802-0.9901) |
| Sumber lexicon+llm (leksikon & Claude sepakat) | 1681 | baseline | 0.9905 (0.9857-0.9946) | 0.9865 (0.9799-0.9925) |
| Sumber lexicon+llm (leksikon & Claude sepakat) | 1681 | augmented | 0.9982 (0.9958-1.0000) | 0.9974 (0.9940-1.0000) |
| Sumber llm (leksikon ambigu) | 412 | baseline | 0.9782 (0.9636-0.9903) | 0.9717 (0.9525-0.9879) |
| Sumber llm (leksikon ambigu) | 412 | augmented | 0.9806 (0.9660-0.9927) | 0.9752 (0.9566-0.9910) |
| Sumber manual (leksikon != Claude, sulit) | 302 | baseline | 0.9404 (0.9106-0.9636) | 0.9401 (0.9096-0.9636) |
| Sumber manual (leksikon != Claude, sulit) | 302 | augmented | 0.9305 (0.9007-0.9570) | 0.9303 (0.9001-0.9569) |
| Kemiripan dengan train >= 0,7 | 370 | baseline | 0.9973 (0.9919-1.0000) | 0.9960 (0.9866-1.0000) |
| Kemiripan dengan train >= 0,7 | 370 | augmented | 0.9919 (0.9811-1.0000) | 0.9881 (0.9725-1.0000) |
| Kemiripan dengan train 0,5-0,7 | 667 | baseline | 0.9880 (0.9790-0.9955) | 0.9880 (0.9790-0.9955) |
| Kemiripan dengan train 0,5-0,7 | 667 | augmented | 0.9895 (0.9820-0.9970) | 0.9895 (0.9820-0.9970) |
| Kemiripan dengan train < 0,5 (benar-benar baru) | 1358 | baseline | 0.9750 (0.9654-0.9831) | 0.9552 (0.9380-0.9690) |
| Kemiripan dengan train < 0,5 (benar-benar baru) | 1358 | augmented | 0.9838 (0.9764-0.9904) | 0.9701 (0.9569-0.9823) |
| Test sulit = manual ATAU kemiripan < 0,5 | 1467 | baseline | 0.9755 (0.9673-0.9823) | 0.9628 (0.9508-0.9732) |
| Test sulit = manual ATAU kemiripan < 0,5 | 1467 | augmented | 0.9823 (0.9755-0.9884) | 0.9725 (0.9614-0.9827) |

Uji McNemar baseline vs augmentasi per lapisan:

| Lapisan | Baseline benar, aug salah | Baseline salah, aug benar | p-value | p Bonferroni |
|---|---|---|---|---|
| Seluruh test | 14 | 25 | 0.1081 | - (uji utama) |
| Sumber lexicon+llm (leksikon & Claude sepakat) | 2 | 15 | 0.0023 | 0.0164 |
| Sumber llm (leksikon ambigu) | 5 | 6 | 1.0000 | 1.0000 |
| Sumber manual (leksikon != Claude, sulit) | 7 | 4 | 0.5488 | 1.0000 |
| Kemiripan dengan train >= 0,7 | 2 | 0 | 0.5000 | 1.0000 |
| Kemiripan dengan train 0,5-0,7 | 4 | 5 | 1.0000 | 1.0000 |
| Kemiripan dengan train < 0,5 (benar-benar baru) | 8 | 20 | 0.0357 | 0.2499 |
| Test sulit = manual ATAU kemiripan < 0,5 | 10 | 20 | 0.0987 | 0.6912 |

Uji utama adalah 'Seluruh test'. 7 lapisan lain diuji sekaligus, sehingga peluang 'signifikan' karena kebetulan ikut naik: pakai kolom p Bonferroni (p x jumlah lapisan). Lapisan dibuat setelah melihat hasil, jadi sifatnya eksploratif, bukan bukti utama.

## 4. Rating bintang vs label (deskriptif; rating TIDAK dipakai model)

| Bintang | Negatif | Positif |
|---|---|---|
| 1 | 1086 | 2 |
| 2 | 171 | 0 |
| 3 | 152 | 2 |
| 4 | 58 | 36 |
| 5 | 88 | 800 |

Aturan sederhana 'bintang >= 4 = positif' (tanpa bintang 3): akurasi 0.9340.