# Uji ketahanan split (baseline IndoBERT)

Pertanyaan: apakah skor ±98% bertahan bila split dibuat lebih ketat? Ketiga varian memakai aturan 1:3 dan seed yang sama; isi test berbeda, jadi bandingkan pola, bukan angka persis. TF-IDF + regresi logistik dilatih di train varian yang sama sebagai ukuran tingkat kesulitan test.

| Varian | Test (neg/pos) | Rentang tanggal test | Test mirip train >= 0,9 | >= 0,8 | TF-IDF macro F1 | IndoBERT accuracy | IndoBERT macro F1 | Akurasi sumber manual |
|---|---|---|---|---|---|---|---|---|
| Split utama (group, kemiripan 0,9) | 1556/840 | 05-04-2026 s.d. 23-10-2026 | 0 | 182 | 0.9590 | 0.9821 | 0.9804 | 0.9404 |
| Group split lebih ketat (kemiripan 0,8) | 1549/716 | 05-04-2026 s.d. 23-10-2026 | 0 | 0 | 0.9552 | 0.9810 | 0.9779 | 0.9200 |
| Split menurut waktu (test = 20% terbaru) | 1822/579 | 14-09-2026 s.d. 23-10-2026 | 50 | 143 | 0.9392 | 0.9758 | 0.9676 | 0.9144 |

Catatan: split utama memakai label test terkoreksi untuk 1 ulasan yang dibuang; varian lain membuang ulasan itu dari dataset. Split waktu tidak memakai pengelompokan ulasan mirip, karena ulasan berpola sama memang muncul lagi di dunia nyata.