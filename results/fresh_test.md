# Test ulasan baru berlabel buta

Ulasan Gojek yang terbit setelah ulasan terakhir dataset (2026-10-23 s.d. 2026-10-26), diambil acak 200 ulasan. Peneliti melabeli dari teks asli tanpa melihat rating, label leksikon/Claude, atau prediksi model. Model tidak dilatih ulang.

* Netral / campur / tidak jelas (`-`, tidak dihitung): 6 (3.0%) - ulasan seperti ini juga dibuang dari dataset saat pelabelan
* Dievaluasi: 194 ulasan (135 negatif, 59 positif)

## 1. Kinerja

Selang kepercayaan 95%: akurasi = Wilson, macro F1 = bootstrap 1.000 kali.

| Model | Accuracy (CI 95%) | Macro F1 (CI 95%) | Recall positif | FP | FN |
|---|---|---|---|---|---|
| TF-IDF kata 1-2 gram + regresi logistik | 0.9433 (0.9013-0.9680) | 0.9320 (0.8859-0.9694) | 0.8814 | 4 | 7 |
| IndoBERT baseline (seed 42) | 0.9588 (0.9208-0.9790) | 0.9508 (0.9148-0.9809) | 0.9153 | 3 | 5 |
| IndoBERT augmented (seed 42) | 0.9588 (0.9208-0.9790) | 0.9503 (0.9155-0.9815) | 0.8983 | 2 | 6 |
| IndoBERT baseline (seed 1) | 0.9639 (0.9274-0.9824) | 0.9563 (0.9239-0.9860) | 0.8983 | 1 | 6 |
| IndoBERT augmented (seed 1) | 0.9588 (0.9208-0.9790) | 0.9498 (0.9130-0.9813) | 0.8814 | 1 | 7 |
| IndoBERT baseline (seed 2) | 0.9536 (0.9142-0.9754) | 0.9438 (0.9073-0.9761) | 0.8814 | 2 | 7 |
| IndoBERT augmented (seed 2) | 0.9588 (0.9208-0.9790) | 0.9508 (0.9151-0.9794) | 0.9153 | 3 | 5 |

### Setelah koreksi peneliti (angka tambahan)

2 label dikoreksi (no 64, 90) setelah prediksi terlihat. Yang dicek hanya ulasan yang salah diprediksi, jadi angka ini cenderung menguntungkan model; angka utama tetap tabel di atas.

| Model | Accuracy (CI 95%) | Macro F1 (CI 95%) | Recall positif | FP | FN |
|---|---|---|---|---|---|
| TF-IDF kata 1-2 gram + regresi logistik | 0.9536 (0.9142-0.9754) | 0.9438 (0.9017-0.9767) | 0.9123 | 4 | 5 |
| IndoBERT baseline (seed 42) | 0.9691 (0.9342-0.9858) | 0.9627 (0.9306-0.9879) | 0.9474 | 3 | 3 |
| IndoBERT augmented (seed 42) | 0.9691 (0.9342-0.9858) | 0.9623 (0.9298-0.9878) | 0.9298 | 2 | 4 |
| IndoBERT baseline (seed 1) | 0.9742 (0.9411-0.9889) | 0.9685 (0.9381-0.9934) | 0.9298 | 1 | 4 |
| IndoBERT augmented (seed 1) | 0.9691 (0.9342-0.9858) | 0.9619 (0.9282-0.9884) | 0.9123 | 1 | 5 |
| IndoBERT baseline (seed 2) | 0.9639 (0.9274-0.9824) | 0.9558 (0.9202-0.9833) | 0.9123 | 2 | 5 |
| IndoBERT augmented (seed 2) | 0.9691 (0.9342-0.9858) | 0.9627 (0.9282-0.9874) | 0.9474 | 3 | 3 |

## 2. McNemar baseline vs augmentasi (seed sama)

| Seed | Baseline benar, aug salah | Baseline salah, aug benar | p-value |
|---|---|---|---|
| 42 | 2 | 2 | 1.0000 |
| 1 | 1 | 0 | 1.0000 |
| 2 | 1 | 2 | 1.0000 |

## 3. Ulasan yang salah diprediksi (seed 42)

| No | Teks asli | Label manusia | Baseline | Augmented |
|---|---|---|---|---|
| 27 | apk nya oke,cuman kadang sebel sama driver pas nganter makanan kan ya,uangnya lebih tapi dia ngomongnya pas....stres btw aku di Cilacap tepatnya di kroya wkwk | 0 | 0 | 1 |
| 64 | definisi nyuruh Kelaperan sih nih aplikasi. pengantaran murah di gofood bikin stress bgt ,di batalin ga bisa,,di tunggu PHP., daripada PHP mening di ilangin aja tuh pengantaran gofood murahh. 👊 | 1 | 0 | 0 |
| 71 | oke, semoga semakin baik, tapi semuanya jadi mahal 🥲 | 0 | 1 | 1 |
| 80 | Aplikasi yang pake Orang Dewasa tapi hadiah bonusnya hadiah untuk anak² bahkan jika saya dapat hadiahnya lalu saya berikan ke anak tentu saja anak itu tidak mengerti ini hadiah apa. Wajar² saja lah kalau kasih hadiah. | 1 | 0 | 0 |
| 90 | parah banget delay food nya | 1 | 0 | 0 |
| 114 | aplikasi ya suka² ngesedot | 0 | 1 | 0 |
| 123 | gobang gocir , gojek sampe ngacir | 1 | 0 | 0 |
| 145 | dpt info dr gojek dpt piinjam dr gojek eh pas dibuka malah di tolak. cm sekali doang dpt pinjaman pembayaran jg gak telat sblm jth tempo dilunasin eh malah bgni. ok lh makasih | 0 | 1 | 0 |
| 170 | Saya suka Gojek, tetapi cara kerja sistem alamatnya perlu diperbaiki. Jangan sampai setiap kali pengguna ingin memesan, aplikasi terus menggunakan alamat lama secara otomatis tanpa memberikan pilihan yang jelas untuk menentukan alamat baru. Seharusnya, Gojek bisa mengembangkan sistem yang lebih cerdas dengan memberikan rekomendasi alamat berdasarkan lokasi pengguna saat ini. Jika pengguna berada di tempat yang berbeda dari sebelumnya, aplikasi harus menawarkan pilihan untuk menggunakan alamat b | 1 | 1 | 0 |
| 190 | plis bang gojek tutor jadi cowo greenflag soalnya aku udah berusaha jadi cwo greenflag sama aja masih di sia'in🗿😆 | 1 | 0 | 0 |