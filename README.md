# Analisis Sentimen Ulasan Gojek dengan IndoBERT

Proyek Tugas Akhir: klasifikasi sentimen **positif / negatif** pada ulasan aplikasi Gojek
(Google Play Store) dengan **fine-tuning IndoBERT**, **pelabelan hibrida** (leksikon + LLM + manual),
dan **augmentasi data dengan LLM**. README ini cukup untuk memahami seluruh cara kerja proyek.

---

## 1. Gambaran singkat

```
 scrape 20.000 ulasan
        │
 preprocessing (bersihkan, huruf kecil, normalisasi slang)
        │
 pelabelan:  InSet Lexicon ──┐
                             ├─ sama / leksikon ambigu? ── ya ──> pakai label LLM
             LLM (Claude)  ──┘                         └── tidak ──> MANUAL (berbantuan LLM)
        │
 group split stratified: train 72% | validasi 8% | test 20% (ulasan hampir kembar tidak dipisah)
        │
 (positif:negatif kurang timpang → buang sebagian kelas MINORITAS di TRAIN sampai 1:3)
        │
        ├── Skenario 1 BASELINE : train apa adanya
        └── Skenario 2 AUGMENTED: train + data buatan LLM untuk kelas minoritas
        │
 fine-tuning IndoBERT (dilatih 2 kali, satu per skenario)
        │
 evaluasi di data test yang SAMA → bandingkan
```

Pertanyaan penelitian: **apakah augmentasi LLM pada kelas minoritas menaikkan recall & F1 kelas minoritas?**

---

## 2. Struktur folder

| File | Fungsi |
|---|---|
| `config.py` | **Semua pengaturan** (jumlah data, hyperparameter, ambang, dll). Ubah di sini saja |
| `utils.py` | Fungsi yang dipakai beberapa tahap: preprocessing, skor InSet, kemiripan & split, statistik (McNemar, Wilson, bootstrap), pembanding TF-IDF |
| `download_resources.py` | Unduh InSet & IndoCollex ke `resources/` |
| `01_scrape.py` … `12_fresh_test.py` | Tahapan pipeline, dijalankan berurutan (09 = beberapa seed, 10 = diagnostik, 11 = uji ketahanan split, 12 = test ulasan baru) |
| `data/` | Semua file data perantara (CSV) |
| `models/` | Model hasil training (`baseline/`, `augmented/`) |
| `results/` | Metrik (`*_metrics.json`), prediksi, `comparison.md`, dan `comparison_verified.md` |

---

## 3. Cara menjalankan

Persiapan sekali saja:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124   # PyTorch versi GPU
pip install -r requirements.txt
python download_resources.py        # unduh kamus
```

Lalu jalankan berurutan:

```bash
python 01_scrape.py                 # ambil 20.000 ulasan           -> data/reviews_raw.csv
python 02_preprocess.py             # bersihkan teks                -> data/reviews_clean.csv
python 03a_claude_export.py         # ulasan -> data/claude_in/ (dilabeli Claude ke data/claude_out/, lihat Tahap 3)
python 03b_claude_import.py         # label Claude                  -> data/llm_labels.csv
python 03_label_auto.py             # leksikon vs LLM               -> data/labeled_auto.csv, data/manual_label.csv
#   >>> BUKA data/manual_label.csv di Excel, isi kolom manual_label (lihat bagian 5) <<<
python 04_label_merge.py            # gabung label otomatis+manual  -> data/labeled.csv
python 05_split_augment.py          # split; simpan train_baseline/val/test; ekspor teks utk augmentasi -> data/aug_in/
#   >>> Claude menulis parafrase ke data/aug_out/ (lihat Tahap 5) <<<
python 05b_claude_aug_import.py     # impor parafrase               -> data/augmented_cache.csv
python 05_split_augment.py          # jalankan lagi                 -> data/train_augmented.csv
python 06_train.py --mode baseline  # training skenario 1 (boleh jalan sebelum augmentasi selesai)
python 06_train.py --mode augmented # training skenario 2
python 07_compare.py                # tabel perbandingan            -> results/comparison.md
python 08_evaluate_verified.py      # pertama kali: buat data/test_verification.csv
#   >>> isi kolom verified (cek label test oleh manusia, lihat Tahap 8) <<<
python 08_evaluate_verified.py      # evaluasi pada label terverifikasi -> results/comparison_verified.md
python 10_diagnostics.py            # cek kebocoran, pembanding sederhana, skor per lapisan -> results/diagnostics.md
python 11_robust_splits.py          # buat split uji ketahanan -> data/robust_dedup08/, data/robust_waktu/
DATA_DIR=data/robust_dedup08 python 06_train.py --mode baseline --tag robust_dedup08
DATA_DIR=data/robust_waktu   python 06_train.py --mode baseline --tag robust_waktu
python 11_robust_splits.py          # laporan uji ketahanan -> results/robustness.md
python 12_fresh_test.py scrape      # ulasan terbaru, sampel acak 200 -> data/fresh/fresh_labels.csv
#   >>> isi kolom label tanpa melihat rating/label/prediksi apa pun (lihat Tahap 12) <<<
python 12_fresh_test.py evaluate    # test ulasan baru berlabel buta -> results/fresh_test.md
```

Setiap tahap boleh diulang sendiri. Hasil Claude tersimpan sebagai file (`llm_labels.csv`, `augmented_cache.csv`),
jadi menjalankan ulang tahap 03–05 memberi hasil yang sama tanpa melabeli atau memparafrasekan ulang.

---

## 4. Penjelasan tiap tahap

### Tahap 1 — Scraping (`01_scrape.py`)
Memakai `google-play-scraper` (`com.gojek.app`, bahasa & negara Indonesia, urut terbaru), mengambil
20.000 ulasan per 200 ulasan. Kolom disimpan: `id, content, rating, date`.
> Catatan: karena urut *terbaru*, periode ulasan mengikuti **kapan kamu menjalankan scraping**
> (bukan otomatis Januari–Februari 2026 seperti di proposal). Sesuaikan penulisan di laporan.

### Tahap 2 — Preprocessing (`02_preprocess.py`)
1. **Cleaning**: hapus URL, mention/hashtag, angka, dan semua karakter non-huruf (diganti spasi supaya kata tidak menempel).
2. **Case folding**: semua huruf kecil.
3. **Normalisasi slang**: kata tidak baku diganti bentuk bakunya lewat kamus **IndoCollex** (`bgt → banget`).
   Kata negasi (`gak, ga, gk, nggak, tdk…`) dipaksa menjadi `tidak` agar konsisten dengan InSet.
4. Ulasan **< 3 kata** dibuang, lalu **duplikat** dibuang.
Tidak ada stemming / stopword removal (sesuai proposal: BERT butuh kalimat utuh).
Dari 20.000 ulasan mentah, tersisa sekitar **12.800** (banyak ulasan Gojek sangat pendek, mis. "bagus").

### Tahap 3 — Pelabelan otomatis (`03_label_auto.py`)
Dua "juri" melabeli setiap ulasan:
* **InSet Lexicon**: jumlahkan bobot kata (-5...+5). Skor >= 1 -> positif, <= -1 -> negatif, di antaranya -> *ambigu*.
  InSet mentah ternyata bermasalah (kata seperti "bagus" ada di daftar positif DAN negatif, dan kata fungsi seperti "yang" bernilai -5),
  sehingga `utils.load_inset()` membuang kata yang muncul di kedua daftar dan sebuah daftar kecil kata fungsi (`NOISE_WORDS`).
  Tanpa perbaikan ini, ~2.900 ulasan jelas positif keliru dinilai negatif oleh leksikon. Catat ini di laporan.
* **LLM (Claude, tanpa API)**: membaca teks **asli** (agar emoji/sarkasme terbaca), menjawab positif / negatif / netral.
  `03a_claude_export.py` memecah ulasan jadi 43 potongan (`data/claude_in/`), Claude melabeli per potongan (`data/claude_out/`),
  lalu `03b_claude_import.py` memasukkannya ke `data/llm_labels.csv`, yang dibaca `03_label_auto.py`.
  Hasilnya divalidasi terhadap rating bintang: ulasan bintang 1-2 dilabeli negatif ~98%, dan tidak ada potongan yang labelnya bergeser.
  (API Gemini sempat dicoba, tetapi free tier hanya ~20 permintaan/hari, jadi tidak praktis untuk 12.000+ ulasan dan kodenya sudah dihapus.)

Aturan keputusan:
| Leksikon | LLM | Hasil |
|---|---|---|
| positif | positif | label **1** otomatis |
| negatif | negatif | label **0** otomatis |
| ambigu | positif / negatif | ikut label **LLM** (aturan dilonggarkan) |
| bertentangan (positif vs negatif) atau LLM netral | | masuk `manual_label.csv` |

Skrip juga mencetak **Cohen's kappa** antara leksikon dan LLM (bisa dikutip di laporan sebagai tingkat kesepakatan).

### Tahap 4 — Anotasi manual & penggabungan (`04_label_merge.py`)
Lihat bagian 5. Hasilnya `data/labeled.csv` berisi `text, label, source` (`lexicon+llm`, `llm (leksikon ambigu)`, atau `manual`).
Skrip mencetak jumlah per kelas dan **rasio minoritas/mayoritas**.

**Yang terjadi di proyek ini** (kolom `label_source` di `manual_label.csv` mencatat asal tiap label):
| Asal label baris manual | Jumlah | Keterangan |
|---|---|---|
| `user` | 80 | dilabeli peneliti sendiri (8 blok × 10) |
| `rule:ikut Claude` | 1.395 | leksikon vs Claude bertentangan → ikut Claude; pola ini cocok 44/44 dengan label peneliti, lalu 50 sampel acak dicek ulang peneliti dan dinyatakan benar |
| `claude_style` | 832 | Claude menjawab netral → dinilai Claude meniru gaya peneliti (±90% dibuang, sisanya jelas keluhan/pujian) |

Hasil akhir: **12.006 ulasan berlabel** (7.705 negatif, 4.301 positif). Di laporan, sebut ini *pelabelan hibrida berbantuan LLM*,
bukan anotasi manual penuh.

**Audit label (buta).** 100 ulasan acak dari `labeled.csv` (50 per kelas, seed 2026, `data/audit_sample.csv`) dilabeli ulang
oleh peneliti **tanpa melihat label dataset**. Hasil: 99 dari 99 yang diberi label cocok (100%, Cohen's kappa 1,000;
batas bawah selang kepercayaan 95% Wilson = 96,3%), per sumber label: lexicon+llm 65/65, llm (leksikon ambigu) 21/21, manual 13/13; 1 ulasan campuran (pujian + keluhan) dibuang.
Koreksi audit disimpan di `data/label_corrections.csv` (kolom `id, label`; `1`/`0` mengganti label, `-` membuang) dan diterapkan
otomatis oleh `04_label_merge.py`.

### Tahap 5 — Split & augmentasi (`05_split_augment.py`)
1. **Group split stratified** (proporsi kelas dijaga, `seed=42`): ulasan yang **sangat mirip** (cosine TF-IDF n-gram karakter
   ≥ `DEDUP_SIM` = 0,90, mis. "sangat mudah dan cepat" vs "sangat cepat dan mudah") digabung jadi satu kelompok, dan satu kelompok
   selalu masuk partisi yang sama (`StratifiedGroupKFold`). 20% test; sisanya train, lalu 10% train untuk validasi.
   Skrip mencetak cek kebocoran (jumlah ulasan test yang mirip dengan train) — hasilnya harus 0.
   *Mengapa:* split acak sebelumnya menempatkan 66 ulasan test (2,7%) yang hampir kembar dengan ulasan train/sintetis; semuanya
   diprediksi benar sehingga skor sedikit terangkat (±0,001). Hasil split acak itu diarsipkan di `results/archive_split_acak/`.
2. **Buat ketimpangan 1:3.** Jika kelas minoritas di train lebih banyak dari `TARGET_RATIO` (1/3) × kelas mayoritas,
   sebagian **kelas minoritas di train** dibuang acak (seed 42) sampai minoritas : mayoritas = 1 : 3. Data yang dibuang
   disimpan di `data/train_removed_minority.csv`. Tujuannya agar ketimpangan jelas dan efek augmentasi bisa diukur.
3. **Augmentasi LLM** untuk kelas minoritas sampai jumlahnya sama dengan mayoritas. LLM menulis ulang ulasan
   minoritas **yang tersisa** dengan sentimen tetap, dengan 4 gaya berbeda (ubah susunan kalimat, santai, ringkas, tambah detail).
   Karena butuh ±2× jumlah data asli, satu ulasan bisa punya 2–3 parafrase berbeda.
   Setiap parafrase disimpan bersama teks sumbernya (`augmented_cache.csv`: `source, text`), sehingga:
   * parafrase dari ulasan yang **sudah dibuang** di langkah 2 tidak dipakai (agar data yang dibuang tidak "masuk lagi");
   * parafrase melewati `preprocess()` yang sama, dibuang bila < 3 kata, kembar, sama dengan teks mana pun di train/val/test,
     atau **sangat mirip (≥ 0,90) dengan ulasan validasi/test** (agar data sintetis tidak "membocorkan" test).

Dua file train dihasilkan agar perbandingan **adil** (sama-sama dari train yang sudah dikurangi):
* `train_baseline.csv` — tanpa data sintetis
* `train_augmented.csv` — dengan data sintetis (kolom `source = llm_augmented`)

**Validation dan test tidak pernah disentuh** oleh pengurangan maupun augmentasi (mencegah *data leakage*).

**Alur augmentasi (Claude, tanpa API).** Jika parafrase di cache belum cukup, skrip mengekspor teks sumber ke `data/aug_in/part_*.txt`
(format `nomor<TAB>gaya<TAB>teks`, dilebihkan 15% untuk cadangan duplikat) lalu berhenti. Claude menulis satu parafrase
per baris ke `data/aug_out/part_*.txt`, `05b_claude_aug_import.py` memasukkannya ke cache, dan skrip dijalankan lagi.

Angka di proyek ini (group split): train 8.641 (5.543 negatif, 3.098 positif, rasio 0,56), validasi 969, test 2.396
(1.556 negatif, 840 positif). Positif di train dikurangi 1.251 → **baseline: 5.543 negatif, 1.847 positif (1:3)**.
Augmentasi menambah 3.696 parafrase positif (±2 per ulasan) dari `part_1`–`part_18`, hanya yang sumbernya termasuk ulasan
positif yang dipertahankan → **augmented: 5.543 : 5.543**.
Kecocokan sumber–parafrase dicek: rata-rata kesamaan kata baris sejajar 0,42–0,47 vs 0,03 bila digeser satu baris.

### Tahap 6 — Fine-tuning IndoBERT (`06_train.py`)
Model `indobenchmark/indobert-base-p1` + lapisan klasifikasi 2 kelas (`BertForSequenceClassification`).

| Pengaturan | Nilai |
|---|---|
| Learning rate | 2e-5 (dengan warmup 10% lalu turun linear) |
| Batch efektif | 16 (= 8 × akumulasi 2, lihat bagian 6) |
| Epoch | maksimal 3 |
| Optimizer | AdamW, weight decay 0.01 |
| Panjang maks token | 128 (padding dinamis per batch) |
| Loss | Cross-Entropy |
| Early stopping | berhenti jika val loss tidak turun 1 epoch |
| Seed | 42 |

Model terbaik (val loss terendah) disimpan di `models/<mode>/`, lalu dievaluasi di data test.

**Memantau progres.** Setiap 100 langkah dicetak baris seperti:
```
  [ 12.3%] epoch 1/3 step 400/1081 | loss 0.3120 | berjalan 2m 05d | sisa ~14m 50d
```
Persen dihitung dari total langkah 3 epoch; "sisa" = rata-rata waktu per langkah × langkah tersisa (berhenti lebih cepat bila early stopping).
Agar bisa dipantau saat berjalan di background, simpan ke log:
`python -u 06_train.py --mode baseline > results/train_baseline.log`, lalu buka file log itu.
Di GTX 1650 Ti (±0,9 detik per 8 data): baseline (7.390 data) ±20 menit per epoch, augmented (11.086 data) ±27 menit per epoch.
Run terakhir: baseline berhenti setelah 2 epoch (±40 menit), augmented setelah 3 epoch (±81 menit).

### Tahap 7 — Laporan eksperimen (`07_compare.py`)
Menulis `results/comparison.md` berisi: konfigurasi & seed, jumlah data per kelas di setiap partisi, precision/recall/F1/support
**per kelas**, confusion matrix (TP, TN, FP, FN), tabel perbandingan, dan **uji McNemar** (apakah selisih kedua model signifikan
pada test set yang sama). Test set hanya dipakai di sini — pemilihan model (early stopping) memakai data validasi.

### Tahap 8 — Evaluasi pada label test terverifikasi manusia (`08_evaluate_verified.py`)
Label test sebagian besar berasal dari leksikon + Claude, jadi label itu sendiri perlu diperiksa manusia.
1. Jalankan pertama kali → dibuat `data/test_verification.csv` (kolom `no, id, text, raw_text, label, source, verified`).
2. Manusia membaca `raw_text` dan label, lalu mengisi `verified`: `1` / `0` (label yang benar) atau `-` (buang: netral/tidak jelas).
   Baris yang belum dicek dibiarkan kosong. Di proyek ini pengecekan dilakukan di chat per blok 10 ulasan dengan label ditampilkan,
   dan peneliti hanya menulis koreksi.
3. Jalankan lagi → `results/comparison_verified.md` berisi:
   * **audit label** per sumber (termasuk label hasil kesepakatan leksikon + Claude) dengan selang kepercayaan 95% Wilson dan Cohen's kappa;
   * **kinerja kedua model** pada (a) seluruh test setelah koreksi (label `verified` menggantikan label dataset, baris `-` dibuang)
     dan (b) hanya ulasan yang sudah dicek, masing-masing dengan confusion matrix dan uji McNemar;
   * **jumlah salah prediksi per sumber label**, untuk analisis kesalahan.
Model tidak dilatih ulang, karena verifikasi hanya mengubah label yang dipakai untuk menilai.

### Tahap 9 — Beberapa seed (`06_train.py --seed`, `09_multiseed.py`)
`python 06_train.py --mode baseline --seed 1` melatih ulang dengan seed training lain. Split dan test tetap sama, dan hasilnya disimpan
di `models/seed_1/` serta `results/seed_1/`, sehingga hasil seed 42 tidak tertimpa. Daftar seed ada di `EXTRA_SEEDS` (`config.py`).
`09_multiseed.py` merangkum rata-rata ± simpangan baku dan selisih augmentasi per seed ke `results/multiseed.md`.
Status: seed 42, 1, dan 2 selesai untuk kedua skenario (hasil di bagian 6b). Run terakhir: seed 1 augmented ±50 menit,
seed 2 baseline ±60 menit, dan seed 2 augmented ±85 menit di GTX 1650 Ti.

### Tahap 10 — Diagnostik (`10_diagnostics.py`)
Menjawab pertanyaan "apakah skor setinggi ini wajar?". Isinya: cek kebocoran (ID, teks identik, hampir kembar terhadap train/val/sintetis),
pembanding sederhana (kelas mayoritas, InSet, TF-IDF + regresi logistik yang dilatih di train yang sama), skor per lapisan kesulitan
(sumber label, tingkat kemiripan dengan train) dengan selang kepercayaan bootstrap 95% dan McNemar terkoreksi Bonferroni,
serta rating vs label. Tidak ada yang disetel memakai test. Variasi antar seed ada di Tahap 9.

### Tahap 11 — Uji ketahanan split (`11_robust_splits.py`)
Menguji apakah skor bergantung pada cara data dibagi. Ada dua split tambahan; keduanya hanya melatih baseline dengan aturan 1:3 dan seed yang sama:
* `robust_dedup08`: group split dengan ambang kemiripan **0,8** (lebih ketat dari 0,9), sehingga ulasan mirip ≥ 0,8 tidak terpisah antara train dan test.
* `robust_waktu`: test = **20% ulasan terbaru** (14 September – 23 Oktober 2026), train dan validasi = ulasan yang lebih lama.
  Ini mendekati pemakaian nyata, ketika model menilai ulasan yang ditulis setelah ia dilatih.

Skrip pertama kali membuat `data/robust_<varian>/` (tidak pernah ditimpa). Training memakai env `DATA_DIR` dan `--tag`, sehingga model dan
hasil tersimpan di `models/robust_<varian>/` dan `results/robust_<varian>/`; split utama tidak berubah. Jalankan skrip lagi untuk
laporan `results/robustness.md`, yang juga melatih TF-IDF pada split yang sama sebagai ukuran tingkat kesulitan test.

### Tahap 12 — Test ulasan baru berlabel buta (`12_fresh_test.py`)
Bukti terakhir di luar dataset. `scrape` mengambil ulasan yang terbit **setelah** ulasan terakhir di `reviews_raw.csv`, memakai
preprocessing yang sama, membuang teks yang identik dengan dataset lama, lalu mengambil acak `FRESH_SIZE` (200) ulasan ke
`data/fresh/fresh_labels.csv` (tidak pernah ditimpa). Peneliti mengisi kolom `label` (`1` / `0` / `-` netral atau tidak jelas)
hanya dari `raw_text`. Rating, label leksikon/Claude, dan prediksi tidak ditampilkan, dan model baru dijalankan setelah pelabelan selesai.
Kolom opsional `koreksi` menampung label yang diperbaiki setelah prediksi terlihat; hasilnya dilaporkan terpisah sebagai angka tambahan.
`evaluate` menjalankan keenam model IndoBERT (3 seed × 2 skenario) tanpa training ulang, ditambah TF-IDF + regresi logistik
yang dilatih di train baseline, lalu menulis `results/fresh_test.md`.

---

## 5. Cara melakukan anotasi manual

1. Buka `data/manual_label.csv` di Excel / Google Sheets.
2. Kolom yang membantu: `raw_text` (teks asli), `rating`, `lex_label` (tebakan leksikon), `llm_label` (tebakan LLM).
3. Isi kolom **`manual_label`**:
   * `1` = positif (puas, memuji, merekomendasikan)
   * `0` = negatif (keluhan, kecewa, kritik)
   * **kosong** atau `-` = buang (netral, tidak relevan, atau tidak yakin)
4. Aturan: baca makna keseluruhan kalimat (sarkasme dinilai dari maknanya, bukan kata literal). Jika campuran,
   pilih polaritas yang dominan; jika tidak ada yang dominan, kosongkan.
5. Simpan sebagai **CSV** (bukan .xlsx), lalu jalankan `04_label_merge.py`.

Tips: file ini bisa berisi **ribuan** baris karena leksikon dan LLM sering berbeda (leksikon tidak memahami
negasi/sarkasme). Baris yang dikosongkan **dibuang**, jadi jika waktu terbatas kamu boleh melabeli sebagian saja,
tetapi tulis jumlahnya di laporan. Menjalankan ulang `03_label_auto.py` tidak menghapus isianmu.
Untuk reliabilitas (Lampiran C proposal), minta orang kedua melabeli ulang 10% baris, lalu hitung Cohen's kappa
(`sklearn.metrics.cohen_kappa_score`), targetnya ≥ 0,60.

---

## 6. Hal teknis penting

* **GPU**: laptop ini memakai **GTX 1650 Ti (VRAM 4 GB)**. Agar muat, training memakai
  *mixed precision* (fp16) dan *gradient accumulation*: 2 batch kecil (8) dijumlahkan menjadi 1 update (batch efektif 16,
  sesuai proposal). Saat diuji, pemakaian VRAM sekitar 3,7 GB. Tutup program lain yang memakai GPU (browser dengan akselerasi, game).
  Jika muncul *CUDA out of memory*, turunkan `MICRO_BATCH` ke 4 atau `MAX_LEN` ke 64 di `config.py`.
* **Mengapa `macro` average?** Dataset timpang, jadi accuracy saja menyesatkan. Laporan memuat accuracy, serta
  precision/recall/F1 **per kelas** dan **macro**. Fokus penelitian: recall & F1 **kelas minoritas**.
* **Confusion matrix** disimpan dalam format `[[TN, FP], [FN, TP]]` dengan label 0 = negatif, 1 = positif.
* **Reprodusibilitas**: `SEED = 42` untuk split, sampling, dan PyTorch. Hasil LLM tidak sepenuhnya deterministik, karena itu
  hasil LLM disimpan sebagai cache dan file data (`labeled.csv`, `train_augmented.csv`) sebaiknya diarsipkan.
* **Dua kali training, satu test set**: kedua skenario dites pada `data/test.csv` yang identik.
* **LLM**: pelabelan dan augmentasi dikerjakan **Claude** (lewat chat, tanpa API) dengan alur ekspor → Claude → impor.
* **Keamanan**: file `.env` berisi API key Gemini lama yang sudah tidak dipakai kode. Jangan dibagikan atau di-commit.
  Karena key pernah dikirim lewat chat, cabut key itu di Google AI Studio, lalu file `.env` boleh dihapus.
* **Kode lama** (sebelum disederhanakan, termasuk jalur Gemini) diarsipkan di `arsip_kode_lama_2026-10-28.zip`. Penyederhanaan
  tidak mengubah data, model, maupun hasil: setiap tahap 02–12 (kecuali training 06, yang tidak diubah) dijalankan ulang dengan kode
  lama dan baru, dan file keluarannya dibandingkan. Setelah itu laporan Tahap 10–12 dipangkas dari bagian yang dobel atau tidak dipakai
  (variasi seed dan panjang ulasan di Tahap 10, kolom train dan recall di Tahap 11, kelas mayoritas, InSet, dan prediksi ulasan `-`
  di Tahap 12). Semua angka yang tersisa sama persis dengan laporan sebelumnya, dan split uji ketahanan yang dibuat ulang identik.

## 6b. Hasil eksperimen (group split, seed 42)

Laporan lengkap ada di `results/comparison.md` (Tahap 7) dan `results/comparison_verified.md` (Tahap 8).

**Data.** Train baseline 7.390 (5.543 negatif : 1.847 positif = 1:3). Train augmented 11.086 (5.543 : 5.543, berisi 3.696 parafrase
Claude). Validasi 969. Test 2.396 (1.556 negatif, 840 positif); setelah verifikasi menjadi **2.395** karena 1 ulasan dibuang.
Kedua model dinilai pada test yang sama.

**Verifikasi label test oleh manusia (Tahap 8).** Sebanyak 1.002 dari 2.396 ulasan test sudah dicek, yaitu no 1–700 dan
**seluruh 302 ulasan bersumber manual** (no 2095–2396).

| Sumber label | Di test | Dicek | Diubah | Dibuang | Akurasi label (CI 95%) |
|---|---|---|---|---|---|
| lexicon+llm (leksikon & Claude sepakat) | 1.682 | 606 | 0 | 1 | 100% (99,4–100%) |
| llm (leksikon ambigu) | 412 | 94 | 0 | 0 | 100% (96,1–100%) |
| manual: aturan "ikut Claude" (leksikon ≠ Claude) | 271 | 271 | 0 | 0 | 100% (98,6–100%) |
| manual: gaya peneliti (dibantu Claude) | 19 | 19 | 0 | 0 | 100% (83,2–100%) |
| manual: dilabeli peneliti | 12 | 12 | 0 | 0 | 100% (75,8–100%) |
| **Semua** | 2.396 | 1.002 | 0 | 1 | **100% (99,6–100%)**, kappa 1,000 |

Label dataset terbukti sangat andal. Ini juga sejalan dengan audit buta 100 ulasan di Tahap 4.

**Kinerja model pada seluruh test setelah koreksi (2.395 ulasan).**

| Metrik | Baseline | + Augmentasi Claude | Selisih |
|---|---|---|---|
| Accuracy | 0,9820 | 0,9866 | +0,0046 |
| Macro F1 | 0,9804 | 0,9853 | +0,0049 |
| Positif (minoritas) precision | 0,9639 | 0,9844 | +0,0205 |
| Positif (minoritas) recall | 0,9857 | 0,9774 | −0,0083 |
| Positif (minoritas) F1 | 0,9747 | 0,9809 | +0,0062 |
| TP / TN / FP / FN | 828 / 1.524 / 31 / 12 | 821 / 1.542 / 13 / 19 | |
| Jumlah salah prediksi | 43 | 32 | −11 |

* **Epoch terbaik**: baseline di epoch 1 (val loss 0,0568), augmented di epoch 2 (val loss 0,0445). Keduanya dihentikan oleh early stopping.
* **Uji McNemar (exact)**: 14 ulasan benar di baseline tetapi salah di augmented, dan 25 ulasan sebaliknya. p = 0,108, jadi
  **perbedaannya tidak signifikan** (α = 0,05).
* **Polanya**: augmentasi mengurangi FP (ulasan negatif yang ditebak positif) dari 31 menjadi 13, tetapi menambah FN dari 12 menjadi 19.
  Akibatnya precision positif naik dan recall positif sedikit turun.
* **Pada 1.001 ulasan yang sudah dicek manusia saja**: accuracy 0,9740 vs 0,9760 dan macro F1 0,9694 vs 0,9714.
  McNemar 9 vs 11, p = 0,824, jadi **tidak signifikan**.

**Di mana model salah? (analisis kesalahan)**

| Sumber label | Ulasan | Salah baseline | Salah augmented |
|---|---|---|---|
| lexicon+llm | 1.681 | 16 | 3 |
| llm (leksikon ambigu) | 412 | 9 | 8 |
| manual (semua) | 302 | 18 | 21 |

Ulasan bersumber manual adalah ulasan sulit: leksikon dan Claude berbeda pendapat, dan kebanyakan bersentimen campuran
("bagus, tapi…"). Jumlahnya hanya 12,6% dari test, tetapi menyumbang 42% kesalahan baseline dan 66% kesalahan augmented.
Pada seed 42, augmentasi tampak membantu pada ulasan yang jelas (16 → 3 salah) tetapi tidak pada ulasan campuran (18 → 21 salah).
**Pola ini tidak berulang di seed lain** (lihat "Hasil beberapa seed"), jadi jangan dijadikan temuan. Yang konsisten di semua seed
hanyalah bahwa kesalahan menumpuk di ulasan campuran bersumber manual.

**Hasil beberapa seed (`09_multiseed.py` → `results/multiseed.md`).** Split dan test sama; hanya seed training yang berbeda.

| Seed | Macro F1 baseline | Macro F1 + augmentasi | Selisih | Recall positif (base → aug) | McNemar p |
|---|---|---|---|---|---|
| 42 | 0,9804 | 0,9853 | +0,0049 | 0,9857 → 0,9774 | 0,108 |
| 1 | 0,9803 | 0,9812 | +0,0009 | 0,9750 → 0,9726 | 0,845 |
| 2 | 0,9857 | 0,9849 | −0,0009 | 0,9750 → 0,9810 | 0,845 |
| **Rata-rata ± SD** | **0,9821 ± 0,0031** | **0,9838 ± 0,0023** | **+0,0016 ± 0,0030** | 0,9786 → 0,9770 | 0/3 signifikan |

* Selisih augmentasi rata-rata hanya +0,16 poin. Simpangan bakunya (0,30 poin) **lebih besar** daripada selisih itu sendiri.
* Augmentasi lebih baik di 2 dari 3 seed, tetapi tidak satu pun signifikan. Recall positif, yang menjadi fokus hipotesis, malah lebih rendah di 2 dari 3 seed.
* Variasi baseline akibat seed saja (0,9803–0,9857) sudah sebesar selisih terbesar akibat augmentasi (+0,0049 di seed 42).
  Hasil seed 42 yang tampak paling menguntungkan adalah salah satu ujung variasi, bukan efek yang stabil.

**Kesimpulan.** Augmentasi parafrase LLM **tidak memberi peningkatan yang konsisten maupun signifikan**: rata-rata 3 seed hanya
+0,16 poin macro F1, dan recall kelas minoritas tidak naik. Jadi **hipotesis 3 tidak didukung**.
Penyebab yang paling mungkin adalah efek langit-langit: baseline sudah ±98%, karena ulasan netral/tidak jelas dibuang saat pelabelan.
Selain itu, parafrase sintetis cenderung berupa kalimat positif yang "bersih", sedangkan kesalahan model ada di ulasan campuran.
Ini sejalan dengan catatan proposal bahwa efek augmentasi pada model pra-latih besar terbatas [12].

**Apakah skor ±98% wajar? (`10_diagnostics.py` → `results/diagnostics.md`)**

* **Tidak ada kebocoran langsung**: 0 ID, 0 teks identik, dan 0 ulasan hampir kembar (≥ 0,9) antara test dan train/val/data sintetis.
* **Tugasnya memang mudah**, karena ulasan netral/ambigu dibuang saat pelabelan. Model sederhana sudah tinggi:

  | Model | Macro F1 (CI 95%) |
  |---|---|
  | Kelas mayoritas | 0,394 |
  | InSet Lexicon | 0,696 |
  | TF-IDF + regresi logistik | 0,959 (0,951–0,967) |
  | IndoBERT baseline | 0,980 (0,975–0,986) |
  | IndoBERT + augmentasi | 0,985 (0,980–0,991) |

  Nilai tambah IndoBERT dibanding TF-IDF sekitar 2 poin, dan selang kepercayaannya tidak bertumpuk.
* **Skor di ulasan sulit lebih rendah.** Ulasan bersumber manual (leksikon ≠ Claude) mendapat 0,940 (baseline) dan 0,930 (augmentasi).
  Ulasan yang tidak mirip dengan train mana pun (kemiripan < 0,5, 1.358 ulasan) mendapat macro F1 0,955 dan 0,970.
  Ulasan berpola umum yang mirip dengan train (≥ 0,7, 370 ulasan) hampir selalu benar (99,7%).
* **Augmentasi per lapisan (eksploratif, seed 42 saja).** Di seed 42, lapisan lexicon+llm tampak signifikan (15 vs 2, p Bonferroni 0,016).
  **Hasil ini tidak berulang**: seed 1 memberi 6 vs 6, dan seed 2 memberi 1 vs 3 (augmentasi justru lebih buruk). Ini contoh mengapa
  analisis subkelompok dari satu run tidak boleh dijadikan klaim. Uji utama pada seluruh test tidak signifikan di ketiga seed.
* **Label tidak menyalin rating.** Aturan "bintang ≥ 4 = positif" hanya 93,4% benar, dan ada 88 ulasan bintang 5 yang berlabel negatif.

**Uji ketahanan split (`11_robust_splits.py` → `results/robustness.md`, baseline, seed 42)**

| Split | Test (neg/pos) | TF-IDF macro F1 | IndoBERT macro F1 | Akurasi IndoBERT | Akurasi sumber manual |
|---|---|---|---|---|---|
| Utama (group, kemiripan 0,9) | 1.556 / 840 | 0,959 | 0,980 | 0,982 | 0,940 |
| Group lebih ketat (kemiripan 0,8) | 1.549 / 716 | 0,955 | **0,978** | 0,981 | 0,920 |
| Menurut waktu (20% terbaru) | 1.822 / 579 | 0,939 | **0,968** | 0,976 | 0,914 |

* **Split lebih ketat** hanya menurunkan macro F1 sekitar 0,25 poin, dan TF-IDF turun dengan besar yang sama. Ulasan mirip 0,8–0,9 di test hampir tidak
  menyumbang skor, jadi **skor ±98% tidak berasal dari ulasan hampir kembar**.
* **Split waktu** menurunkan macro F1 sekitar 1,3 poin, menjadi 0,968. Ulasan terbaru lebih banyak yang negatif (24% positif, padahal di
  keseluruhan data 35%), sehingga test-nya memang lebih sulit: TF-IDF turun lebih besar, sekitar 2 poin.
  IndoBERT lebih tahan daripada TF-IDF, dan selisih keduanya melebar dari 2,1 menjadi 2,8 poin.
  **Angka ±96,8% inilah perkiraan yang paling jujur untuk kinerja pada ulasan baru.**
* Di ketiga split, ulasan bersumber manual (kalimat campuran) tetap yang tersulit (91–94%).
* Split waktu sengaja tidak mengelompokkan ulasan mirip (50 ulasan test mirip ≥ 0,9 dengan train), tetapi skornya justru paling rendah.
  Ini menguatkan bahwa kemiripan bukan sumber skor tinggi.
* Setiap varian baru dilatih dengan 1 seed. Penurunan split ketat (0,25 poin) sedikit di bawah rentang variasi seed split utama
  (0,9803–0,9857), sehingga perlu dibaca sebagai indikasi, bukan angka pasti.

**Test ulasan baru berlabel buta (`12_fresh_test.py` → `results/fresh_test.md`)**

Sampel acak 200 dari 202 ulasan yang terbit 23–26 Oktober 2026, setelah ulasan terakhir dataset. Peneliti melabelinya dari teks asli
tanpa melihat rating, label, atau prediksi. 6 ulasan (3%) ditandai netral/tidak jelas dan tidak dihitung, sehingga tersisa
194 ulasan (135 negatif, 59 positif).

| Model | Accuracy (CI 95%) | Macro F1 (CI 95%) |
|---|---|---|
| TF-IDF + regresi logistik | 0,943 (0,901–0,968) | 0,932 (0,886–0,969) |
| IndoBERT baseline, seed 42 / 1 / 2 | 0,959 / 0,964 / 0,954 | 0,951 / 0,956 / 0,944 |
| IndoBERT + augmentasi, seed 42 / 1 / 2 | 0,959 / 0,959 / 0,959 | 0,950 / 0,950 / 0,951 |

Selang kepercayaan 95% keenam model IndoBERT berkisar 0,91–0,98 untuk accuracy dan 0,91–0,99 untuk macro F1 (rincian di laporan).
Sebagai patokan, menebak "selalu negatif" sudah benar 69,6% (135 dari 194).

* Pada ulasan yang benar-benar baru dengan label manusia yang dibuat buta, IndoBERT mencapai **accuracy ±96% dan macro F1 ±95%**.
  Angka ini sedikit di bawah split waktu (0,968), tetapi masih di dalam selang kepercayaannya. Jadi perkiraan "±95–97% untuk ulasan
  baru" konsisten dari dua cara yang berbeda.
* Penurunan dari test utama (0,98) masuk akal karena labelnya berasal dari penilai lain. Label test utama dibuat oleh proses yang sama
  dengan label train (leksikon + Claude), sedangkan di sini oleh manusia. Sebagian besar kesalahan berupa kalimat campuran
  ("oke, tapi mahal") atau candaan.
* Dua ulasan berisi keluhan jelas tetapi berlabel positif (no 64 dan 90), kemungkinan salah ketik saat melabeli. Peneliti mengoreksinya
  menjadi negatif di kolom terpisah `koreksi`; kolom `label` (buta) tidak diubah. Setelah koreksi, IndoBERT mencapai accuracy
  0,964–0,974 dan macro F1 0,956–0,969, sedangkan TF-IDF 0,954 / 0,944. Angka ini hanya **angka tambahan**: koreksi dibuat setelah prediksi
  terlihat, dan yang dicek hanya ulasan yang salah diprediksi, sehingga cenderung menguntungkan model. **Angka utama tetap tabel di atas.**
* **Augmentasi tetap tidak membantu**: rata-rata accuracy baseline dan augmented sama persis (0,9588), dan McNemar p = 1,000 di ketiga seed.
* IndoBERT unggul atas TF-IDF ±1,2–2,4 poin macro F1. Namun dengan 194 ulasan selang kepercayaannya masih tumpang tindih, sehingga
  test sekecil ini belum cukup untuk membuktikan selisih tersebut.
* Hanya 3% ulasan baru yang netral/tidak jelas. Membuang ulasan ambigu saat pelabelan tidak menyembunyikan bagian besar dari ulasan nyata.

**Perbandingan dengan split acak (arsip `results/archive_split_acak/`).** Pada split acak sebelumnya, baseline 0,9904 dan
augmented 0,9871 (p = 0,169). Arah selisihnya berlawanan dengan hasil group split, tetapi keduanya tidak signifikan.
Artinya, efek augmentasi kecil dan tidak stabil antar pembagian data. Angka kedua split tidak bisa dibandingkan langsung karena isi test-nya berbeda.

**Keterbatasan.**
* Baru 3 seed training dengan satu split data. Simpangan baku dari 3 seed masih kasar; untuk kesimpulan yang lebih kuat, tambah seed di
  `EXTRA_SEEDS` (±2 jam per seed) atau ulangi dengan split lain.
* Analisis per lapisan di `10_diagnostics.py` memakai prediksi seed 42 saja, dan sifatnya eksploratif.
* Uji ketahanan split hanya untuk baseline, dengan 1 seed per varian. Label test pada varian baru belum diverifikasi manusia,
  kecuali ulasan yang kebetulan termasuk 1.002 ulasan terverifikasi.
* Verifikasi test baru sebagian (1.002 dari 2.396). Bagian yang dicek tidak acak (no 1–700 + seluruh sumber manual), dilakukan oleh
  satu orang, dan label ditampilkan saat dicek, sehingga ada kemungkinan bias konfirmasi. Risiko ini diimbangi oleh audit buta 100 ulasan (100% cocok).
* Test ulasan baru hanya 194 ulasan dari 4 hari dan dilabeli satu orang, sehingga selang kepercayaannya lebar (sekitar ±3 poin).

## 7. Perbedaan dengan proposal (catat di laporan)

| Proposal | Proyek ini | Alasan |
|---|---|---|
| Label: leksikon, manual hanya untuk ambigu | leksikon **vs LLM (Claude)**; leksikon ambigu → ikut LLM; bertentangan/netral → manual berbantuan LLM | Label lebih andal; beban manual 2.307 baris terlalu besar |
| InSet dipakai apa adanya | kata ambigu (ada di daftar positif & negatif) dan kata fungsi dibuang | InSet mentah salah menilai ~2.900 ulasan positif |
| Augmentasi *synonym replacement* | **parafrase LLM (Claude)** | Variasi lebih beragam |
| Augmentasi pada data yang sudah timpang | kelas **minoritas** di train dikurangi sampai 1:3 | Rasio asli 0,56 kurang timpang; agar efek augmentasi terukur |
| Validasi saat early stopping | 10% train dijadikan validasi | Proposal tidak menyebut sumber val |
| Split acak stratified | **group split** (ulasan sangat mirip tidak dipisah antar partisi) | Mencegah kebocoran ulasan hampir kembar |
| Test dinilai dengan label dataset | 1.002 label test **diverifikasi manusia** (Tahap 8), evaluasi akhir memakai label terkoreksi | Memastikan nilai model tidak tertipu label yang salah |
| Periode data Jan–Feb 2026 | ulasan terbaru saat scraping | Scraper tidak menyaring tanggal |
| Preprocessing: `[^a-zA-Z\s]` → kosong | → spasi | Mencegah kata menempel ("inisemoga") |
| Colab T4 | laptop GTX 1650 Ti | Sesuai perangkat |

## 8. Pemecahan masalah

* `UnicodeEncodeError` saat cetak: sudah ditangani di `config.py`.
* Scraping terhenti: skrip otomatis mengulang; jika jumlah < 20.000, ulasan Gojek berbahasa Indonesia memang habis.
* `04_label_merge.py` mengambil sedikit data: periksa kolom `manual_label` hanya berisi `0`, `1`, atau kosong.
