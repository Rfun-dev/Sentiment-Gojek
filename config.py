"""Semua pengaturan penelitian ada di sini. Ubah di satu tempat saja."""
import os
import sys
from pathlib import Path

# Konsol Windows sering gagal mencetak emoji -> ganti karakter bermasalah, jangan crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

BASE = Path(__file__).parent
DATA = Path(os.environ.get("DATA_DIR", BASE / "data"))   # bisa diganti lewat env DATA_DIR
RESOURCES = BASE / "resources"
MODELS = BASE / "models"
RESULTS = BASE / "results"

SEED = 42

# ---------- 1. Scraping ----------
APP_ID = "com.gojek.app"
SCRAPE_COUNT = 20000

# ---------- 2. Preprocessing ----------
MIN_WORDS = 3            # ulasan < 3 kata (setelah preprocessing) dibuang

# ---------- 3. Pelabelan ----------
TAU = 1                  # |skor InSet| < TAU dianggap ambigu

# ---------- 4. Split & augmentasi ----------
TEST_SIZE = 0.2          # 80% train, 20% test
VAL_SIZE = 0.1           # 10% dari train dipakai validasi (early stopping)
# Ulasan yang "sangat mirip" (cosine TF-IDF karakter >= DEDUP_SIM, mis. "sangat cepat dan mudah" vs
# "sangat mudah dan cepat") dikelompokkan dan satu kelompok selalu masuk partisi yang sama (group split),
# supaya tidak ada ulasan kembar yang tersebar di train dan test. Data sintetis yang terlalu mirip
# dengan ulasan validasi/test juga dibuang.
DEDUP_SIM = 0.9
# Jika kelas minoritas di TRAIN lebih banyak dari TARGET_RATIO x kelas mayoritas (data "tidak terlalu
# timpang"), sebagian kelas minoritas dibuang acak sampai minoritas : mayoritas = TARGET_RATIO.
# Tujuannya: ada ketimpangan yang jelas, sehingga efek augmentasi bisa diukur.
TARGET_RATIO = 1 / 3     # 1 : 3

# ---------- 5. Training ----------
MODEL_NAME = "indobenchmark/indobert-base-p1"
MAX_LEN = 128
BATCH_SIZE = 16          # batch efektif
MICRO_BATCH = 8          # yang benar-benar masuk GPU sekaligus (hemat VRAM 4 GB)
LR = 2e-5
EPOCHS = 3
WEIGHT_DECAY = 0.01
PATIENCE = 1             # early stopping
# Ulangan training dengan seed lain (split data tetap dari SEED). Dipilih sebelum melihat hasilnya.
EXTRA_SEEDS = [1, 2]

# ---------- 6. Test ulasan baru (tahap 12) ----------
FRESH_SIZE = 200         # ulasan terbaru yang dilabeli manusia tanpa melihat label/prediksi apa pun
