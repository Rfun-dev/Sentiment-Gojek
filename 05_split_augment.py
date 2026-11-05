"""TAHAP 5 - Split data, kurangi kelas minoritas di train, lalu augmentasi parafrase Claude.

Urutan:
  1. Group split stratified: ulasan yang sangat mirip dikelompokkan (DEDUP_SIM) dan satu kelompok
     selalu masuk partisi yang sama. Test 20%, sisanya train; 10% train jadi validation.
  2. Jika kelas minoritas di train lebih dari TARGET_RATIO (1:3) x kelas mayoritas, sebagian kelas
     minoritas di TRAIN dibuang acak sampai rasionya 1:3 (agar ketimpangan jelas & efek augmentasi terukur).
  3. Skenario 1 (baseline)  = train hasil langkah 2, tanpa augmentasi.
     Skenario 2 (augmented) = train langkah 2 + parafrase Claude dari kelas minoritas yang TERSISA,
     sampai jumlahnya = kelas mayoritas.
  Test & validation TIDAK PERNAH diubah/diaugmentasi (mencegah data leakage).

Augmentasi tanpa API: jika parafrase di data/augmented_cache.csv belum cukup, teks sumber diekspor ke
data/aug_in/, Claude menulis parafrase ke data/aug_out/, 05b_claude_aug_import.py mengisi cache, lalu
skrip ini dijalankan lagi.
Input : data/labeled.csv (+ data/augmented_cache.csv)
Output: data/train_baseline.csv, data/train_augmented.csv, data/val.csv, data/test.csv,
        data/train_removed_minority.csv, data/split_info.json"""
import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from config import DATA, DEDUP_SIM, SEED, TEST_SIZE, VAL_SIZE
from utils import group_split, max_similarity, reduce_minority, similar_groups

df = pd.read_csv(DATA / "labeled.csv")[["id", "text", "label", "source"]].reset_index(drop=True)
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(df["text"])

# ---------- 1. Group split ----------
df["group"] = similar_groups(vectorizer, df["text"].tolist(), DEDUP_SIM)
n_groups = df["group"].nunique()
print(f"{len(df)} ulasan -> {n_groups} kelompok ({len(df) - n_groups} ulasan punya kembaran mirip >= {DEDUP_SIM})")
train, test = group_split(df, round(1 / TEST_SIZE))
train, val = group_split(train, round(1 / VAL_SIZE))
print(f"Split -> train {len(train)}, val {len(val)}, test {len(test)}")
leak = (max_similarity(vectorizer, test["text"], train["text"]) >= DEDUP_SIM).sum()
print(f"Cek kebocoran: ulasan test yang mirip (>= {DEDUP_SIM}) dengan train = {leak}")
train, val, test = (d.drop(columns="group") for d in (train, val, test))

# ---------- 2. Kurangi kelas minoritas agar timpang 1:3 ----------
print(f"Train sebelum dikurangi: {train['label'].value_counts().to_dict()}")
train, removed = reduce_minority(train)
counts = train["label"].value_counts()
major, minor = counts.idxmax(), counts.idxmin()
print(f"Kelas minoritas dikurangi {len(removed)} data -> {counts.to_dict()}, rasio {counts[minor] / counts[major]:.2f}")

# Simpan data skenario baseline sekarang (training baseline bisa jalan sambil menunggu augmentasi)
train.to_csv(DATA / "train_baseline.csv", index=False)
val.to_csv(DATA / "val.csv", index=False)
test.to_csv(DATA / "test.csv", index=False)
removed.to_csv(DATA / "train_removed_minority.csv", index=False)

# ---------- 3. Augmentasi parafrase Claude untuk kelas minoritas ----------
n_needed = counts[major] - counts[minor]
minor_texts = train.loc[train["label"] == minor, "text"].tolist()

# Parafrase hanya boleh berasal dari minoritas yang TERSISA (bukan dari data yang sudah dibuang), minimal 3 kata,
# tidak boleh sama dengan teks mana pun di train/val/test, dan tidak boleh sangat mirip ulasan val/test.
cache_file = DATA / "augmented_cache.csv"
cache = pd.read_csv(cache_file) if cache_file.exists() else pd.DataFrame(columns=["source", "text"])
allowed = set(minor_texts)
rows = [(s, t) for s, t in zip(cache["source"], cache["text"]) if s in allowed and isinstance(t, str) and len(t.split()) >= 3]
sims = max_similarity(vectorizer, [t for _, t in rows], pd.concat([val["text"], test["text"]]))
seen = set(train["text"]) | set(val["text"]) | set(test["text"]) | set(removed["text"])
pairs = []                                             # (sumber, parafrase)
for (src, t), sim in zip(rows, sims):
    if t not in seen and sim < DEDUP_SIM:
        seen.add(t)
        pairs.append((src, t))
print(f"Cache augmentasi: {len(pairs)} parafrase valid (dari minoritas yang tersisa), dibutuhkan {n_needed}")

if len(pairs) < n_needed:
    # Ekspor teks sumber untuk ditulis ulang Claude. Aturan untuk Claude: satu versi baru per baris, sentimen tetap,
    # makna inti sama, bahasa Indonesia, panjang mirip, huruf kecil tanpa emoji, ikuti gaya di kolom 2, jangan menyalin.
    STYLES = ["ubah susunan kalimat dan pilihan kata", "pakai gaya santai/informal pengguna biasa",
              "ringkas tapi tetap jelas", "tambahkan sedikit detail situasi yang masuk akal"]
    in_dir, out_dir = DATA / "aug_in", DATA / "aug_out"
    in_dir.mkdir(exist_ok=True)
    waiting = [f.name for f in sorted(in_dir.glob("part_*.txt")) if not (out_dir / f.name).exists()]
    if waiting:
        raise SystemExit(f"Masih menunggu hasil Claude untuk: {', '.join(waiting)}. Isi data/aug_out/ dulu, lalu jalankan 05b.")
    short = n_needed - len(pairs)
    have = pd.Series([s for s, _ in pairs]).value_counts().to_dict()
    sources = []                                       # dahulukan ulasan yang parafrasenya paling sedikit
    for r in range(10):
        sources += [t for t in minor_texts if have.get(t, 0) <= r]
    sources = sources[:int(short * 1.15) + 1]          # lebihkan 15% untuk cadangan duplikat
    first = len(list(in_dir.glob("part_*.txt"))) + 1
    for n, start in enumerate(range(0, len(sources), 500), first):
        lines = [f"{i}\t{STYLES[(start + i + n) % len(STYLES)]}\t{t}" for i, t in enumerate(sources[start:start + 500])]
        (in_dir / f"part_{n}.txt").write_text("\n".join(lines), encoding="utf-8")
    raise SystemExit(f"Perlu {short} parafrase lagi. {len(sources)} teks sumber diekspor ke data/aug_in/part_{first}..part_{n}.txt. "
                     "Setelah Claude menulis hasil ke data/aug_out/, jalankan 05b_claude_aug_import.py lalu skrip ini lagi.")

pairs = pairs[:n_needed]
aug = pd.DataFrame({"text": [t for _, t in pairs], "label": minor, "source": "llm_augmented"})
print(f"Data sintetis dipakai: {len(aug)} (dari {len({s for s, _ in pairs})} ulasan sumber)")
train_aug = pd.concat([train, aug]).sample(frac=1, random_state=SEED).reset_index(drop=True)

# ---------- Simpan ----------
train_aug.to_csv(DATA / "train_augmented.csv", index=False)
info = {"minority_label": int(minor), "majority_label": int(major),
        "minority_removed_from_train": len(removed),
        "train_baseline": train["label"].value_counts().to_dict(),
        "train_augmented": train_aug["label"].value_counts().to_dict(),
        "val": val["label"].value_counts().to_dict(), "test": test["label"].value_counts().to_dict()}
json.dump(info, open(DATA / "split_info.json", "w"), indent=2, default=int)
print(json.dumps(info, indent=2, default=int))
