"""TAHAP 5b (tanpa API) - Impor hasil parafrase Claude -> data/augmented_cache.csv (kolom: source, text)
Input : data/aug_in/part_*.txt   (baris: nomor<TAB>gaya<TAB>teks sumber)
        data/aug_out/part_*.txt  (baris: nomor<TAB>teks hasil tulis ulang)
Pasangan sumber-parafrase dicocokkan lewat nama file + nomor baris.
Setelah ini jalankan lagi 05_split_augment.py (ia membaca cache dan membuang yang tidak valid)."""
import pandas as pd

from config import DATA
from utils import preprocess

pairs = []
for f_in in sorted((DATA / "aug_in").glob("part_*.txt"), key=lambda f: int(f.stem.split("_")[1])):
    f_out = DATA / "aug_out" / f_in.name
    if not f_out.exists():
        print(f"{f_in.name}: belum ada hasil")
        continue
    src = {}
    for line in f_in.read_text(encoding="utf-8").splitlines():
        bits = line.split("\t")
        src[int(bits[0])] = bits[2]
    n_file = 0
    for line in f_out.read_text(encoding="utf-8").splitlines():
        bits = line.split("\t", 1)
        if len(bits) == 2 and bits[0].strip().isdigit() and int(bits[0]) in src:
            pairs.append((src[int(bits[0])], preprocess(bits[1])))     # samakan format dengan data lain
            n_file += 1
    print(f"{f_in.name}: {n_file} parafrase dari {len(src)} sumber")

cache = pd.DataFrame(pairs, columns=["source", "text"])
cache = cache[cache["text"].str.split().str.len() >= 3].drop_duplicates("text")
cache.to_csv(DATA / "augmented_cache.csv", index=False)
print(f"Total parafrase unik di cache: {len(cache)}")
