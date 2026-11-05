"""TAHAP 3a (opsional, tanpa API) - Ekspor ulasan ke potongan teks agar bisa dilabeli oleh Claude di chat.
Output: data/claude_in/chunk_001.txt ...   (satu baris per ulasan: nomor<TAB>teks asli)"""
import pandas as pd

from config import DATA

CHUNK = 300
df = pd.read_csv(DATA / "reviews_clean.csv")
out = DATA / "claude_in"
out.mkdir(exist_ok=True)
for n, start in enumerate(range(0, len(df), CHUNK), 1):
    part = df.iloc[start:start + CHUNK]
    lines = [f"{i}\t{' '.join(str(t).split())[:300]}" for i, t in enumerate(part["raw_text"])]
    (out / f"chunk_{n:03d}.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"{len(df)} ulasan -> {n} potongan di data/claude_in/")
