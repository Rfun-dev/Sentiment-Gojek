"""TAHAP 3b (opsional, tanpa API) - Impor label dari Claude -> data/llm_labels.csv (cache yang dipakai tahap 3).
Input: data/claude_out/chunk_001.txt ... (baris: nomor<TAB>p|n|x)  p=positif n=negatif x=netral"""
import pandas as pd

from config import DATA

CHUNK = 300
df = pd.read_csv(DATA / "reviews_clean.csv")
names = {"p": "positif", "n": "negatif", "x": "netral"}
rows, missing = [], 0
for n, start in enumerate(range(0, len(df), CHUNK), 1):
    part = df.iloc[start:start + CHUNK]
    f = DATA / "claude_out" / f"chunk_{n:03d}.txt"
    got = {}
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            bits = line.strip().split("\t") if "\t" in line else line.strip().split()
            if len(bits) == 2 and bits[0].isdigit() and bits[1].lower() in names:
                got[int(bits[0])] = names[bits[1].lower()]
    for i, rid in enumerate(part["id"]):
        if i in got:
            rows.append((rid, got[i]))
        else:
            missing += 1
out = pd.DataFrame(rows, columns=["id", "llm_label"])
out.to_csv(DATA / "llm_labels.csv", index=False)
print(f"Label Claude terimpor: {len(out)} | belum ada: {missing}")
print(out["llm_label"].value_counts().to_string())
