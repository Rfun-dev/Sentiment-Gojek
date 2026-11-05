"""TAHAP 2 - Preprocessing: cleaning, case folding, normalisasi slang, buang duplikat
Input : data/reviews_raw.csv      Output: data/reviews_clean.csv"""
import pandas as pd

from config import DATA, MIN_WORDS
from utils import preprocess

df = pd.read_csv(DATA / "reviews_raw.csv")
print("Data mentah:", len(df))

df["raw_text"] = df["content"].astype(str)
df["text"] = df["raw_text"].map(preprocess)

df = df[df["text"].str.split().str.len() >= MIN_WORDS]       # buang ulasan terlalu pendek
print(f"Setelah buang ulasan < {MIN_WORDS} kata:", len(df))
df = df.drop_duplicates(subset="text")                         # buang duplikat
print("Setelah buang duplikat:", len(df))

df[["id", "raw_text", "text", "rating"]].to_csv(DATA / "reviews_clean.csv", index=False)
print("Contoh hasil:")
print(df[["raw_text", "text"]].head(5).to_string())
