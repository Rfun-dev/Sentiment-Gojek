"""TAHAP 3 - Pelabelan otomatis: bandingkan InSet Lexicon vs label Claude.
  - Hasil SAMA (sama-sama positif/negatif)  -> label dipakai langsung
  - Leksikon ambigu, Claude positif/negatif -> ikut label Claude (aturan dilonggarkan)
  - Hasil BERBEDA / Claude netral           -> masuk data/manual_label.csv untuk diisi manual
Input : data/reviews_clean.csv, data/llm_labels.csv (label Claude dari 03a_claude_export.py -> Claude -> 03b_claude_import.py)
Output: data/labeled_auto.csv, data/manual_label.csv"""
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from config import DATA, TAU
from utils import lexicon_score

df = pd.read_csv(DATA / "reviews_clean.csv")

# ---------- A. Label leksikon ----------
df["lex_score"] = df["text"].map(lexicon_score)
df["lex_label"] = df["lex_score"].map(lambda s: "positif" if s >= TAU else "negatif" if s <= -TAU else "ambigu")

# ---------- B. Label Claude ----------
df = df.merge(pd.read_csv(DATA / "llm_labels.csv").drop_duplicates("id"), on="id", how="left")
df["llm_label"] = df["llm_label"].fillna("netral")        # belum/gagal dilabeli -> diperiksa manual

# ---------- C. Bandingkan ----------
llm_polar = df["llm_label"].isin(["positif", "negatif"])
both_same = (df["lex_label"] == df["llm_label"]) & llm_polar
lex_ambiguous = (df["lex_label"] == "ambigu") & llm_polar     # aturan dilonggarkan: leksikon ambigu -> ikut Claude
agree = both_same | lex_ambiguous
both = df[df["lex_label"].isin(["positif", "negatif"]) & llm_polar]
print(f"Cohen's kappa leksikon vs Claude (yang sama-sama berpolaritas): {cohen_kappa_score(both.lex_label, both.llm_label):.3f}")
print(f"Sepakat: {agree.sum()} ({agree.mean():.1%}) | Perlu manual: {(~agree).sum()} ({(~agree).mean():.1%})")

auto = df[agree].copy()
auto["label"] = (auto["llm_label"] == "positif").astype(int)
auto["source"] = auto["lex_label"].map(lambda s: "llm (leksikon ambigu)" if s == "ambigu" else "lexicon+llm")
auto[["id", "text", "label", "source"]].to_csv(DATA / "labeled_auto.csv", index=False)

# ---------- D. File untuk anotasi manual (isian lama manual_label & label_source tidak ditimpa) ----------
manual = df[~agree][["id", "raw_text", "text", "rating", "lex_score", "lex_label", "llm_label"]].copy()
manual["manual_label"] = ""
manual_file = DATA / "manual_label.csv"
if manual_file.exists():
    old = pd.read_csv(manual_file, encoding="utf-8-sig", dtype={"manual_label": str})
    filled = [c for c in ["manual_label", "label_source"] if c in old]
    manual = manual.drop(columns="manual_label").merge(old[["id"] + filled], on="id", how="left")
    print("manual_label.csv sudah ada -> isian lama dipertahankan")
manual.to_csv(manual_file, index=False, encoding="utf-8-sig")
print("\nBuka data/manual_label.csv, isi kolom manual_label: 1 = positif, 0 = negatif, kosong = buang.")
