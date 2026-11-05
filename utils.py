"""Fungsi yang dipakai beberapa tahap:
  1. teks      : preprocessing dan skor InSet Lexicon
  2. kemiripan : kemiripan antarulasan, group split, pengurangan kelas minoritas
  3. statistik : McNemar, selang kepercayaan Wilson & bootstrap, macro F1, pembanding TF-IDF"""
import re
from functools import lru_cache
from math import comb, sqrt

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from config import RESOURCES, SEED, TARGET_RATIO

# ======================= 1. Teks =======================
# Kata negasi dipaksa jadi "tidak" supaya konsisten dengan proposal.
OVERRIDES = {w: "tidak" for w in ["gak", "ga", "gk", "nggak", "ngga", "enggak", "tak", "tdk"]}
OVERRIDES["gabisa"] = "tidak bisa"


@lru_cache(maxsize=1)
def load_slang():
    """Kamus slang -> baku dari IndoCollex (ambil bentuk baku paling sering muncul)."""
    df = pd.read_csv(RESOURCES / "indocollex.csv")
    df = df[(df["In-dictionary"] == 1) & (df["slang"] != df["formal"])]
    slang = df.groupby("slang")["formal"].agg(lambda s: s.mode().iloc[0]).to_dict()
    slang.update(OVERRIDES)
    return slang


# InSet memuat kata fungsi/netral dengan bobot besar (mis. "yang" = -5). Kata-kata ini dibuang
# agar skor leksikon tidak menyesatkan. Ditentukan dari kata paling sering di korpus ulasan.
NOISE_WORDS = set("""yang saya aplikasi saja lagi sama dari lebih ya buat itu kalau sekali apa harus pada semua lain
ambil dalam cari mencari sistem chat titik tau datang pun pihak begini makanan masuk cuma hanya harga pelayanan
promo diskon saldo layanan tetap langsung pas terima kadang sendiri seperti kayak sering meminta""".split())


@lru_cache(maxsize=1)
def load_inset():
    """InSet Lexicon: dict kata -> bobot (-5..+5).
    Dibuang: kata majemuk, kata yang ada di daftar positif DAN negatif (ambigu, mis. "bagus" +2 / -4),
    dan kata fungsi/netral di NOISE_WORDS."""
    pos = pd.read_csv(RESOURCES / "inset_positive.tsv", sep="\t")
    neg = pd.read_csv(RESOURCES / "inset_negative.tsv", sep="\t")
    ambiguous = set(pos["word"]) & set(neg["word"])
    both = pd.concat([pos, neg])
    both = both[~both["word"].str.contains(" ") & ~both["word"].isin(ambiguous | NOISE_WORDS)]
    return dict(zip(both["word"], both["weight"]))


def clean_text(text: str) -> str:
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)   # URL
    text = re.sub(r"[@#]\w+", " ", text)                  # mention & hashtag
    text = re.sub(r"\d+", " ", text)                      # angka
    text = re.sub(r"[^a-zA-Z\s]", " ", text)              # selain huruf (spasi agar kata tidak menempel)
    return re.sub(r"\s+", " ", text).strip()


def preprocess(text: str) -> str:
    """cleaning -> case folding -> normalisasi slang"""
    slang = load_slang()
    return " ".join(slang.get(tok, tok) for tok in clean_text(str(text)).lower().split())


def lexicon_score(text: str) -> int:
    """Jumlah bobot InSet dari semua kata di teks (persamaan 3.1 proposal)."""
    inset = load_inset()
    return sum(inset.get(tok, 0) for tok in text.split())


# ======================= 2. Kemiripan & split =======================
def max_similarity(vectorizer, texts, reference):
    """Untuk tiap teks: kemiripan tertinggi (cosine TF-IDF karakter) dengan teks mana pun di reference."""
    if len(texts) == 0 or len(reference) == 0:
        return np.zeros(len(texts))
    A, B = vectorizer.transform(texts), vectorizer.transform(reference)
    return np.concatenate([(A[i:i + 500] @ B.T).max(axis=1).toarray().ravel() for i in range(0, A.shape[0], 500)])


def similar_groups(vectorizer, texts, threshold):
    """Nomor kelompok tiap ulasan: ulasan dengan kemiripan >= threshold (langsung atau berantai) satu kelompok."""
    X = vectorizer.transform(texts)
    parent = list(range(len(texts)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for start in range(0, len(texts), 500):
        for i, j in np.argwhere((X[start:start + 500] @ X.T).toarray() >= threshold):
            parent[root(start + i)] = root(j)
    return [root(i) for i in range(len(texts))]


def group_split(data, n_folds):
    """Ambil 1/n_folds data sebagai partisi baru: stratified per label, satu kelompok tidak dipecah."""
    folds = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    rest, part = next(folds.split(data, data["label"], data["group"]))
    return data.iloc[rest], data.iloc[part]


def reduce_minority(train):
    """Jika kelas minoritas > TARGET_RATIO x mayoritas, sebagian minoritas dibuang acak sampai 1:3.
    Return (train yang sudah diacak urutannya, data minoritas yang dibuang)."""
    counts = train["label"].value_counts()
    major, minor = counts.idxmax(), counts.idxmin()
    removed = train.iloc[0:0]
    if counts[minor] / counts[major] > TARGET_RATIO:
        minority = train[train["label"] == minor]
        kept = minority.sample(int(counts[major] * TARGET_RATIO), random_state=SEED)
        removed = minority.drop(kept.index)
        train = pd.concat([train[train["label"] == major], kept])
    return train.sample(frac=1, random_state=SEED).reset_index(drop=True), removed


# ======================= 3. Statistik =======================
def mcnemar(y, pred_a, pred_b):
    """Uji McNemar exact pada test yang sama. Return (A benar & B salah, A salah & B benar, p-value)."""
    ok_a, ok_b = np.asarray(pred_a) == np.asarray(y), np.asarray(pred_b) == np.asarray(y)
    only_a, only_b = int((ok_a & ~ok_b).sum()), int((~ok_a & ok_b).sum())
    n, k = only_a + only_b, min(only_a, only_b)
    return only_a, only_b, (min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0)


def wilson(ok, n, z=1.96):
    """Proporsi ok/n + batas bawah & atas selang kepercayaan 95% Wilson (tetap masuk akal walau ok = n)."""
    p = ok / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, centre - half, min(1, centre + half)


def macro_f1(y, p):
    f1s = []
    for c in (0, 1):
        tp, fp, fn = np.sum((p == c) & (y == c)), np.sum((p == c) & (y != c)), np.sum((p != c) & (y == c))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return float(np.mean(f1s))


def bootstrap_ci(y, p, rng, n_boot=1000):
    """Selang kepercayaan 95% bootstrap (sampel ulang ulasan test). Return ((acc bawah, atas), (macro F1 bawah, atas))."""
    y, p = np.asarray(y), np.asarray(p)
    idx = rng.integers(0, len(y), (n_boot, len(y)))
    acc = np.percentile((y[idx] == p[idx]).mean(axis=1), [2.5, 97.5])
    f1 = np.percentile([macro_f1(y[i], p[i]) for i in idx], [2.5, 97.5])
    return acc, f1


def tfidf_lr(train, texts, **tfidf_args):
    """Pembanding sederhana: TF-IDF + regresi logistik, dilatih di train (kolom text & label), memprediksi texts."""
    vectorizer = TfidfVectorizer(min_df=2, **tfidf_args)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(vectorizer.fit_transform(train["text"]), train["label"])
    return clf.predict(vectorizer.transform(texts))
