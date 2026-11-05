"""TAHAP 10 - Diagnostik: apakah skor tinggi model masuk akal dan bukan karena kebocoran?
Input : split di data/ (train, minoritas yang dibuang, val, test, data sintetis), test_verification.csv,
        reviews_clean.csv (rating), results/<mode>_predictions.csv (seed 42)
Output: results/diagnostics.md
Isi: (1) cek kebocoran, (2) pembanding sederhana, (3) skor per lapisan kesulitan + McNemar, (4) rating vs label.
Pembanding hanya dilatih di train; tidak ada yang disetel memakai test."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from config import DATA, DEDUP_SIM, RESULTS, SEED, TAU
from utils import bootstrap_ci, lexicon_score, macro_f1, max_similarity, mcnemar, tfidf_lr

rng = np.random.default_rng(SEED)                      # satu rng untuk semua bootstrap, dipakai berurutan


def with_ci(y, p):
    """Teks 'accuracy (CI 95%) | macro F1 (CI 95%)'."""
    (a_lo, a_hi), (f_lo, f_hi) = bootstrap_ci(y, p, rng)
    return f"{(y == p).mean():.4f} ({a_lo:.4f}-{a_hi:.4f}) | {macro_f1(y, p):.4f} ({f_lo:.4f}-{f_hi:.4f})"


# ---------- Data: test dengan label terkoreksi verifikasi (sama seperti Tahap 8) ----------
train, removed, val, test = (pd.read_csv(DATA / f"{f}.csv") for f in ["train_baseline", "train_removed_minority", "val", "test"])
syn = pd.read_csv(DATA / "train_augmented.csv").query("source == 'llm_augmented'")
for mode in ["baseline", "augmented"]:
    p = pd.read_csv(RESULTS / f"{mode}_predictions.csv")
    assert (p["text"].values == test["text"].values).all(), "urutan prediksi tidak sama dengan test.csv"
    test[mode] = p["pred"].values
ver = pd.read_csv(DATA / "test_verification.csv", encoding="utf-8-sig", dtype={"verified": str})
test["v"] = test["id"].map(dict(zip(ver["id"], ver["verified"].fillna("").str.strip()))).fillna("")
test = test[test["v"] != "-"].reset_index(drop=True)   # dibuang saat verifikasi
fixed = test["v"].isin(["0", "1"])
test.loc[fixed, "label"] = test.loc[fixed, "v"].astype(int)
y = test["label"].values

L = ["# Diagnostik model: kebocoran, tingkat kesulitan, dan pembanding", "",
     f"Test: {len(test)} ulasan (label terkoreksi verifikasi; {(y == 1).sum()} positif, {(y == 0).sum()} negatif). "
     "Selang kepercayaan 95% = bootstrap 1.000 kali.", ""]

# ---------- 1. Kebocoran ----------
other = pd.concat([train, removed, val])
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(pd.concat([other["text"], test["text"], syn["text"]]))
test["sim"] = max_similarity(vec, test["text"], other["text"])
sim_syn = max_similarity(vec, test["text"], syn["text"])
L += ["## 1. Cek kebocoran", "", "| Pemeriksaan | Hasil |", "|---|---|",
      f"| ID test yang muncul di train/val | {test['id'].isin(other['id']).sum()} |",
      f"| Teks test identik dengan train/val | {test['text'].isin(set(other['text'])).sum()} |",
      f"| Teks test identik dengan data sintetis | {test['text'].isin(set(syn['text'])).sum()} |",
      f"| Ulasan test hampir kembar (>= {DEDUP_SIM}) dengan train/val | {(test['sim'] >= DEDUP_SIM).sum()} |",
      f"| Ulasan test hampir kembar (>= {DEDUP_SIM}) dengan data sintetis | {(sim_syn >= DEDUP_SIM).sum()} |", ""]

# ---------- 2. Pembanding sederhana (dilatih di train baseline yang sama) ----------
majority = train["label"].mode()[0]
lex = test["text"].map(lexicon_score)
models = {"Kelas mayoritas (selalu negatif)": np.full(len(y), majority),
          "InSet Lexicon (ambigu -> mayoritas)": np.where(lex >= TAU, 1, np.where(lex <= -TAU, 0, majority)),
          "TF-IDF kata 1-2 gram + regresi logistik": tfidf_lr(train, test["text"], ngram_range=(1, 2)),
          "TF-IDF karakter 3-5 gram + regresi logistik": tfidf_lr(train, test["text"], analyzer="char_wb", ngram_range=(3, 5)),
          "IndoBERT baseline": test["baseline"].values,
          "IndoBERT + augmentasi": test["augmented"].values}
L += ["## 2. Pembanding sederhana", "", "| Model | Accuracy (CI 95%) | Macro F1 (CI 95%) |", "|---|---|---|"]
L += [f"| {name} | {with_ci(y, p)} |" for name, p in models.items()]
L += ["", "Jika model sederhana sudah tinggi, tugasnya memang mudah; selisih IndoBERT dengan TF-IDF adalah nilai tambah "
      "sebenarnya dari model bahasa.", ""]

# ---------- 3. Skor per lapisan kesulitan ----------
source, sim = test["source"].values, test["sim"].values
layers = {"Seluruh test": np.ones(len(test), bool),
          "Sumber lexicon+llm (leksikon & Claude sepakat)": source == "lexicon+llm",
          "Sumber llm (leksikon ambigu)": source == "llm (leksikon ambigu)",
          "Sumber manual (leksikon != Claude, sulit)": source == "manual",
          "Kemiripan dengan train >= 0,7": sim >= 0.7,
          "Kemiripan dengan train 0,5-0,7": (sim >= 0.5) & (sim < 0.7),
          "Kemiripan dengan train < 0,5 (benar-benar baru)": sim < 0.5,
          "Test sulit = manual ATAU kemiripan < 0,5": (source == "manual") | (sim < 0.5)}
base, aug = test["baseline"].values, test["augmented"].values
L += ["## 3. Skor per lapisan kesulitan", "",
      "| Lapisan | Ulasan | Model | Accuracy (CI 95%) | Macro F1 (CI 95%) |", "|---|---|---|---|---|"]
for name, m in layers.items():
    L.append(f"| {name} | {m.sum()} | baseline | {with_ci(y[m], base[m])} |")
    L.append(f"| {name} | {m.sum()} | augmented | {with_ci(y[m], aug[m])} |")

n_sub = len(layers) - 1                                # 'Seluruh test' = uji utama, tidak dikoreksi
L += ["", "Uji McNemar baseline vs augmentasi per lapisan:", "",
      "| Lapisan | Baseline benar, aug salah | Baseline salah, aug benar | p-value | p Bonferroni |", "|---|---|---|---|---|"]
for i, (name, m) in enumerate(layers.items()):
    b, a, p = mcnemar(y[m], base[m], aug[m])
    L.append(f"| {name} | {b} | {a} | {p:.4f} | {'- (uji utama)' if i == 0 else f'{min(1.0, p * n_sub):.4f}'} |")
L += ["", f"Uji utama adalah 'Seluruh test'. {n_sub} lapisan lain diuji sekaligus, sehingga peluang 'signifikan' karena "
      "kebetulan ikut naik: pakai kolom p Bonferroni (p x jumlah lapisan). Lapisan dibuat setelah melihat hasil, "
      "jadi sifatnya eksploratif, bukan bukti utama.", ""]

# ---------- 4. Rating bintang vs label ----------
rating = test["id"].map(pd.read_csv(DATA / "reviews_clean.csv").set_index("id")["rating"])
tab = pd.crosstab(rating, test["label"])
L += ["## 4. Rating bintang vs label (deskriptif; rating TIDAK dipakai model)", "",
      "| Bintang | Negatif | Positif |", "|---|---|---|"]
L += [f"| {r} | {tab.loc[r].get(0, 0)} | {tab.loc[r].get(1, 0)} |" for r in tab.index]
not3 = rating != 3
L += ["", f"Aturan sederhana 'bintang >= 4 = positif' (tanpa bintang 3): akurasi "
      f"{((rating[not3] >= 4).astype(int) == test['label'][not3]).mean():.4f}."]

text = "\n".join(L)
(RESULTS / "diagnostics.md").write_text(text, encoding="utf-8")
print(text)
