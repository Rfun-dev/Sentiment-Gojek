"""TAHAP 11 - Uji ketahanan split: apakah skor tetap tinggi bila data dibagi lebih ketat?
Dua split tambahan, BASELINE saja (aturan 1:3 dan seed sama dengan tahap 5):
  robust_dedup08 : group split dengan kemiripan >= 0,8 (lebih ketat dari DEDUP_SIM 0,9)
  robust_waktu   : test = 20% ulasan TERBARU; train & validasi = ulasan yang lebih lama (validasi acak 10%)
Pemakaian:
  python 11_robust_splits.py                     -> buat data/robust_<varian>/ (sekali, tidak ditimpa)
  DATA_DIR=data/robust_dedup08 python 06_train.py --mode baseline --tag robust_dedup08
  DATA_DIR=data/robust_waktu   python 06_train.py --mode baseline --tag robust_waktu
  python 11_robust_splits.py                     -> laporan dengan hasil IndoBERT -> results/robustness.md
Split utama (data/train_baseline.csv, val.csv, test.csv) tidak diubah."""
import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from config import DATA, DEDUP_SIM, RESULTS, SEED, TEST_SIZE, VAL_SIZE
from utils import group_split, max_similarity, reduce_minority, similar_groups, tfidf_lr

df = pd.read_csv(DATA / "labeled.csv")[["id", "text", "label", "source"]]
ver = pd.read_csv(DATA / "test_verification.csv", encoding="utf-8-sig", dtype={"verified": str})
df = df[~df["id"].isin(ver.loc[ver["verified"].str.strip() == "-", "id"])].reset_index(drop=True)   # dibuang saat verifikasi
df["date"] = pd.to_datetime(df["id"].map(pd.read_csv(DATA / "reviews_raw.csv").set_index("id")["date"]))
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(df["text"])

# ---------- 1. Buat split varian (sekali saja; fungsi split & pengurangan minoritas sama dengan tahap 5) ----------
for name in ["robust_dedup08", "robust_waktu"]:
    out = DATA / name
    if out.exists():                                    # jangan menimpa split yang mungkin sudah dipakai training
        continue
    if name == "robust_dedup08":
        data = df.assign(group=similar_groups(vectorizer, df["text"].tolist(), 0.8))
        train, test = group_split(data, round(1 / TEST_SIZE))
        train, val = group_split(train, round(1 / VAL_SIZE))
    else:
        cut = df["date"].quantile(1 - TEST_SIZE)
        test, train = df[df["date"] >= cut], df[df["date"] < cut]
        train, val = train_test_split(train, test_size=VAL_SIZE, stratify=train["label"], random_state=SEED)
    out.mkdir()
    cols = ["id", "text", "label", "source"]
    reduce_minority(train)[0][cols].to_csv(out / "train_baseline.csv", index=False)
    val[cols].to_csv(out / "val.csv", index=False)
    test[cols].to_csv(out / "test.csv", index=False)
    print(f"{name}: dibuat di {out}")

# ---------- 2. Laporan ----------
VARIANTS = {"": "Split utama (group, kemiripan 0,9)",            # "" -> folder data/ dan results/ utama
            "robust_dedup08": "Group split lebih ketat (kemiripan 0,8)",
            "robust_waktu": "Split menurut waktu (test = 20% terbaru)"}
L = ["# Uji ketahanan split (baseline IndoBERT)", "",
     "Pertanyaan: apakah skor ±98% bertahan bila split dibuat lebih ketat? Ketiga varian memakai aturan 1:3 dan seed yang sama; "
     "isi test berbeda, jadi bandingkan pola, bukan angka persis. TF-IDF + regresi logistik dilatih di train varian yang sama "
     "sebagai ukuran tingkat kesulitan test.", "",
     "| Varian | Test (neg/pos) | Rentang tanggal test | Test mirip train >= 0,9 | >= 0,8 | TF-IDF macro F1 | "
     "IndoBERT accuracy | IndoBERT macro F1 | Akurasi sumber manual |", "|---|---|---|---|---|---|---|---|---|"]
for tag, name in VARIANTS.items():
    train, val, test = (pd.read_csv(DATA / tag / f) for f in ["train_baseline.csv", "val.csv", "test.csv"])
    seen = pd.concat([train, val])
    if tag == "":                                       # split utama: minoritas yang dibuang di tahap 5 juga bukan test
        seen = pd.concat([seen, pd.read_csv(DATA / "train_removed_minority.csv")])
    sims = max_similarity(vectorizer, test["text"], seen["text"])
    tf_f1 = f1_score(test["label"], tfidf_lr(train, test["text"], ngram_range=(1, 2)), average="macro")
    d = test["id"].map(dict(zip(df["id"], df["date"])))
    row = (f"| {name} | {(test['label'] == 0).sum()}/{(test['label'] == 1).sum()} | "
           f"{d.min():%d-%m-%Y} s.d. {d.max():%d-%m-%Y} | {(sims >= DEDUP_SIM).sum()} | {(sims >= 0.8).sum()} | {tf_f1:.4f} | ")
    res = RESULTS / tag
    if (res / "baseline_metrics.json").exists():
        m = json.load(open(res / "baseline_metrics.json"))
        p = pd.read_csv(res / "baseline_predictions.csv")
        manual = (test["source"] == "manual").values
        row += f"{m['accuracy']:.4f} | {m['macro']['f1']:.4f} | {(p['pred'].values[manual] == p['label'].values[manual]).mean():.4f} |"
    else:
        row += "belum dilatih | | |"
    L.append(row)

L += ["", "Catatan: split utama memakai label test terkoreksi untuk 1 ulasan yang dibuang; varian lain membuang ulasan itu dari "
      "dataset. Split waktu tidak memakai pengelompokan ulasan mirip, karena ulasan berpola sama memang muncul lagi di dunia nyata."]
text = "\n".join(L)
(RESULTS / "robustness.md").write_text(text, encoding="utf-8")
print(text)
