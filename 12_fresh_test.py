"""TAHAP 12 - Test ulasan baru berlabel buta: bukti terakhir bahwa skor model berlaku di luar dataset.
Pemakaian:
  python 12_fresh_test.py scrape     -> ulasan yang terbit SETELAH ulasan terakhir di reviews_raw.csv
                                        -> data/fresh/reviews_fresh.csv + lembar label data/fresh/fresh_labels.csv
  (peneliti mengisi kolom `label`: 1 positif, 0 negatif, - netral/campur/tidak jelas; hanya dari raw_text,
   tanpa melihat rating, label leksikon/Claude, atau prediksi model)
  python 12_fresh_test.py evaluate   -> keenam model IndoBERT + TF-IDF -> results/fresh_test.md
Model tidak dilatih ulang. Kolom opsional `koreksi` = label yang diperbaiki SETELAH prediksi terlihat;
hanya dilaporkan sebagai angka tambahan, angka utama tetap memakai `label` (buta)."""
import argparse
import json

import numpy as np
import pandas as pd

from config import APP_ID, DATA, EXTRA_SEEDS, FRESH_SIZE, MAX_LEN, MIN_WORDS, MODELS, RESULTS, SEED
from utils import bootstrap_ci, macro_f1, mcnemar, preprocess, tfidf_lr, wilson

FRESH = DATA / "fresh"
LABELS = FRESH / "fresh_labels.csv"


def scrape():
    from google_play_scraper import Sort, reviews

    if LABELS.exists():                                 # lembar label mungkin sudah diisi -> jangan ditimpa
        print(f"{LABELS} sudah ada, tidak diambil ulang.")
        return
    old = pd.read_csv(DATA / "reviews_raw.csv")
    cutoff = pd.to_datetime(old["date"]).max()
    rows, token = [], None
    while True:                                         # urut terbaru; berhenti saat sampai ulasan lama
        batch, token = reviews(APP_ID, lang="id", country="id", sort=Sort.NEWEST, count=200, continuation_token=token)
        rows += batch
        if not batch or token is None or pd.Timestamp(batch[-1]["at"]) <= cutoff:
            break
    df = pd.DataFrame(rows).rename(columns={"reviewId": "id", "content": "raw_text", "score": "rating", "at": "date"})
    df = df.dropna(subset=["raw_text"])
    df = df[(pd.to_datetime(df["date"]) > cutoff) & ~df["id"].isin(old["id"])]

    # preprocessing sama dengan tahap 2; teks yang identik dengan dataset lama ikut dibuang (tahap 2 juga membuang duplikat)
    df["text"] = df["raw_text"].astype(str).map(preprocess)
    df = df[df["text"].str.split().str.len() >= MIN_WORDS].drop_duplicates(subset="text")
    df = df[~df["text"].isin(set(pd.read_csv(DATA / "reviews_clean.csv")["text"]))]
    FRESH.mkdir(exist_ok=True)
    df[["id", "raw_text", "text", "rating", "date"]].to_csv(FRESH / "reviews_fresh.csv", index=False)

    sample = df.sample(min(FRESH_SIZE, len(df)), random_state=SEED).reset_index(drop=True)
    sample.insert(0, "no", range(1, len(sample) + 1))
    sample["label"] = ""
    sample[["no", "id", "raw_text", "label"]].to_csv(LABELS, index=False, encoding="utf-8-sig")
    print(f"Lembar label: {LABELS} ({len(sample)} dari {len(df)} ulasan baru)")


def indobert_predict(folder, texts):
    import torch
    from transformers import BertForSequenceClassification, BertTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizer.from_pretrained(folder)
    model = BertForSequenceClassification.from_pretrained(folder).to(device).eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            enc = tokenizer(texts[i:i + 32], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(device)
            preds.append(model(**enc).logits.argmax(1).cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(preds)


def performance(y, preds):
    """Tabel kinerja: accuracy (CI Wilson), macro F1 (CI bootstrap), recall positif, FP, FN."""
    rows = ["| Model | Accuracy (CI 95%) | Macro F1 (CI 95%) | Recall positif | FP | FN |", "|---|---|---|---|---|---|"]
    for name, p in preds.items():
        acc, a_lo, a_hi = wilson(int((p == y).sum()), len(y))
        f_lo, f_hi = bootstrap_ci(y, p, np.random.default_rng(SEED))[1]
        tp, fp, fn = int(((p == 1) & (y == 1)).sum()), int(((p == 1) & (y == 0)).sum()), int(((p == 0) & (y == 1)).sum())
        rows.append(f"| {name} | {acc:.4f} ({a_lo:.4f}-{a_hi:.4f}) | {macro_f1(y, p):.4f} ({f_lo:.4f}-{f_hi:.4f}) | "
                    f"{tp / max(1, tp + fn):.4f} | {fp} | {fn} |")
    return rows


def evaluate():
    sheet = pd.read_csv(LABELS, encoding="utf-8-sig", dtype={"label": str, "koreksi": str})
    sheet["label"] = sheet["label"].str.strip()
    assert sheet["label"].isin(["0", "1", "-"]).all(), "isi semua label dengan 1, 0, atau -"
    sheet = sheet.join(pd.read_csv(FRESH / "reviews_fresh.csv").set_index("id")[["text", "date"]], on="id")
    test = sheet[sheet["label"] != "-"].reset_index(drop=True)
    y, texts = test["label"].astype(int).values, test["text"].tolist()

    # ---------- Prediksi ----------
    preds = {"TF-IDF kata 1-2 gram + regresi logistik": tfidf_lr(pd.read_csv(DATA / "train_baseline.csv"), texts, ngram_range=(1, 2))}
    for seed in [SEED] + EXTRA_SEEDS:
        for mode in ["baseline", "augmented"]:
            print(f"Prediksi {mode} seed {seed} ...", flush=True)
            folder = MODELS / mode if seed == SEED else MODELS / f"seed_{seed}" / mode
            preds[f"IndoBERT {mode} (seed {seed})"] = indobert_predict(folder, texts)

    # ---------- Laporan ----------
    n_dash = len(sheet) - len(test)
    L = ["# Test ulasan baru berlabel buta", "",
         f"Ulasan Gojek yang terbit setelah ulasan terakhir dataset ({sheet['date'].min()[:10]} s.d. {sheet['date'].max()[:10]}), "
         f"diambil acak {len(sheet)} ulasan. Peneliti melabeli dari teks asli tanpa melihat rating, label leksikon/Claude, "
         "atau prediksi model. Model tidak dilatih ulang.", "",
         f"* Netral / campur / tidak jelas (`-`, tidak dihitung): {n_dash} ({n_dash / len(sheet):.1%}) - "
         "ulasan seperti ini juga dibuang dari dataset saat pelabelan",
         f"* Dievaluasi: {len(test)} ulasan ({(y == 0).sum()} negatif, {(y == 1).sum()} positif)", "",
         "## 1. Kinerja", "", "Selang kepercayaan 95%: akurasi = Wilson, macro F1 = bootstrap 1.000 kali.", ""]
    L += performance(y, preds)

    y_fix = test.get("koreksi", test["label"]).fillna(test["label"]).str.strip()
    changed = (y_fix != test["label"]).values
    if changed.any():                                   # angka tambahan: label setelah koreksi peneliti
        keep = (y_fix != "-").values
        L += ["", "### Setelah koreksi peneliti (angka tambahan)", "",
              f"{changed.sum()} label dikoreksi (no {', '.join(map(str, test.loc[changed, 'no']))}) setelah prediksi terlihat. "
              "Yang dicek hanya ulasan yang salah diprediksi, jadi angka ini cenderung menguntungkan model; "
              "angka utama tetap tabel di atas.", ""]
        L += performance(y_fix[keep].astype(int).values, {k: v[keep] for k, v in preds.items()})

    L += ["", "## 2. McNemar baseline vs augmentasi (seed sama)", "",
          "| Seed | Baseline benar, aug salah | Baseline salah, aug benar | p-value |", "|---|---|---|---|"]
    for seed in [SEED] + EXTRA_SEEDS:
        only_b, only_a, pv = mcnemar(y, preds[f"IndoBERT baseline (seed {seed})"], preds[f"IndoBERT augmented (seed {seed})"])
        L.append(f"| {seed} | {only_b} | {only_a} | {pv:.4f} |")

    base, aug = preds[f"IndoBERT baseline (seed {SEED})"], preds[f"IndoBERT augmented (seed {SEED})"]
    L += ["", f"## 3. Ulasan yang salah diprediksi (seed {SEED})", "",
          "| No | Teks asli | Label manusia | Baseline | Augmented |", "|---|---|---|---|---|"]
    for i in np.where((base != y) | (aug != y))[0]:
        raw = str(test.loc[i, "raw_text"]).replace("|", "/").replace("\n", " ")
        L.append(f"| {test.loc[i, 'no']} | {raw} | {y[i]} | {base[i]} | {aug[i]} |")

    text = "\n".join(L)
    (RESULTS / "fresh_test.md").write_text(text, encoding="utf-8")
    json.dump({k: v.tolist() for k, v in preds.items()}, open(RESULTS / "fresh_test_predictions.json", "w"))
    print(text)


parser = argparse.ArgumentParser()
parser.add_argument("step", choices=["scrape", "evaluate"])
{"scrape": scrape, "evaluate": evaluate}[parser.parse_args().step]()
