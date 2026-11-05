"""TAHAP 8 - Evaluasi akhir dengan label test yang DIVERIFIKASI MANUSIA (boleh baru sebagian).
Input : data/test_verification.csv  (kolom verified: 1 / 0 / - ; kosong = belum dicek)
        results/baseline_predictions.csv, results/augmented_predictions.csv (urutan sama dengan data/test.csv)
Output: results/comparison_verified.md
Jika test_verification.csv belum ada, skrip membuatnya (kolom verified kosong) lalu berhenti.
Isi: (1) audit label dataset per sumber + selang kepercayaan 95% (Wilson),
     (2) kinerja kedua model pada (a) seluruh test setelah koreksi verifikasi dan
         (b) hanya ulasan yang sudah dicek manusia, masing-masing dengan uji McNemar,
     (3) jumlah salah prediksi per sumber label.
Model TIDAK dilatih ulang: test set hanya dipakai untuk evaluasi."""
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, precision_recall_fscore_support

from config import DATA, RESULTS
from utils import mcnemar, wilson


def pct_ci(ok, n):
    """Mis. '99.80% (99.20% - 100.00%)': proporsi + selang kepercayaan 95% Wilson."""
    if n == 0:
        return "-"
    p, lo, hi = wilson(ok, n)
    return f"{p:.2%} ({lo:.2%} - {hi:.2%})"


vfile = DATA / "test_verification.csv"
if not vfile.exists():                                 # pertama kali: buat lembar verifikasi kosong dari test.csv
    sheet = pd.read_csv(DATA / "test.csv").merge(pd.read_csv(DATA / "reviews_clean.csv")[["id", "raw_text"]],
                                                 on="id", how="left")
    sheet.insert(0, "no", range(1, len(sheet) + 1))
    sheet["verified"] = ""
    sheet[["no", "id", "text", "raw_text", "label", "source", "verified"]].to_csv(vfile, index=False, encoding="utf-8-sig")
    print(f"{vfile} dibuat. Isi kolom verified (1 / 0 / -), lalu jalankan skrip ini lagi.")
    raise SystemExit

v = pd.read_csv(vfile, encoding="utf-8-sig", dtype={"verified": str})
v["verified"] = v["verified"].fillna("").str.strip()
# sumber "manual" dirinci menurut asal labelnya (user sendiri / aturan ikut Claude / gaya user oleh Claude)
manual = pd.read_csv(DATA / "manual_label.csv", encoding="utf-8-sig")
origin = v["id"].map(dict(zip(manual["id"], manual["label_source"]))).fillna("?")
v["sumber"] = v["source"].where(v["source"] != "manual", "manual: " + origin)
checked = v[v["verified"] != ""]
kept = checked[checked["verified"] != "-"].copy()
kept["verified"] = kept["verified"].astype(int)
kept["benar"] = kept["verified"] == kept["label"]

complete = len(checked) == len(v)
lines = ["# Evaluasi pada test set terverifikasi manusia", "",
         f"Sudah diverifikasi: {len(checked)} dari {len(v)} ulasan test "
         f"({'LENGKAP' if complete else 'SEBAGIAN'})."]
if not complete:
    lines.append(f"Ulasan yang dicek BUKAN sampel acak: proporsi positifnya {(checked['label'] == 1).mean():.0%}, "
                 f"sedangkan seluruh test {(v['label'] == 1).mean():.0%}. Lihat kolom 'Di test' vs 'Dicek' di bawah.")

# ---------- 1. Audit label dataset ----------
lines += ["", "## 1. Audit label dataset (label dataset vs label terverifikasi)", "",
          "| Sumber label | Di test | Dicek | Benar | Akurasi label (CI 95%) | Diubah | Dibuang |",
          "|---|---|---|---|---|---|---|"]
for src, g in v.groupby("sumber"):
    k = kept[kept["sumber"] == src]
    lines.append(f"| {src} | {len(g)} | {(g['verified'] != '').sum()} | {k['benar'].sum()} | "
                 f"{pct_ci(k['benar'].sum(), len(k))} | {(~k['benar']).sum()} | {(g['verified'] == '-').sum()} |")
lines.append(f"| **Semua** | {len(v)} | {len(checked)} | {kept['benar'].sum()} | "
             f"{pct_ci(kept['benar'].sum(), len(kept))} | {(~kept['benar']).sum()} | {(checked['verified'] == '-').sum()} |")
lines.append(f"\nCohen's kappa label dataset vs terverifikasi: {cohen_kappa_score(kept['label'], kept['verified']):.3f}")

# ---------- 2. Evaluasi model ----------
test = pd.read_csv(DATA / "test.csv")
preds = {}
for mode in ["baseline", "augmented"]:
    p = pd.read_csv(RESULTS / f"{mode}_predictions.csv")
    assert (p["text"].values == test["text"].values).all(), "urutan prediksi tidak sama dengan test.csv"
    preds[mode] = dict(zip(test["id"], p["pred"]))

# (a) seluruh test: label terverifikasi menggantikan label dataset, yang ditandai "-" dibuang
final = v[v["verified"] != "-"].copy()
final["y"] = final["label"]
is_checked = final["verified"] != ""
final.loc[is_checked, "y"] = final.loc[is_checked, "verified"].astype(int)
eval_sets = [(f"2a. Seluruh test setelah koreksi verifikasi ({is_checked.sum()} dicek manusia)", final["id"], final["y"])]
if not complete:
    eval_sets.append(("2b. Hanya ulasan yang sudah dicek manusia", kept["id"], kept["verified"]))

for title, ids, y in eval_sets:
    y = y.values
    lines += ["", f"## {title}: {len(y)} ulasan, positif {(y == 1).mean():.0%}", "",
              "| Model | Kelas | Precision | Recall | F1 | Support |", "|---|---|---|---|---|---|"]
    summary, yhat = [], {}
    for name, mode in [("Baseline", "baseline"), ("+ Augmentasi", "augmented")]:
        yh = yhat[mode] = ids.map(preds[mode]).values
        pr, rc, f1, sup = precision_recall_fscore_support(y, yh, labels=[0, 1], zero_division=0)
        lines += [f"| {name} | {cls} | {pr[i]:.4f} | {rc[i]:.4f} | {f1[i]:.4f} | {sup[i]} |"
                  for i, cls in enumerate(["negatif", "positif"])]
        (tn, fp), (fn, tp) = confusion_matrix(y, yh, labels=[0, 1])
        summary.append(f"| {name} | {accuracy_score(y, yh):.4f} | {f1.mean():.4f} | {pr[1]:.4f} | {rc[1]:.4f} | "
                       f"{f1[1]:.4f} | {tp} | {tn} | {fp} | {fn} |")
    lines += ["", "| Model | Accuracy | Macro F1 | Precision positif | Recall positif | F1 positif | TP | TN | FP | FN |",
              "|---|---|---|---|---|---|---|---|---|---|"] + summary

    only_b, only_a, p_value = mcnemar(y, yhat["baseline"], yhat["augmented"])
    lines.append(f"\nUji McNemar (exact): baseline benar & augmented salah = {only_b}, kebalikannya = {only_a}, "
                 f"p-value = {p_value:.4f} -> {'berbeda signifikan' if p_value < 0.05 else 'TIDAK berbeda signifikan'} "
                 f"(alpha = 0,05)")

# ---------- 3. Di mana model salah? ----------
lines += ["", "## 3. Jumlah salah prediksi per sumber label (seluruh test setelah koreksi)", "",
          "| Sumber label | Ulasan | Salah baseline | Salah augmented |", "|---|---|---|---|"]
for src, g in final.groupby("sumber"):
    wrong = [(g["id"].map(preds[mode]) != g["y"]).sum() for mode in ["baseline", "augmented"]]
    lines.append(f"| {src} | {len(g)} | {wrong[0]} | {wrong[1]} |")

text = "\n".join(lines)
(RESULTS / "comparison_verified.md").write_text(text, encoding="utf-8")
print(text)
