"""TAHAP 7 - Laporan eksperimen baseline vs augmented -> results/comparison.md
Isi: konfigurasi & seed, jumlah data per kelas tiap partisi, metrik per kelas, confusion matrix
(TP/TN/FP/FN), perbandingan, dan uji signifikansi McNemar pada test set yang sama."""
import json

import pandas as pd

import config
from config import DATA, RESULTS
from utils import mcnemar

base = json.load(open(RESULTS / "baseline_metrics.json"))
aug = json.load(open(RESULTS / "augmented_metrics.json"))
info = json.load(open(DATA / "split_info.json"))
minority = "positif" if info["minority_label"] == 1 else "negatif"
CM = "confusion_matrix [[TN,FP],[FN,TP]]"

lines = ["# Laporan eksperimen", "", "## Konfigurasi", "", "| Parameter | Nilai |", "|---|---|"]
for k in ["SEED", "MODEL_NAME", "MAX_LEN", "BATCH_SIZE", "MICRO_BATCH", "LR", "EPOCHS", "WEIGHT_DECAY",
          "PATIENCE", "TEST_SIZE", "VAL_SIZE", "DEDUP_SIM", "TARGET_RATIO"]:
    v = getattr(config, k)
    lines.append(f"| {k} | {f'{v:.4g}' if isinstance(v, float) else v} |")

lines += ["", "## Jumlah data per kelas (0 = negatif, 1 = positif)", "", "| Partisi | Negatif | Positif | Total |", "|---|---|---|---|"]
for part in ["train_baseline", "train_augmented", "val", "test"]:
    c = {int(k): v for k, v in info[part].items()}
    lines.append(f"| {part} | {c.get(0, 0)} | {c.get(1, 0)} | {c.get(0, 0) + c.get(1, 0)} |")
lines.append(f"\nKelas minoritas: **{minority}**; data minoritas yang dibuang dari train: {info['minority_removed_from_train']}")

lines += ["", "## Metrik per kelas (data test)", "",
          "| Model | Kelas | Precision | Recall | F1 | Support |", "|---|---|---|---|---|---|"]
for name, m in [("Baseline", base), ("+ Augmentasi", aug)]:
    for cls, v in m["per_class"].items():
        lines.append(f"| {name} | {cls} | {v['precision']:.4f} | {v['recall']:.4f} | {v['f1']:.4f} | {v['support']} |")

lines += ["", "## Confusion matrix (positif = kelas 1)", "", "| Model | TP | TN | FP | FN | Total salah |", "|---|---|---|---|---|---|"]
for name, m in [("Baseline", base), ("+ Augmentasi", aug)]:
    (tn, fp), (fn, tp) = m[CM]
    lines.append(f"| {name} | {tp} | {tn} | {fp} | {fn} | {fp + fn} |")

rows = [("Accuracy", base["accuracy"], aug["accuracy"]),
        ("Macro Precision", base["macro"]["precision"], aug["macro"]["precision"]),
        ("Macro Recall", base["macro"]["recall"], aug["macro"]["recall"]),
        ("Macro F1", base["macro"]["f1"], aug["macro"]["f1"])]
for m in ["precision", "recall", "f1"]:
    rows.append((f"Minoritas ({minority}) {m}", base["per_class"][minority][m], aug["per_class"][minority][m]))
lines += ["", "## Perbandingan", "", "| Metrik | Baseline | + Augmentasi | Selisih |", "|---|---|---|---|"]
lines += [f"| {n} | {b:.4f} | {a:.4f} | {a - b:+.4f} |" for n, b, a in rows]

# Uji McNemar (exact): apakah kedua model berbeda secara signifikan pada test set yang sama?
pb, pa = pd.read_csv(RESULTS / "baseline_predictions.csv"), pd.read_csv(RESULTS / "augmented_predictions.csv")
only_b, only_a, p_value = mcnemar(pb["label"], pb["pred"], pa["pred"])
lines += ["", "## Uji signifikansi McNemar (exact)", "",
          f"- Baseline benar, augmented salah: {only_b}",
          f"- Baseline salah, augmented benar: {only_a}",
          f"- p-value = {p_value:.4f} -> {'berbeda signifikan' if p_value < 0.05 else 'TIDAK berbeda signifikan'} (alpha = 0,05)"]

text = "\n".join(lines)
(RESULTS / "comparison.md").write_text(text, encoding="utf-8")
print(text)
