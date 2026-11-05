"""TAHAP 9 - Ringkasan beberapa seed training: apakah selisih baseline vs augmented konsisten?
Input : results/<mode>_* (SEED utama) dan results/seed_<n>/<mode>_* (EXTRA_SEEDS), dari 06_train.py --seed
Output: results/multiseed.md
Data split & test set SAMA untuk semua seed; yang berbeda hanya seed training (inisialisasi, urutan batch).
Seed yang hasilnya belum ada dilewati dan dicatat di laporan."""
import json
from statistics import mean, stdev

import pandas as pd

from config import EXTRA_SEEDS, RESULTS, SEED
from utils import mcnemar

METRICS = [("Accuracy", lambda m: m["accuracy"]), ("Macro F1", lambda m: m["macro"]["f1"]),
           ("Positif precision", lambda m: m["per_class"]["positif"]["precision"]),
           ("Positif recall", lambda m: m["per_class"]["positif"]["recall"]),
           ("Positif F1", lambda m: m["per_class"]["positif"]["f1"])]

rows, missing = [], []
for seed in [SEED] + EXTRA_SEEDS:
    folder = RESULTS if seed == SEED else RESULTS / f"seed_{seed}"
    files = {mode: folder / f"{mode}_metrics.json" for mode in ["baseline", "augmented"]}
    if not all(f.exists() for f in files.values()):
        missing.append(seed)
        continue
    m = {mode: json.load(open(f)) for mode, f in files.items()}
    p = {mode: pd.read_csv(folder / f"{mode}_predictions.csv") for mode in files}
    row = {"seed": seed, "mcnemar_p": mcnemar(p["baseline"]["label"], p["baseline"]["pred"], p["augmented"]["pred"])[2]}
    for name, get in METRICS:
        row[f"{name}|baseline"], row[f"{name}|augmented"] = get(m["baseline"]), get(m["augmented"])
    rows.append(row)

lines = ["# Ringkasan beberapa seed training", "",
         f"Seed selesai: {', '.join(str(r['seed']) for r in rows) or '-'}"
         + (f" | belum ada hasil: {', '.join(map(str, missing))}" if missing else ""), "",
         "## Per seed", "", "| Seed | Metrik | Baseline | + Augmentasi | Selisih |", "|---|---|---|---|---|"]
for r in rows:
    for name, _ in METRICS:
        b, a = r[f"{name}|baseline"], r[f"{name}|augmented"]
        lines.append(f"| {r['seed']} | {name} | {b:.4f} | {a:.4f} | {a - b:+.4f} |")
    lines.append(f"| {r['seed']} | McNemar p | | | {r['mcnemar_p']:.4f} |")

if len(rows) >= 2:
    lines += ["", f"## Rata-rata ± simpangan baku ({len(rows)} seed)", "",
              "| Metrik | Baseline | + Augmentasi | Selisih rata-rata | Seed augmentasi lebih baik |", "|---|---|---|---|---|"]
    for name, _ in METRICS:
        b = [r[f"{name}|baseline"] for r in rows]
        a = [r[f"{name}|augmented"] for r in rows]
        diff = [y - x for x, y in zip(b, a)]
        lines.append(f"| {name} | {mean(b):.4f} ± {stdev(b):.4f} | {mean(a):.4f} ± {stdev(a):.4f} | "
                     f"{mean(diff):+.4f} ± {stdev(diff):.4f} | {sum(d > 0 for d in diff)}/{len(diff)} |")
    signif = sum(r["mcnemar_p"] < 0.05 for r in rows)
    lines += ["", f"Seed dengan perbedaan signifikan (McNemar, alpha 0,05): {signif}/{len(rows)}.",
              "Simpangan baku dengan sedikit seed masih kasar; laporkan sebagai indikasi kestabilan, bukan uji formal."]

text = "\n".join(lines)
(RESULTS / "multiseed.md").write_text(text, encoding="utf-8")
print(text)
