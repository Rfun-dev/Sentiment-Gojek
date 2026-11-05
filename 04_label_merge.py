"""TAHAP 4 - Gabungkan label otomatis + label manual -> data/labeled.csv
Baris manual dengan manual_label kosong (atau selain 0/1) dianggap dibuang.
Jika ada data/label_corrections.csv (kolom id, label: 1 / 0 / -), koreksi hasil audit label
diterapkan terakhir: 1/0 mengganti label, "-" membuang ulasan."""
import pandas as pd

from config import DATA

auto = pd.read_csv(DATA / "labeled_auto.csv")
manual = pd.read_csv(DATA / "manual_label.csv", encoding="utf-8-sig", dtype={"manual_label": str})
n_manual_total = len(manual)

manual["manual_label"] = manual["manual_label"].str.strip().str.replace(".0", "", regex=False)
manual = manual[manual["manual_label"].isin(["0", "1"])].copy()
manual["label"] = manual["manual_label"].astype(int)
manual["source"] = "manual"

final = pd.concat([auto, manual[["id", "text", "label", "source"]]], ignore_index=True)
final = final.drop_duplicates("text")

corrections_file = DATA / "label_corrections.csv"
if corrections_file.exists():                          # koreksi dari audit label
    corr = pd.read_csv(corrections_file, dtype={"label": str})
    new = final["id"].map(dict(zip(corr["id"], corr["label"].str.strip())))    # NaN = tidak dikoreksi
    changed = (new.notna() & (new != final["label"].astype(str))).sum()
    final.loc[new.isin(["0", "1"]), "label"] = new[new.isin(["0", "1"])].astype(int)
    final = final[new.ne("-")]
    print(f"Koreksi audit diterapkan: {changed} label berubah/dibuang")

final.to_csv(DATA / "labeled.csv", index=False)

print(f"Label otomatis (sepakat): {len(auto)}")
print(f"Label manual terisi     : {len(manual)} dari {n_manual_total} (sisanya dibuang)")
print(f"Total data berlabel     : {len(final)}")
print("Distribusi kelas (1=positif, 0=negatif):")
print(final["label"].value_counts().to_string())
print(f"Rasio minoritas/mayoritas: {final['label'].value_counts().min() / final['label'].value_counts().max():.2f}")
