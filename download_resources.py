"""Mengunduh InSet Lexicon (polaritas kata) dan IndoCollex (kamus slang) ke folder resources/."""
import pandas as pd
import requests
from config import RESOURCES

RESOURCES.mkdir(exist_ok=True)
INSET = "https://raw.githubusercontent.com/fajri91/InSet/master/{}.tsv"
SLANG = "https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv"

for name in ["positive", "negative"]:
    (RESOURCES / f"inset_{name}.tsv").write_bytes(requests.get(INSET.format(name), timeout=60).content)
(RESOURCES / "indocollex.csv").write_bytes(requests.get(SLANG, timeout=120).content)

pos = pd.read_csv(RESOURCES / "inset_positive.tsv", sep="\t")
neg = pd.read_csv(RESOURCES / "inset_negative.tsv", sep="\t")
slang = pd.read_csv(RESOURCES / "indocollex.csv")
print(f"InSet positif: {len(pos)} kata, negatif: {len(neg)} kata, IndoCollex: {len(slang)} baris")
