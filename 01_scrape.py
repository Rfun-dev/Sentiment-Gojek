"""TAHAP 1 - Scraping ulasan Gojek dari Google Play Store -> data/reviews_raw.csv"""
import time

import pandas as pd
from google_play_scraper import Sort, reviews
from tqdm import tqdm

from config import APP_ID, DATA, SCRAPE_COUNT

DATA.mkdir(exist_ok=True)
rows, token = [], None
bar = tqdm(total=SCRAPE_COUNT, desc="Scraping")
while len(rows) < SCRAPE_COUNT:
    try:
        batch, token = reviews(APP_ID, lang="id", country="id", sort=Sort.NEWEST,
                               count=200, continuation_token=token)
    except Exception as e:                     # koneksi putus -> coba lagi
        print("Error:", e, "- ulangi 10 detik lagi")
        time.sleep(10)
        continue
    if not batch:
        break                                  # ulasan habis
    rows += batch
    bar.update(len(batch))
    if token is None:
        break
bar.close()

df = pd.DataFrame(rows)[["reviewId", "content", "score", "at"]]
df = df.rename(columns={"reviewId": "id", "score": "rating", "at": "date"})
df = df.dropna(subset=["content"]).head(SCRAPE_COUNT)
df.to_csv(DATA / "reviews_raw.csv", index=False)
print(f"Tersimpan {len(df)} ulasan -> data/reviews_raw.csv")
