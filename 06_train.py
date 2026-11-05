"""TAHAP 6 - Fine-tuning IndoBERT lalu evaluasi di data test.
Pemakaian:
    python 06_train.py --mode baseline     # tanpa augmentasi
    python 06_train.py --mode augmented    # dengan augmentasi LLM
    python 06_train.py --mode baseline --seed 1   # ulangan dengan seed training lain (data & split tetap sama)
Output: models/<mode>/ (model terbaik), results/<mode>_metrics.json, results/<mode>_predictions.csv
        seed selain SEED di config disimpan terpisah: models/seed_<n>/<mode>/, results/seed_<n>/
    DATA_DIR=data/robust_waktu python 06_train.py --mode baseline --tag robust_waktu   # split lain (tahap 11)"""
import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from transformers import BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup

from config import (BATCH_SIZE, DATA, EPOCHS, LR, MAX_LEN, MICRO_BATCH, MODEL_NAME, MODELS,
                    PATIENCE, RESULTS, SEED, WEIGHT_DECAY)

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["baseline", "augmented"], required=True)
parser.add_argument("--seed", type=int, default=SEED, help="seed training; split data tidak ikut berubah")
parser.add_argument("--tag", help="nama folder hasil (mis. robust_waktu); dipakai bersama env DATA_DIR")
args = parser.parse_args()
mode, seed = args.mode, args.seed
tag = args.tag or ("" if seed == SEED else f"seed_{seed}")   # seed utama memakai folder lama, lainnya folder sendiri
out_dir = RESULTS / tag

# ---------- Seed (supaya hasil bisa diulang) ----------
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = device == "cuda"
print(f"Mode: {mode} | seed: {seed} | device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if use_amp else ""))

# ---------- Data & tokenisasi ----------
train_df = pd.read_csv(DATA / f"train_{mode}.csv")
val_df, test_df = pd.read_csv(DATA / "val.csv"), pd.read_csv(DATA / "test.csv")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


def batches(df, batch_size, shuffle):
    """Menghasilkan batch (input_ids, attention_mask, label). Padding mengikuti kalimat terpanjang di batch."""
    idx = np.random.permutation(len(df)) if shuffle else np.arange(len(df))
    for i in range(0, len(df), batch_size):
        part = df.iloc[idx[i:i + batch_size]]
        enc = tokenizer(part["text"].tolist(), truncation=True, max_length=MAX_LEN,
                        padding=True, return_tensors="pt")
        yield (enc["input_ids"].to(device), enc["attention_mask"].to(device),
               torch.tensor(part["label"].values).to(device))


# ---------- Model ----------
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
accum = BATCH_SIZE // MICRO_BATCH                      # berapa micro-batch per satu update bobot
steps_per_epoch = -(-len(train_df) // BATCH_SIZE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps_per_epoch * EPOCHS), steps_per_epoch * EPOCHS)
scaler = torch.amp.GradScaler(enabled=use_amp)         # mixed precision (hemat VRAM)


def predict(df):
    model.eval()
    logits, labels, loss_sum = [], [], 0.0
    with torch.no_grad():
        for ids, mask, y in batches(df, 32, shuffle=False):
            with torch.autocast(device, dtype=torch.float16, enabled=use_amp):
                out = model(input_ids=ids, attention_mask=mask, labels=y)
            loss_sum += out.loss.item() * len(y)
            logits.append(out.logits.float().cpu()); labels.append(y.cpu())
    logits, labels = torch.cat(logits), torch.cat(labels).numpy()
    return logits.argmax(1).numpy(), labels, loss_sum / len(df)


def fmt(sec):
    sec = int(sec)
    return f"{sec // 3600}j {sec % 3600 // 60:02d}m {sec % 60:02d}d" if sec >= 3600 else f"{sec // 60}m {sec % 60:02d}d"


# ---------- Training loop ----------
best_loss, bad_epochs = float("inf"), 0
save_dir = MODELS / tag / mode
steps_per_epoch_micro = -(-len(train_df) // MICRO_BATCH)
total_steps, done_steps, t_start = steps_per_epoch_micro * EPOCHS, 0, time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    total, n = 0.0, 0
    for step, (ids, mask, y) in enumerate(batches(train_df, MICRO_BATCH, shuffle=True), 1):
        done_steps += 1
        with torch.autocast(device, dtype=torch.float16, enabled=use_amp):
            loss = model(input_ids=ids, attention_mask=mask, labels=y).loss
        scaler.scale(loss / accum).backward()
        total += loss.item(); n += 1
        if step % accum == 0:                           # update bobot tiap `accum` micro-batch
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad()
        if step % 100 == 0 or step == steps_per_epoch_micro:       # progres: persen & perkiraan sisa waktu
            elapsed = time.time() - t_start
            eta = elapsed / done_steps * (total_steps - done_steps)
            print(f"  [{100 * done_steps / total_steps:5.1f}%] epoch {epoch}/{EPOCHS} step {step}/{steps_per_epoch_micro}"
                  f" | loss {total / n:.4f} | berjalan {fmt(elapsed)} | sisa ~{fmt(eta)}", flush=True)
    preds, labels, val_loss = predict(val_df)
    f1 = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)[2]
    print(f"Epoch {epoch}: train loss {total / n:.4f} | val loss {val_loss:.4f} | "
          f"val acc {accuracy_score(labels, preds):.4f} | val F1 {f1:.4f}")
    if val_loss < best_loss:                            # simpan model terbaik
        best_loss, bad_epochs = val_loss, 0
        model.save_pretrained(save_dir); tokenizer.save_pretrained(save_dir)
    else:
        bad_epochs += 1
        if bad_epochs >= PATIENCE:
            print("Early stopping.")
            break

# ---------- Evaluasi di data test dengan model terbaik ----------
model = BertForSequenceClassification.from_pretrained(save_dir).to(device)
preds, labels, _ = predict(test_df)
p, r, f1, support = precision_recall_fscore_support(labels, preds, labels=[0, 1], zero_division=0)
mp, mr, mf1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
metrics = {
    "mode": mode, "seed": seed, "train_size": len(train_df), "accuracy": accuracy_score(labels, preds),
    "macro": {"precision": mp, "recall": mr, "f1": mf1},
    "per_class": {name: {"precision": p[i], "recall": r[i], "f1": f1[i], "support": int(support[i])}
                  for i, name in enumerate(["negatif", "positif"])},
    "confusion_matrix [[TN,FP],[FN,TP]]": confusion_matrix(labels, preds).tolist(),
}
out_dir.mkdir(parents=True, exist_ok=True)
json.dump(metrics, open(out_dir / f"{mode}_metrics.json", "w"), indent=2, default=float)
pd.DataFrame({"text": test_df["text"], "label": labels, "pred": preds}).to_csv(
    out_dir / f"{mode}_predictions.csv", index=False)
print(json.dumps(metrics, indent=2, default=float))
