import os
import json
import argparse
import random
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

QTL_HINTS = (
    "qtl", "quantitative trait", "gwas", "genome-wide association",
    "snp", "marker", "linkage", "lod score", "qtls", "association study"
)

def apply_keyword_nudge(texts, logits, boost=0.25):
    import numpy as np
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    boosted = []
    for txt, lg in zip(texts, logits):
        lg2 = lg.copy()
        t = txt.lower()
        if any(k in t for k in QTL_HINTS):
            lg2[1] += boost
        boosted.append(lg2)
    return np.vstack(boosted)

def read_qtl_text_json(path: str) -> pd.DataFrame:
    with open(path, "r") as f:
        data = json.load(f)
    rows = []
    for d in data:
        title = (d.get("Title") or "").strip()
        abstract = (d.get("Abstract") or "").strip()
        cat = d.get("Category")
        if cat is None:
            continue
        text = (title + "\n\n" + abstract).strip()
        rows.append({"Title": title, "Abstract": abstract, "text": text, "label": int(cat)})
    return pd.DataFrame(rows)

def read_qtl_test_tsv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype={"PMID": str})
    df["Title"] = df["Title"].fillna("")
    df["Abstract"] = df["Abstract"].fillna("")
    df["text"] = (df["Title"].astype(str) + "\n\n" + df["Abstract"].astype(str)).str.strip()
    return df

class TextDataset(Dataset):
    def __init__(self, encodings: Dict[str, torch.Tensor], labels: Optional[List[int]] = None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_length", type=int, default=512)  # will be clamped to 512 anyway
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test", action="store_true", default=False)
    args = parser.parse_args()

    # set_seed(args.seed)

    # Check GPU
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # I/O paths
    train_path = os.path.join(args.data_dir, "QTL_text.json")
    test_path = os.path.join(args.data_dir, "QTL_test_unlabeled.tsv")
    save_root = "./outputs"
    save_name = args.model_name.replace('/', '_')
    save_dir = os.path.join(save_root, save_name)
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    df = read_qtl_text_json(train_path)
    test_df = read_qtl_test_tsv(test_path)

    tr_df, dv_df = train_test_split(
        df, test_size=0.1, random_state=args.seed, stratify=df["label"]
    )
    print(f"Train size: {len(tr_df)} | Dev size: {len(dv_df)} | Test size: {len(test_df)}")

    # Simple positive-class oversampling
    pos = tr_df[tr_df["label"] == 1]
    neg = tr_df[tr_df["label"] == 0]
    if len(pos) > 0:
        reps = max(1, len(neg) // len(pos))
        pos_os = pd.concat([pos] * reps, ignore_index=True)
        tr_df = pd.concat([neg, pos_os], ignore_index=True).sample(frac=1, random_state=args.seed)
    print("After oversampling:", tr_df["label"].value_counts().to_dict())

    # Tokenizer (from model or from saved dir in --test mode)
    if args.test:
        print(f"[TEST] Loading tokenizer from saved dir: {save_dir}")
        tokenizer = AutoTokenizer.from_pretrained(save_dir, use_fast=True)
    else:
        print(f"[TRAIN] Loading tokenizer & model: {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # BERT has a PAD token; this is just a safety net
    if tokenizer.pad_token is None and hasattr(tokenizer, "eos_token"):
        tokenizer.pad_token = tokenizer.eos_token

    effective_max_len = min(int(args.max_length), 512)  # hard cap for BERT

    def tok(batch_texts: List[str]):
        return tokenizer(
            batch_texts,
            truncation=True,
            max_length=effective_max_len,
            padding=True,
            return_tensors="pt"
        )

    print("Tokenizing...")
    tr_enc = tok(tr_df["text"].tolist())
    dv_enc = tok(dv_df["text"].tolist())
    te_enc = tok(test_df["text"].tolist())

    train_ds = TextDataset(tr_enc, tr_df["label"].tolist())
    dev_ds = TextDataset(dv_enc, dv_df["label"].tolist())
    test_ds = TextDataset(te_enc, labels=None)

    # Model
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if args.test:
        print(f"[TEST] Loading model from: {save_dir}")
        model = AutoModelForSequenceClassification.from_pretrained(
            save_dir,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=2,
            problem_type="single_label_classification"
        )

    model.config.pad_token_id = tokenizer.pad_token_id

    # Trainer / TrainingArguments
    training_args = TrainingArguments(
        output_dir=save_dir,
        overwrite_output_dir=True,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=0 if args.test else args.epochs,
        logging_steps=100,
        eval_steps=100,
        save_steps=100,
        eval_strategy="no" if args.test else "steps",
        save_strategy="no" if args.test else "steps",
        load_best_model_at_end=not args.test,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        fp16=False,
        bf16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        save_total_limit=5,
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)] if not args.test else None

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )

    if not args.test:
        print("[TRAIN] Training...")
        trainer.train()

        print(f"[TRAIN] Saving model & tokenizer to {save_dir}")
        trainer.save_model(save_dir)
        tokenizer.save_pretrained(save_dir)

    # Predict test and write Kaggle CSV
    print("[PREDICT] Predicting on test set...")
    raw_logits = trainer.predict(test_ds).predictions
    texts = test_df["text"].tolist()
    test_logits = apply_keyword_nudge(texts, raw_logits, boost=0.35)
    test_labels = test_logits.argmax(-1)

    sub = pd.DataFrame({"PMID": test_df["PMID"].astype(str), "Label": test_labels.astype(int)})
    sub_path = os.path.join(save_dir, f"{save_name}_test.csv")
    sub.to_csv(sub_path, index=False)
    print(f"[PREDICT] Wrote Kaggle submission to: {sub_path}")

if __name__ == "__main__":
    main()
