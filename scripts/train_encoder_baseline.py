#!/usr/bin/env python
"""Encoder baseline trainer for SynBullying strict splits.

Reads instruction-tuning style JSONL (instruction/input/output) and trains a
sequence classification head on a HF encoder model. Output metric JSON matches
the existing LoRA evaluation scripts.
"""
import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, EarlyStoppingCallback,
                          Trainer, TrainingArguments)


LABELS = ["not_harmful", "harmful"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_label(output: str) -> int:
    s = output.strip().lower()
    if s.startswith("not"):
        return 0
    if s.startswith("harmful"):
        return 1
    if "not harmful" in s or "not_harmful" in s:
        return 0
    if "harmful" in s:
        return 1
    raise ValueError(f"Cannot parse label from: {output!r}")


def load_split(path: str) -> Dataset:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            rows.append({"text": d["input"], "label": parse_label(d["output"])})
    return Dataset.from_list(rows)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], average=None, zero_division=0
    )
    pm, rm, f1m, _ = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], average="macro", zero_division=0
    )
    return {
        "accuracy": float(acc),
        "not_harmful_f1": float(f1[0]),
        "harmful_f1": float(f1[1]),
        "macro_f1": float(f1m),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--data_dir",
                    default="<DATA_ROOT>/data")
    ap.add_argument("--window", type=int, default=4,
                    choices=[0, 1, 2, 4, 8])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--metrics_out", required=True)
    ap.add_argument("--predictions_out", required=True)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=float, default=5.0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--no_fp16", dest="fp16", action="store_false")
    args = ap.parse_args()

    set_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    train_p = f"{args.data_dir}/synbullying_strict_w{args.window}_train.jsonl"
    dev_p = f"{args.data_dir}/synbullying_strict_w{args.window}_dev.jsonl"
    test_p = f"{args.data_dir}/synbullying_strict_w{args.window}_test.jsonl"
    train_ds = load_split(train_p)
    dev_ds = load_split(dev_p)
    test_ds = load_split(test_p)
    print(f"[data] train={len(train_ds)} dev={len(dev_ds)} test={len(test_ds)}")

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or "[PAD]"
    # When context is much larger than max_length, the *target* utterance (at
    # the end of the input) is what we must preserve. Left-truncation keeps it.
    tok.truncation_side = "left"

    def encode(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_length)

    train_ds = train_ds.map(encode, batched=True, remove_columns=["text"])
    dev_ds = dev_ds.map(encode, batched=True, remove_columns=["text"])
    test_text = list(test_ds["text"])
    test_ds = test_ds.map(encode, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "not_harmful", 1: "harmful"},
        label2id={"not_harmful": 0, "harmful": 1},
    )

    collator = DataCollatorWithPadding(tokenizer=tok)
    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        seed=args.seed,
        data_seed=args.seed,
        report_to="none",
        fp16=args.fp16,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    # Save the best (already loaded due to load_best_model_at_end) to the
    # top-level output_dir so it survives our checkpoint-* cleanup.
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

    out = trainer.predict(test_ds)
    preds = out.predictions.argmax(-1)
    labels = out.label_ids
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], average=None, zero_division=0
    )
    pm, rm, f1m, _ = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], average="macro", zero_division=0
    )
    cm = confusion_matrix(labels, preds, labels=[0, 1]).tolist()
    report = classification_report(labels, preds, target_names=LABELS,
                                   zero_division=0)

    metrics = {
        "model": f"{args.model_name} strict w{args.window} seed={args.seed}",
        "total": int(len(labels)),
        "valid_predictions": int(len(labels)),
        "invalid_predictions": 0,
        "accuracy": float(acc),
        "not_harmful_precision": float(p[0]),
        "not_harmful_recall": float(r[0]),
        "not_harmful_f1": float(f1[0]),
        "harmful_precision": float(p[1]),
        "harmful_recall": float(r[1]),
        "harmful_f1": float(f1[1]),
        "macro_precision": float(pm),
        "macro_recall": float(rm),
        "macro_f1": float(f1m),
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": cm,
        "classification_report": report,
        "config": {
            "model_name": args.model_name,
            "window": args.window,
            "seed": args.seed,
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "fp16": args.fp16,
        },
    }
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_out, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)

    Path(args.predictions_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.predictions_out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "text", "label", "pred"])
        for i, (t, l, pv) in enumerate(zip(test_text, labels, preds)):
            w.writerow([i, t[:500], LABELS[int(l)], LABELS[int(pv)]])

    print("DONE", json.dumps({
        "model": args.model_name, "seed": args.seed, "window": args.window,
        "accuracy": float(acc), "macro_f1": float(f1m),
        "harmful_f1": float(f1[1]),
    }))


if __name__ == "__main__":
    main()
