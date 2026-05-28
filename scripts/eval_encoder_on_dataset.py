#!/usr/bin/env python
"""Run a trained encoder classifier (BERT/RoBERTa/HateBERT/DeBERTa) on any
binary-labeled JSONL (matching SynBullying format). Used for cross-domain
evaluation (e.g., applying SynBullying-trained model to OLID).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding)


LABELS = ["not_harmful", "harmful"]


def parse_label(o):
    s = o.strip().lower()
    if s.startswith("not"):
        return 0
    if s.startswith("harmful"):
        return 1
    return -1


def load_split(path):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            lab = d.get("label", parse_label(d.get("output", "")))
            rows.append({"text": d["input"], "label": int(lab)})
    return Dataset.from_list(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test_jsonl", required=True)
    ap.add_argument("--metrics_out", required=True)
    ap.add_argument("--predictions_out", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=256)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.ckpt, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or "[PAD]"
    model = AutoModelForSequenceClassification.from_pretrained(args.ckpt).cuda().eval()

    ds = load_split(args.test_jsonl)
    test_text = list(ds["text"])

    def encode(b):
        return tok(b["text"], truncation=True, max_length=args.max_length)

    ds = ds.map(encode, batched=True, remove_columns=["text"])
    coll = DataCollatorWithPadding(tokenizer=tok)
    loader = torch.utils.data.DataLoader(
        ds.with_format("torch", columns=["input_ids", "attention_mask", "label"]),
        batch_size=args.batch_size, shuffle=False, collate_fn=coll,
    )

    preds: list[int] = []
    labels: list[int] = []
    probs1: list[float] = []  # P(class=1 | x) -- harmful probability
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].cuda()
            attn = batch["attention_mask"].cuda()
            logits = model(input_ids=input_ids, attention_mask=attn).logits
            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds.extend(logits.argmax(-1).cpu().tolist())
            probs1.extend(probs.cpu().tolist())
            labels.extend(batch["labels"].tolist() if "labels" in batch else batch["label"].tolist())

    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, labels=[0, 1], average=None, zero_division=0)
    pm, rm, f1m, _ = precision_recall_fscore_support(labels, preds, labels=[0, 1], average="macro", zero_division=0)
    cm = confusion_matrix(labels, preds, labels=[0, 1]).tolist()
    report = classification_report(labels, preds, target_names=LABELS, zero_division=0)

    metrics = {
        "model": args.ckpt,
        "test_jsonl": args.test_jsonl,
        "total": len(labels),
        "valid_predictions": len(labels),
        "invalid_predictions": 0,
        "accuracy": float(acc),
        "not_harmful_precision": float(p[0]), "not_harmful_recall": float(r[0]),
        "not_harmful_f1": float(f1[0]),
        "harmful_precision": float(p[1]), "harmful_recall": float(r[1]),
        "harmful_f1": float(f1[1]),
        "macro_precision": float(pm), "macro_recall": float(rm), "macro_f1": float(f1m),
        "confusion_matrix_labels": LABELS, "confusion_matrix": cm,
        "classification_report": report,
    }
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_out).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    Path(args.predictions_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.predictions_out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "text", "label", "pred", "prob_harmful"])
        for i, (t, l, pv, p1) in enumerate(
            zip(test_text, labels, preds, probs1)
        ):
            w.writerow(
                [i, t[:500], LABELS[int(l)], LABELS[int(pv)], f"{p1:.6f}"]
            )

    print("DONE", json.dumps({
        "ckpt": args.ckpt, "test": args.test_jsonl,
        "accuracy": float(acc), "macro_f1": float(f1m), "harmful_f1": float(f1[1])
    }))


if __name__ == "__main__":
    main()
