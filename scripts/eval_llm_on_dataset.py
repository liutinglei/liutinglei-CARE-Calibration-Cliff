#!/usr/bin/env python
"""Run a trained LoRA adapter on any instruction-style JSONL (matching the
SynBullying input/output schema). Used for cross-domain transfer evaluation
of LoRA-adapted LLMs.
"""
import argparse
import csv
import json
from pathlib import Path

import torch
from peft import PeftModel
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from transformers import AutoModelForCausalLM, AutoTokenizer

LABELS = ["not_harmful", "harmful"]


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
    return -1


def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            lab = d.get("label", parse_label(d.get("output", "")))
            rows.append({"instruction": d["instruction"],
                         "input": d["input"], "output": d.get("output", ""),
                         "label": int(lab)})
    return rows


def format_chat(tok, instruction, user):
    msgs = [{"role": "system", "content": instruction},
            {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True)


@torch.no_grad()
def generate_predictions(model, tok, rows, max_length, max_new=24, batch=4):
    model.eval()
    preds, raws = [], []
    device = next(model.parameters()).device
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        prompts = [format_chat(tok, r["instruction"], r["input"]) for r in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  truncation=True, max_length=max_length - max_new).to(device)
        out = model.generate(**enc, max_new_tokens=max_new,
                             do_sample=False, num_beams=1,
                             pad_token_id=tok.pad_token_id,
                             eos_token_id=tok.eos_token_id)
        for j, o in enumerate(out):
            gen = tok.decode(o[enc["input_ids"].shape[1]:],
                             skip_special_tokens=True)
            raws.append(gen)
            preds.append(parse_label(gen))
    return preds, raws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--test_jsonl", required=True)
    ap.add_argument("--metrics_out", required=True)
    ap.add_argument("--predictions_out", required=True)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True,
                                        use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.config.use_cache = True

    rows = load_rows(args.test_jsonl)
    preds, raws = generate_predictions(model, tok, rows, args.max_length,
                                       batch=args.batch)
    labels = [r["label"] for r in rows]
    invalid = sum(1 for p in preds if p == -1)
    preds_clean = [p if p != -1 else 1 for p in preds]
    acc = accuracy_score(labels, preds_clean)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds_clean, labels=[0, 1], average=None, zero_division=0
    )
    pm, rm, f1m, _ = precision_recall_fscore_support(
        labels, preds_clean, labels=[0, 1], average="macro", zero_division=0
    )
    cm = confusion_matrix(labels, preds_clean, labels=[0, 1]).tolist()
    report = classification_report(labels, preds_clean, target_names=LABELS,
                                   zero_division=0)

    metrics = {
        "base_model": args.base_model, "adapter": args.adapter,
        "test_jsonl": args.test_jsonl,
        "total": len(labels), "valid_predictions": len(labels) - invalid,
        "invalid_predictions": invalid,
        "accuracy": float(acc),
        "not_harmful_precision": float(p[0]), "not_harmful_recall": float(r[0]),
        "not_harmful_f1": float(f1[0]),
        "harmful_precision": float(p[1]), "harmful_recall": float(r[1]),
        "harmful_f1": float(f1[1]),
        "macro_precision": float(pm), "macro_recall": float(rm),
        "macro_f1": float(f1m),
        "confusion_matrix_labels": LABELS, "confusion_matrix": cm,
        "classification_report": report,
    }
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_out).write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.predictions_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.predictions_out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "input", "label", "pred", "raw"])
        for i, (row, l, pv, raw) in enumerate(zip(rows, labels, preds_clean, raws)):
            w.writerow([i, row["input"][:500], LABELS[int(l)],
                        LABELS[int(pv)], raw[:200]])
    print("DONE", json.dumps({
        "adapter": args.adapter, "test": args.test_jsonl,
        "accuracy": float(acc), "macro_f1": float(f1m),
        "harmful_f1": float(f1[1]), "invalid": invalid,
    }))


if __name__ == "__main__":
    main()
