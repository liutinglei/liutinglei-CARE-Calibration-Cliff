#!/usr/bin/env python
"""LoRA fine-tuning for decoder LLMs on SynBullying strict splits.

Designed for InternLM2.5-7B-Chat and Qwen2.5-7B-Instruct. Output metrics format
matches the encoder baseline script so they can be aggregated together.
"""
import argparse
import csv
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, DataCollatorForLanguageModeling,
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
    return -1


def load_split(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            rows.append({
                "instruction": d["instruction"],
                "input": d["input"],
                "output": d["output"],
                "label": parse_label(d["output"]),
            })
    return rows


def format_chat(tokenizer, instruction, user_input, assistant=None):
    msgs = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_input},
    ]
    if assistant is not None:
        msgs.append({"role": "assistant", "content": assistant})
    return tokenizer.apply_chat_template(
        msgs, tokenize=False,
        add_generation_prompt=(assistant is None),
    )


def build_train_dataset(rows, tokenizer, max_length):
    def encode(row):
        prompt = format_chat(tokenizer, row["instruction"], row["input"])
        full = format_chat(tokenizer, row["instruction"], row["input"], row["output"])
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
        if not full_ids or full_ids[-1] != tokenizer.eos_token_id:
            full_ids = full_ids + [tokenizer.eos_token_id]
        if len(full_ids) > max_length:
            full_ids = full_ids[:max_length]
        labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids):]
        labels = labels[:len(full_ids)]
        return {"input_ids": full_ids, "labels": labels,
                "attention_mask": [1] * len(full_ids)}
    return Dataset.from_list([encode(r) for r in rows])


def collator_lm(features, pad_token_id):
    max_len = max(len(f["input_ids"]) for f in features)
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for f in features:
        pad = max_len - len(f["input_ids"])
        out["input_ids"].append(f["input_ids"] + [pad_token_id] * pad)
        out["attention_mask"].append(f["attention_mask"] + [0] * pad)
        out["labels"].append(f["labels"] + [-100] * pad)
    return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}


@torch.no_grad()
def generate_predictions(model, tokenizer, rows, max_length, max_new=40, batch_size=8):
    model.eval()
    preds = []
    raws = []
    device = next(model.parameters()).device
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        prompts = [format_chat(tokenizer, r["instruction"], r["input"]) for r in chunk]
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length - max_new).to(device)
        out = model.generate(
            **enc, max_new_tokens=max_new,
            do_sample=False, num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        for j, o in enumerate(out):
            gen = tokenizer.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            raws.append(gen)
            preds.append(parse_label(gen))
    return preds, raws


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
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--train_bsz", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--eval_bsz", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", default="auto",
                    help="Comma-separated, or 'auto' for per-architecture defaults.")
    ap.add_argument("--quant_4bit", action="store_true",
                    help="Load base model in 4-bit (QLoRA).")
    args = ap.parse_args()

    set_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    train_rows = load_split(f"{args.data_dir}/synbullying_strict_w{args.window}_train.jsonl")
    dev_rows = load_split(f"{args.data_dir}/synbullying_strict_w{args.window}_dev.jsonl")
    test_rows = load_split(f"{args.data_dir}/synbullying_strict_w{args.window}_test.jsonl")
    print(f"[data] train={len(train_rows)} dev={len(dev_rows)} test={len(test_rows)}")

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    quant_cfg = None
    if args.quant_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        quantization_config=quant_cfg,
        device_map="auto",
    )
    if args.quant_4bit:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    # Gradient-checkpointing + LoRA requires re-enabling input grad after
    # PEFT wrapping; otherwise the loss has no grad_fn. We enable it on the
    # base model now and PEFT will preserve it.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    if args.target_modules != "auto":
        target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    else:
        name_lower = args.model_name.lower()
        if "internlm2" in name_lower:
            target_modules = ["w1", "w2", "w3", "wo", "wqkv"]
        else:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"]

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=target_modules, bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_ds = build_train_dataset(train_rows, tok, args.max_length)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_bsz,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.0,
        warmup_ratio=0.03,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=1,
        seed=args.seed,
        data_seed=args.seed,
        report_to="none",
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=train_ds,
        data_collator=lambda f: collator_lm(f, tok.pad_token_id),
    )
    trainer.train()
    # Save the adapter to the top-level output_dir so it survives the
    # checkpoint-* cleanup performed by the runner.
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)

    # Predict on test
    preds, raws = generate_predictions(
        model, tok, test_rows, args.max_length, max_new=24, batch_size=args.eval_bsz,
    )
    labels = [r["label"] for r in test_rows]
    invalid = sum(1 for p in preds if p == -1)
    # Treat invalid predictions as 1 (harmful) to be consistent with original eval scripts
    preds_clean = [p if p != -1 else 1 for p in preds]
    acc = accuracy_score(labels, preds_clean)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds_clean, labels=[0, 1], average=None, zero_division=0
    )
    pm, rm, f1m, _ = precision_recall_fscore_support(
        labels, preds_clean, labels=[0, 1], average="macro", zero_division=0
    )
    cm = confusion_matrix(labels, preds_clean, labels=[0, 1]).tolist()
    report = classification_report(labels, preds_clean, target_names=LABELS, zero_division=0)

    metrics = {
        "model": f"{args.model_name} + LoRA strict w{args.window} seed={args.seed}",
        "total": int(len(labels)),
        "valid_predictions": int(len(labels) - invalid),
        "invalid_predictions": invalid,
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
        "config": vars(args),
    }
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_out, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)

    Path(args.predictions_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.predictions_out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "input", "label", "pred", "raw"])
        for i, (row, l, pv, raw) in enumerate(zip(test_rows, labels, preds_clean, raws)):
            w.writerow([i, row["input"][:500], LABELS[int(l)], LABELS[int(pv)], raw[:200]])

    print("DONE", json.dumps({
        "model": args.model_name, "seed": args.seed, "window": args.window,
        "accuracy": float(acc), "macro_f1": float(f1m),
        "harmful_f1": float(f1[1]), "invalid": invalid,
    }))


if __name__ == "__main__":
    main()
