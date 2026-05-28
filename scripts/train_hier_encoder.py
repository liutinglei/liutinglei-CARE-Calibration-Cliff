"""Hierarchical context-aware encoder baseline.

Architecture:
  per-utterance RoBERTa encoder -> [CLS] embedding per utterance
  -> 2-layer Transformer aggregator over the window utterances
  -> linear classification head on the target-utterance CLS

This directly addresses the reviewer concern that our flat-context
encoders may underestimate the value of explicit conversational
structure. Compared to flat-context RoBERTa, the hierarchical
variant encodes each utterance independently and then attends over
their representations -- the canonical "hierarchical text"
inductive bias used by Yang et al. (HAN, 2016), Yi & Zubiaga (2025),
and many cyberbullying papers.

We train at w=4 for direct comparability with Table III of the
paper. Output schema matches train_encoder_baseline.py so it can
be picked up by the existing aggregators.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support)
from torch import nn
from transformers import (AutoConfig, AutoModel, AutoTokenizer,
                          Trainer, TrainingArguments)

LABELS = ["not_harmful", "harmful"]


def set_seed(seed: int) -> None:
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


def split_utterances(text: str, max_n: int) -> list[str]:
    """Split a window string on newlines into individual utterances.
    Each utterance is something like 'Speaker1: hello there'."""
    parts = [p for p in text.split("\n") if p.strip()]
    return parts[-max_n:]


class HierConfig:
    def __init__(self,
                 backbone: str = "roberta-base",
                 max_utt: int = 5,
                 utt_max_len: int = 96,
                 agg_layers: int = 2,
                 agg_heads: int = 8,
                 num_labels: int = 2) -> None:
        self.backbone = backbone
        self.max_utt = max_utt
        self.utt_max_len = utt_max_len
        self.agg_layers = agg_layers
        self.agg_heads = agg_heads
        self.num_labels = num_labels


class HierEncoder(nn.Module):
    def __init__(self, cfg: HierConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = AutoModel.from_pretrained(cfg.backbone)
        d = self.backbone.config.hidden_size
        agg_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.agg_heads, dim_feedforward=4 * d,
            batch_first=True, activation="gelu",
        )
        self.aggregator = nn.TransformerEncoder(agg_layer, num_layers=cfg.agg_layers)
        self.classifier = nn.Linear(d, cfg.num_labels)
        # position embedding for utterance index (recency)
        self.pos = nn.Embedding(cfg.max_utt, d)

    def forward(self,
                input_ids: torch.Tensor,        # (B, U, T)
                attention_mask: torch.Tensor,   # (B, U, T)
                utt_mask: torch.Tensor,         # (B, U) -- 1 if utterance present
                labels: torch.Tensor | None = None) -> dict:
        B, U, T = input_ids.shape
        flat_ids = input_ids.reshape(B * U, T)
        flat_mask = attention_mask.reshape(B * U, T)
        # mask out padding utterances by setting attention_mask all zeros
        # the backbone needs at least one token to be 1 to not NaN
        non_empty = (flat_mask.sum(-1) > 0).clamp(max=1)
        safe_mask = flat_mask.clone()
        safe_mask[non_empty == 0, 0] = 1  # force at least one 1 to avoid NaN
        out = self.backbone(input_ids=flat_ids, attention_mask=safe_mask)
        cls = out.last_hidden_state[:, 0, :].view(B, U, -1)  # (B, U, d)
        # add positional embedding by utterance index (0 = oldest)
        pos_idx = torch.arange(U, device=cls.device).expand(B, U)
        cls = cls + self.pos(pos_idx)
        # aggregator: mask out padding utterances
        agg_mask = utt_mask == 0  # (B, U) -- True = mask
        agg = self.aggregator(cls, src_key_padding_mask=agg_mask)  # (B, U, d)
        # take the LAST non-padded utterance representation as the
        # target utterance's contextualised CLS
        last_idx = utt_mask.sum(-1).long() - 1
        last_idx = last_idx.clamp(min=0)
        target_cls = agg[torch.arange(B), last_idx]  # (B, d)
        logits = self.classifier(target_cls)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}


def encode_example(row: dict, tokenizer: AutoTokenizer,
                   cfg: HierConfig) -> dict:
    utts = split_utterances(row["text"], cfg.max_utt)
    if not utts:
        utts = [""]
    enc = tokenizer(
        utts,
        padding="max_length",
        truncation=True,
        max_length=cfg.utt_max_len,
        return_tensors="np",
    )
    ids = enc["input_ids"]                       # (n_utt, T)
    mask = enc["attention_mask"]                 # (n_utt, T)
    n = ids.shape[0]
    pad_n = cfg.max_utt - n
    if pad_n > 0:
        pad_ids = np.zeros((pad_n, cfg.utt_max_len), dtype=ids.dtype)
        pad_mask = np.zeros((pad_n, cfg.utt_max_len), dtype=mask.dtype)
        # We want padding utterances at the BEGINNING so the last
        # axis index always points at the target utterance.
        ids = np.concatenate([pad_ids, ids], axis=0)
        mask = np.concatenate([pad_mask, mask], axis=0)
    utt_present = np.concatenate(
        [np.zeros(pad_n, dtype=np.int64),
         np.ones(n, dtype=np.int64)], axis=0
    )
    return {
        "input_ids": ids.tolist(),
        "attention_mask": mask.tolist(),
        "utt_mask": utt_present.tolist(),
        "label": int(row["label"]),
    }


def load_split(path: str) -> Dataset:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            rows.append({"text": d["input"],
                          "label": parse_label(d["output"])})
    return Dataset.from_list(rows)


def compute_metrics(pred) -> dict:
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    _, _, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], average=None, zero_division=0
    )
    macro = float(f1.mean())
    return {"accuracy": acc, "macro_f1": macro,
            "not_harmful_f1": float(f1[0]), "harmful_f1": float(f1[1])}


def collate(batch: list[dict]) -> dict:
    ids = torch.tensor([x["input_ids"] for x in batch], dtype=torch.long)
    mask = torch.tensor([x["attention_mask"] for x in batch], dtype=torch.long)
    utt_mask = torch.tensor([x["utt_mask"] for x in batch], dtype=torch.long)
    labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
    return {"input_ids": ids, "attention_mask": mask,
            "utt_mask": utt_mask, "labels": labels}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="roberta-base")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--metrics_out", required=True)
    ap.add_argument("--predictions_out", required=True)
    ap.add_argument("--epochs", type=float, default=5.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_utt", type=int, default=5)
    ap.add_argument("--utt_max_len", type=int, default=96)
    args = ap.parse_args()

    set_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    cfg = HierConfig(
        backbone=args.backbone,
        max_utt=args.max_utt,
        utt_max_len=args.utt_max_len,
    )
    tok = AutoTokenizer.from_pretrained(args.backbone)
    model = HierEncoder(cfg).cuda()

    train_p = f"{args.data_dir}/synbullying_strict_w{args.window}_train.jsonl"
    dev_p = f"{args.data_dir}/synbullying_strict_w{args.window}_dev.jsonl"
    test_p = f"{args.data_dir}/synbullying_strict_w{args.window}_test.jsonl"
    train_ds = load_split(train_p).map(
        lambda r: encode_example(r, tok, cfg), remove_columns=["text"]
    )
    dev_ds = load_split(dev_p).map(
        lambda r: encode_example(r, tok, cfg), remove_columns=["text"]
    )
    test_ds = load_split(test_p).map(
        lambda r: encode_example(r, tok, cfg), remove_columns=["text"]
    )

    args_t = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        warmup_ratio=0.06,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=True,
        report_to=[],
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=args_t,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collate,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    preds = trainer.predict(test_ds)
    metrics = compute_metrics(preds)
    metrics["seed"] = args.seed
    metrics["window"] = args.window
    metrics["backbone"] = args.backbone
    metrics["total"] = len(test_ds)
    metrics["valid_predictions"] = len(test_ds)
    metrics["invalid_predictions"] = 0
    metrics["confusion_matrix_labels"] = LABELS
    cm = [[0, 0], [0, 0]]
    yhat = preds.predictions.argmax(-1)
    ytrue = preds.label_ids
    for g, p in zip(ytrue, yhat):
        cm[int(g)][int(p)] += 1
    metrics["confusion_matrix"] = cm
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2),
                                       encoding="utf-8")

    with open(args.predictions_out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "text", "label", "pred"])
        for i in range(len(test_ds)):
            w.writerow([
                i,
                "",  # text omitted to keep file small
                LABELS[int(ytrue[i])],
                LABELS[int(yhat[i])],
            ])
    print(f"DONE {json.dumps(metrics)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
