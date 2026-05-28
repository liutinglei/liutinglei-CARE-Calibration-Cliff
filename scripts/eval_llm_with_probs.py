"""Verbalizer-scoring eval of a LoRA-adapted LLM that exports per-example
P(harmful) probabilities for downstream calibration / ECE / temperature-
scaling analysis.

Why a separate script. ``eval_llm_on_dataset.py`` performs free-form
generation and only persists hard labels; it is the reference path used by
the v9 transfer pipeline and is left untouched to avoid breaking the
reproducibility contract. ``eval_llm_strategies.py`` proved that
verbalizer scoring is within 0.005 Macro-F1 of free-form generation on
the in-domain test set (Sec. V.D of the paper), which authorises us to
take verbalizer-scoring probabilities as a faithful proxy of the model's
underlying decision distribution for calibration purposes.

The script loads the base model in 4-bit NF4 (matching the LoRA training
recipe), applies the adapter, runs one forward pass per prompt, reads the
last-position logits for the verbalizer token-ids, and emits a
``predictions.csv`` whose schema matches the encoder eval:

    idx, input, label, pred, prob_harmful

plus the customary ``metrics.json`` summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import torch
from peft import PeftModel
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix,
                             precision_recall_fscore_support)
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

LABELS = ("not_harmful", "harmful")


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


def load_rows(path: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            lab = d.get("label", parse_label(str(d.get("output", ""))))
            rows.append(
                {
                    "instruction": d["instruction"],
                    "input": d["input"],
                    "label": int(lab),  # type: ignore[arg-type]
                }
            )
    return rows


def find_verbalizer_ids(tokenizer) -> tuple[list[int], list[int]]:
    """Return (harm_ids, not_ids): the first sub-token of each surface
    form we accept as a verbalizer. Multiple surface forms are unioned to
    be robust to leading-space tokenization variants.
    """
    harm_strings = (" harmful", "harmful")
    not_strings = (" not", "not")
    harm_ids: list[int] = []
    not_ids: list[int] = []
    for s in harm_strings:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids:
            harm_ids.append(int(ids[0]))
    for s in not_strings:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids:
            not_ids.append(int(ids[0]))
    return sorted(set(harm_ids)), sorted(set(not_ids))


def build_prompt(tokenizer, instruction: str, user: str) -> str:
    msgs = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def score_batched(
    model,
    tokenizer,
    rows: list[dict[str, object]],
    harm_ids: list[int],
    not_ids: list[int],
    max_length: int,
    batch_size: int,
) -> tuple[list[int], list[float]]:
    """For each row, run one forward pass on the prompt and return
    (predicted label, P(harmful)). P(harmful) is obtained by a 2-way
    softmax over the max-logit of harm_ids and the max-logit of not_ids
    at the next-token position.

    Batched: prompts are left-padded to the longest in their batch so
    that index ``-1`` of every row inside ``logits`` is the genuine
    next-token position for its prompt.
    """
    preds: list[int] = []
    probs: list[float] = []
    device = next(model.parameters()).device
    harm_t = torch.tensor(harm_ids, device=device, dtype=torch.long)
    not_t = torch.tensor(not_ids, device=device, dtype=torch.long)

    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        prompts = [
            build_prompt(
                tokenizer, str(r["instruction"]), str(r["input"])
            )
            for r in chunk
        ]
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        out = model(**enc)
        # left-padding => the final column of every row is the actual
        # next-token logit slot for that prompt.
        last_logits = out.logits[:, -1, :]
        harm_max = last_logits.index_select(-1, harm_t).max(dim=-1).values
        not_max = last_logits.index_select(-1, not_t).max(dim=-1).values
        # 2-way softmax done in fp32 for numeric stability
        h32 = harm_max.float()
        n32 = not_max.float()
        m = torch.maximum(h32, n32)
        p_h = (h32 - m).exp() / ((h32 - m).exp() + (n32 - m).exp())
        p_h_list = p_h.cpu().tolist()
        for ph in p_h_list:
            preds.append(1 if ph >= 0.5 else 0)
            probs.append(float(ph))
    return preds, probs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--test_jsonl", required=True)
    ap.add_argument("--metrics_out", required=True)
    ap.add_argument("--predictions_out", required=True)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of prompts to fold into a single forward.",
    )
    ap.add_argument(
        "--quant",
        choices=("bf16", "4bit"),
        default="4bit",
        help="Model load dtype. 4bit NF4 matches the LoRA training recipe.",
    )
    args = ap.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if args.quant == "4bit":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=bnb,
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    model.config.use_cache = False  # not needed for single forward

    harm_ids, not_ids = find_verbalizer_ids(tokenizer)
    print(f"verbalizer harm ids={harm_ids}  not ids={not_ids}")

    rows = load_rows(args.test_jsonl)
    preds, probs = score_batched(
        model,
        tokenizer,
        rows,
        harm_ids,
        not_ids,
        args.max_length,
        args.batch_size,
    )
    labels = [int(r["label"]) for r in rows]  # type: ignore[arg-type]

    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], average=None, zero_division=0
    )
    pm, rm, f1m, _ = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], average="macro", zero_division=0
    )
    cm = confusion_matrix(labels, preds, labels=[0, 1]).tolist()
    report = classification_report(
        labels, preds, target_names=list(LABELS), zero_division=0
    )

    metrics = {
        "mode": "verbalizer_scoring",
        "base_model": args.base_model,
        "adapter": args.adapter,
        "test_jsonl": args.test_jsonl,
        "total": len(labels),
        "valid_predictions": len(labels),
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
        "confusion_matrix_labels": list(LABELS),
        "confusion_matrix": cm,
        "classification_report": report,
        "verbalizer_harm_ids": harm_ids,
        "verbalizer_not_ids": not_ids,
    }
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_out).write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    Path(args.predictions_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.predictions_out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "input", "label", "pred", "prob_harmful"])
        for i, (row, lab, pv, p_harm) in enumerate(
            zip(rows, labels, preds, probs)
        ):
            text = str(row["input"])[:500]
            w.writerow(
                [
                    i,
                    text,
                    LABELS[int(lab)],
                    LABELS[int(pv)],
                    f"{p_harm:.6f}",
                ]
            )
    print(
        "DONE",
        json.dumps(
            {
                "adapter": args.adapter,
                "test": args.test_jsonl,
                "accuracy": float(acc),
                "macro_f1": float(f1m),
                "harmful_f1": float(f1[1]),
            }
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
