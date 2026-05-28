"""Per-strategy inference comparison for a LoRA-adapted LLM.

We run the same model + adapter under three decoding strategies and
report Macro-F1, Harm-F1, accuracy, and the unparsable rate for each.
The goal is to show that the in-domain headline numbers are not an
artefact of free-form parsing.

Strategies::

  free_form   : current default; generate up to 32 tokens, parse the
                first content word as the label, fall back to HARMFUL on
                unparsable output (this matches train_llm_lora.py).
  constrained : restrict the next-token distribution to the two
                verbalizer tokens (``harmful`` / ``not``) using a
                prefix_allowed_tokens_fn; no parsing failure is
                possible by construction.
  verbalizer  : do one forward pass on the prompt and read off the
                logits at the next-token position for the two
                verbalizer tokens; pick the larger as the prediction.

Usage::

    python eval_llm_strategies.py \\
        --base_model <WORK_ROOT>/models/internlm2_5-7b-chat \\
        --adapter   <CKPT_ROOT>/llm/internlm2_5_7b_w4_seed42 \\
        --test_jsonl <DATA_ROOT>/data/synbullying_strict_w4_test.jsonl \\
        --out_dir   <DATA_ROOT>/outputs/inference_strategy

Writes ``{strategy}_metrics.json`` and ``{strategy}_predictions.csv`` in
``out_dir``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

LABELS = ["not_harmful", "harmful"]


def parse_label_free(text: str) -> int:
    s = text.strip().lower()
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
            rows.append(d)
    return rows


def gold_label(d: dict[str, object]) -> int:
    if "label" in d:
        return int(d["label"])  # type: ignore[arg-type]
    return parse_label_free(str(d.get("output", "")))


def build_prompt(tokenizer: AutoTokenizer, instr: str, user: str) -> str:
    msgs = [
        {"role": "system", "content": instr},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(  # type: ignore[no-any-return]
        msgs, tokenize=False, add_generation_prompt=True
    )


def find_verbalizer_ids(tokenizer: AutoTokenizer) -> tuple[list[int], list[int]]:
    """Get the token-id sets that begin the verbalizer surface forms."""
    harm_strings = [" harmful", "harmful"]
    not_strings = [" not", "not"]
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
    return list(set(harm_ids)), list(set(not_ids))


def macro_f1(labels: list[int], preds: list[int]) -> tuple[float, float, float]:
    _, _, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], average=None, zero_division=0
    )
    return float(f1.mean()), float(f1[0]), float(f1[1])


def run(args: argparse.Namespace) -> int:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    rows = load_rows(args.test_jsonl)
    gold = [gold_label(r) for r in rows]

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        quantization_config=bnb,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    harm_ids, not_ids = find_verbalizer_ids(tokenizer)
    allowed_ids = list(set(harm_ids + not_ids))
    print(f"verbalizer harm ids={harm_ids}  not ids={not_ids}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    strategies = {
        "free_form": run_free_form,
        "constrained": run_constrained,
        "verbalizer": run_verbalizer,
    }

    summary: dict[str, object] = {
        "base_model": args.base_model,
        "adapter": args.adapter,
        "test_jsonl": args.test_jsonl,
        "n": len(rows),
    }

    for name, fn in strategies.items():
        if args.strategies and name not in set(args.strategies.split(",")):
            continue
        t0 = time.time()
        preds, raw_outputs, unparsable_n = fn(
            model, tokenizer, rows, harm_ids, not_ids, allowed_ids, args
        )
        elapsed = time.time() - t0
        m_f1, nh_f1, h_f1 = macro_f1(gold, preds)
        acc = float(accuracy_score(gold, preds))
        cm = confusion_matrix(gold, preds, labels=[0, 1]).tolist()
        metrics = {
            "strategy": name,
            "n": len(rows),
            "unparsable_rows": unparsable_n,
            "unparsable_rate": round(unparsable_n / len(rows), 4),
            "macro_f1": m_f1,
            "harmful_f1": h_f1,
            "not_harmful_f1": nh_f1,
            "accuracy": acc,
            "confusion_matrix": cm,
            "wallclock_s": round(elapsed, 2),
        }
        (out_dir / f"{name}_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        with (out_dir / f"{name}_predictions.csv").open(
            "w", encoding="utf-8", newline=""
        ) as fh:
            w = csv.writer(fh)
            w.writerow(["idx", "label", "pred", "raw"])
            for i, (g, p, raw) in enumerate(zip(gold, preds, raw_outputs)):
                w.writerow(
                    [
                        i,
                        LABELS[g] if g >= 0 else "unknown",
                        LABELS[p] if p >= 0 else "unparsable",
                        raw,
                    ]
                )
        summary[name] = metrics
        print(json.dumps(metrics, indent=2))

    (out_dir / "strategy_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return 0


@torch.no_grad()
def run_free_form(
    model, tokenizer, rows, harm_ids, not_ids, allowed_ids, args
):
    preds: list[int] = []
    raws: list[str] = []
    unparsable = 0
    for r in rows:
        prompt = build_prompt(tokenizer, r["instruction"], r["input"])
        ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **ids, max_new_tokens=32, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen = tokenizer.decode(
            out[0, ids.input_ids.shape[1]:], skip_special_tokens=True
        )
        lab = parse_label_free(gen)
        if lab == -1:
            unparsable += 1
            lab = 1  # match training-time fallback to HARMFUL
        preds.append(lab)
        raws.append(gen)
    return preds, raws, unparsable


@torch.no_grad()
def run_constrained(
    model, tokenizer, rows, harm_ids, not_ids, allowed_ids, args
):
    preds: list[int] = []
    raws: list[str] = []
    unparsable = 0

    def allowed_first(batch_id: int, prefix_ids: torch.Tensor) -> list[int]:
        return allowed_ids

    for r in rows:
        prompt = build_prompt(tokenizer, r["instruction"], r["input"])
        ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **ids,
            max_new_tokens=1,
            do_sample=False,
            prefix_allowed_tokens_fn=lambda bi, prefix: allowed_first(bi, prefix),
            pad_token_id=tokenizer.pad_token_id,
        )
        first_id = int(out[0, ids.input_ids.shape[1]])
        lab = 1 if first_id in harm_ids else 0
        preds.append(lab)
        raws.append(tokenizer.decode([first_id], skip_special_tokens=True))
    return preds, raws, unparsable


@torch.no_grad()
def run_verbalizer(
    model, tokenizer, rows, harm_ids, not_ids, allowed_ids, args
):
    preds: list[int] = []
    raws: list[str] = []
    unparsable = 0
    for r in rows:
        prompt = build_prompt(tokenizer, r["instruction"], r["input"])
        ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        logits = model(**ids).logits[0, -1]
        harm_logit = float(logits[harm_ids].max().item())
        not_logit = float(logits[not_ids].max().item())
        lab = 1 if harm_logit > not_logit else 0
        preds.append(lab)
        raws.append(f"harm={harm_logit:.3f} not={not_logit:.3f}")
    return preds, raws, unparsable


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--test_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--strategies",
        default="",
        help="Optional comma list (free_form,constrained,verbalizer).",
    )
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
