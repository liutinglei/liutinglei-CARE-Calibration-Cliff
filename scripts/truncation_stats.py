"""Compute per-window truncation statistics for the strict test split.

For each tokenizer (RoBERTa-base, InternLM2.5-7B-Chat) and each
context window $w$, we report the share of test utterances whose
input would be truncated at the model's training-time
``max_length``. We also count the share of inputs where left-side
truncation would discard part of the target utterance (which is the
failure mode that drove our ``truncation\_side='left'`` choice for
$w{=}8$).

Run on the remote machine because the InternLM tokenizer is large.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

DATA_DIR = Path("<DATA_ROOT>/data")
WINDOWS = (0, 1, 2, 4, 8)
SPLIT = "test"


def load_inputs(window: int) -> list[str]:
    path = DATA_DIR / f"synbullying_strict_w{window}_{SPLIT}.jsonl"
    out: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            out.append(row["input"])
    return out


def run(name: str, hf_path: str, default_ml: int, w8_ml: int | None = None) -> dict[str, object]:
    tok = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
    per_window: dict[str, object] = {"tokenizer": name, "hf_path": hf_path}
    for w in WINDOWS:
        inputs = load_inputs(w)
        ml = w8_ml if (w == 8 and w8_ml is not None) else default_ml
        ids = tok(inputs, add_special_tokens=True, padding=False,
                  truncation=False)["input_ids"]
        lens = [len(x) for x in ids]
        trunc = sum(1 for L in lens if L > ml)
        # Last-utterance preservation diagnostic for the RoBERTa
        # left-truncation regime (only meaningful when w > 0):
        target_only = [x.split("\n")[-1] if "\n" in x else x for x in inputs]
        tgt_ids = tok(target_only, add_special_tokens=False)["input_ids"]
        tgt_lens = [len(x) for x in tgt_ids]
        median_tok = sorted(lens)[len(lens) // 2]
        p95_tok = sorted(lens)[int(0.95 * len(lens))]
        per_window[f"w{w}"] = {
            "n": len(inputs),
            "max_length": ml,
            "median_tokens": median_tok,
            "p95_tokens": p95_tok,
            "max_tokens": max(lens),
            "truncated_rows": trunc,
            "truncation_rate": round(trunc / len(inputs), 4),
            "target_median_tokens": sorted(tgt_lens)[len(tgt_lens) // 2],
        }
    return per_window


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="<DATA_ROOT>/outputs/truncation_stats.json")
    args = ap.parse_args()

    summary: dict[str, object] = {}
    summary["roberta_base"] = run(
        "roberta-base", "roberta-base", default_ml=256, w8_ml=512,
    )
    summary["internlm2_5_7b"] = run(
        "internlm2_5-7b-chat",
        "<WORK_ROOT>/models/internlm2_5-7b-chat",
        default_ml=512,
        w8_ml=512,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
