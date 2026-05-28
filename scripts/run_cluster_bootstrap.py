"""Run the full suite of conversation-level cluster bootstrap pairs that
mirror the original ``outputs/bootstrap/*.json`` set, and persist them to
``outputs/bootstrap_cluster/``.

This script is intentionally a thin wrapper over
``bootstrap_significance_cluster.py`` so the underlying statistics are
identical to the per-pair invocation; it just enumerates the 9 contrasts
the paper actually reports.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
SCRIPT = PAPER / "scripts" / "bootstrap_significance_cluster.py"
CONV_IDS = PAPER / "outputs" / "test_conv_ids.json"
OUT_DIR = PAPER / "outputs" / "bootstrap_cluster"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENC = PAPER / "outputs" / "baselines"
LLM = PAPER / "outputs" / "llm"

PAIRS: list[dict[str, str]] = [
    {
        "name": "roberta_vs_bert_base_w4",
        "dir_a": str(ENC), "prefix_a": "roberta_base_w4",
        "dir_b": str(ENC), "prefix_b": "bert_base_w4",
    },
    {
        "name": "roberta_vs_deberta_v3_w4",
        "dir_a": str(ENC), "prefix_a": "roberta_base_w4",
        "dir_b": str(ENC), "prefix_b": "deberta_v3_w4",
    },
    {
        "name": "roberta_vs_hatebert_w4",
        "dir_a": str(ENC), "prefix_a": "roberta_base_w4",
        "dir_b": str(ENC), "prefix_b": "hatebert_w4",
    },
    {
        "name": "roberta_vs_internlm2_5_7b_w4",
        "dir_a": str(ENC), "prefix_a": "roberta_base_w4",
        "dir_b": str(LLM), "prefix_b": "internlm2_5_7b_w4",
    },
    {
        "name": "roberta_vs_qwen2_5_7b_w4",
        "dir_a": str(ENC), "prefix_a": "roberta_base_w4",
        "dir_b": str(LLM), "prefix_b": "qwen2_5_7b_w4",
    },
    {
        "name": "internlm_vs_qwen_w4",
        "dir_a": str(LLM), "prefix_a": "internlm2_5_7b_w4",
        "dir_b": str(LLM), "prefix_b": "qwen2_5_7b_w4",
    },
    {
        "name": "roberta_w4_vs_w0",
        "dir_a": str(ENC), "prefix_a": "roberta_base_w4",
        "dir_b": str(ENC), "prefix_b": "roberta_base_w0",
    },
    {
        "name": "roberta_w4_vs_w2",
        "dir_a": str(ENC), "prefix_a": "roberta_base_w4",
        "dir_b": str(ENC), "prefix_b": "roberta_base_w2",
    },
    {
        "name": "roberta_w8_vs_w4",
        "dir_a": str(ENC), "prefix_a": "roberta_base_w8_ml512",
        "dir_b": str(ENC), "prefix_b": "roberta_base_w4",
    },
]


def main() -> int:
    failures: list[str] = []
    for pair in PAIRS:
        out_path = OUT_DIR / f"{pair['name']}.json"
        cmd = [
            sys.executable,
            str(SCRIPT),
            "--pred_dir_a", pair["dir_a"],
            "--prefix_a", pair["prefix_a"],
            "--pred_dir_b", pair["dir_b"],
            "--prefix_b", pair["prefix_b"],
            "--conv_ids", str(CONV_IDS),
            "--B", "10000",
            "--out", str(out_path),
        ]
        print(f">>> {pair['name']}")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stdout[-1000:])
            print(res.stderr[-1000:])
            failures.append(pair["name"])
            continue
        print(res.stdout.splitlines()[-1])
    if failures:
        print(f"\nFAILED: {failures}", file=sys.stderr)
        return 1
    print("\nALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
