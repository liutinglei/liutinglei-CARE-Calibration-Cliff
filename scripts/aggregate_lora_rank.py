"""Aggregate the LoRA rank ablation results (r in {4, 8, 16, 32}) for the
two LoRA-adapted 7B LLMs into one JSON + one LaTeX table fragment.

Reads:
  paper/outputs/llm/{internlm2_5_7b,qwen2_5_7b}_w4_seed42_metrics.json
    (the default r=8 baseline that already exists for both models)
  paper/outputs/llm/{internlm2_5_7b,qwen2_5_7b}_rank{4,16,32}_w4_seed42_metrics.json
    (created by scripts/run_lora_rank_ablation.sh on the remote)

Writes:
  paper/outputs/lora_rank_table.tex
  paper/outputs/lora_rank_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LLM_DIR = ROOT / "outputs" / "llm"
OUT_DIR = ROOT / "outputs"

MODELS: list[tuple[str, str]] = [
    ("InternLM2.5-7B + LoRA", "internlm2_5_7b"),
    ("Qwen2.5-7B + LoRA", "qwen2_5_7b"),
]
RANKS: list[int] = [4, 8, 16, 32]


def metrics_path(short: str, r: int) -> Path:
    if r == 8:
        return LLM_DIR / f"{short}_w4_seed42_metrics.json"
    return LLM_DIR / f"{short}_rank{r}_w4_seed42_metrics.json"


def fmt_cell(metrics: dict[str, float] | None, is_default: bool) -> str:
    if not metrics:
        return "--"
    macro = metrics["macro_f1"]
    label = f"{macro:.3f}"
    if is_default:
        label += "\\,$^{\\dagger}$"
    return label


def main() -> int:
    summary: dict[str, dict[int, dict[str, float]]] = {}
    rows: list[str] = []
    rows.append("\\begin{tabular}{lcccc}")
    rows.append("\\toprule")
    rows.append(
        " & $r{=}4$ & $r{=}8$ & $r{=}16$ & $r{=}32$ \\\\"
    )
    rows.append("\\midrule")

    for label, short in MODELS:
        cells: list[str] = []
        per_rank: dict[int, dict[str, float]] = {}
        for r in RANKS:
            p = metrics_path(short, r)
            if not p.is_file():
                per_rank[r] = {}
                cells.append("--")
                continue
            d = json.loads(p.read_text(encoding="utf-8"))
            per_rank[r] = {
                "macro_f1": float(d["macro_f1"]),
                "harmful_f1": float(d["harmful_f1"]),
                "accuracy": float(d["accuracy"]),
            }
            cells.append(fmt_cell(per_rank[r], is_default=(r == 8)))
        summary[short] = per_rank
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")

    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")

    (OUT_DIR / "lora_rank_table.tex").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )
    (OUT_DIR / "lora_rank_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {OUT_DIR / 'lora_rank_table.tex'}")
    print(f"wrote {OUT_DIR / 'lora_rank_summary.json'}")

    # Print quick-scan ranges
    for short, per_rank in summary.items():
        macros = [v["macro_f1"] for v in per_rank.values() if v]
        if not macros:
            continue
        spread = max(macros) - min(macros)
        print(f"{short}: ranks={list(per_rank.keys())}  "
              f"macro_f1 range [{min(macros):.3f}, {max(macros):.3f}]  "
              f"spread={spread:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
