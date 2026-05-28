"""Aggregate the hierarchical-encoder baseline results across the
three seeds and emit a one-row LaTeX fragment that drops into the
main results table (Table III)."""
from __future__ import annotations

import json
import statistics as stats
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "hier"
TEX = ROOT / "outputs" / "hier_row.tex"


def main() -> int:
    seeds = (41, 42, 43)
    rows: list[dict[str, float]] = []
    for s in seeds:
        path = SRC / f"hier_roberta_w4_seed{s}_metrics.json"
        if not path.is_file():
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "macro_f1": float(d.get("macro_f1", 0.0)),
                "harm_f1": float(d.get("harmful_f1", 0.0)),
                "not_harm_f1": float(d.get("not_harmful_f1", 0.0)),
                "accuracy": float(d.get("accuracy", 0.0)),
            }
        )
    if not rows:
        TEX.write_text(
            "Hierarchical RoBERTa-base & --- & \\multicolumn{5}{c}{\\emph{pending}} \\\\\n",
            encoding="utf-8",
        )
        print("placeholder written")
        return 0

    def stat(key: str) -> tuple[float, float]:
        vals = [r[key] for r in rows]
        return (
            float(stats.fmean(vals)),
            float(stats.pstdev(vals)) if len(vals) > 1 else 0.0,
        )

    m_macro, sd_macro = stat("macro_f1")
    m_harm, sd_harm = stat("harm_f1")
    # The table also has Harm-P/Harm-R columns; we leave them as "--"
    # because the hier_*_metrics.json file does not expose per-class
    # precision/recall (they could be computed from predictions if
    # needed).
    line = (
        f"Hierarchical RoBERTa-base & 125M & "
        f"{m_macro:.4f} ({sd_macro:.4f}) & "
        f"{m_harm:.4f} ({sd_harm:.4f}) & "
        f"-- & -- & "
        f"{stat('accuracy')[0]:.4f} \\\\"
    )
    TEX.write_text(line + "\n", encoding="utf-8")
    print(f"wrote {TEX} with {len(rows)} seeds: macro={m_macro:.3f} (sd {sd_macro:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
