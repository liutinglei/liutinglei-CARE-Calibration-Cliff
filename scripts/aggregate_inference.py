"""Aggregate the 3-seed inference-strategy comparison metrics into the
LaTeX table fragment used by ``Table~\\ref{tab:inference_strategy}``.

Reads
``paper/outputs/inference_strategy/{free_form,constrained,verbalizer}_metrics.json``
for seed 42 and
``paper/outputs/inference_strategy_seed{41,43}/{strategy}_metrics.json``
for seeds 41 and 43.

Each cell is reported as 3-seed mean (s.d.).
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "inference_strategy_table.tex"
SEEDS = (41, 42, 43)

STRATEGIES = [
    ("Free-form (default)", "free_form"),
    ("Constrained decoding", "constrained"),
    ("Verbalizer scoring", "verbalizer"),
]


def metrics_path(strategy: str, seed: int) -> Path:
    if seed == 42:
        return ROOT / "outputs" / "inference_strategy" / f"{strategy}_metrics.json"
    return (
        ROOT / "outputs" / f"inference_strategy_seed{seed}"
        / f"{strategy}_metrics.json"
    )


def cell(values: list[float], digits: int = 3) -> str:
    if not values:
        return "--"
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    mean = statistics.fmean(values)
    sd = statistics.pstdev(values)
    return f"{mean:.{digits}f} ({sd:.{digits}f})"


def main() -> int:
    rows: list[str] = []
    rows.append("\\begin{tabular}{lcccc}")
    rows.append("\\toprule")
    rows.append(
        "Strategy & Macro-F1 & Harm-F1 & Acc. & Unparsable \\\\"
    )
    rows.append("\\midrule")
    any_data = False
    for label, key in STRATEGIES:
        macros: list[float] = []
        harms: list[float] = []
        accs: list[float] = []
        unparses: list[float] = []
        for s in SEEDS:
            p = metrics_path(key, s)
            if not p.is_file():
                continue
            d = json.loads(p.read_text(encoding="utf-8"))
            macros.append(float(d["macro_f1"]))
            harms.append(float(d["harmful_f1"]))
            accs.append(float(d["accuracy"]))
            unparses.append(float(d.get("unparsable_rate", 0.0)))
        if not macros:
            rows.append(f"{label} & -- & -- & -- & -- \\\\")
            continue
        any_data = True
        # unparsable always 0 on this dataset; just take max for display
        max_unp = max(unparses) * 100 if unparses else 0.0
        rows.append(
            f"{label} & "
            f"{cell(macros)} & "
            f"{cell(harms)} & "
            f"{cell(accs)} & "
            f"{max_unp:.1f}\\% \\\\"
        )
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")

    OUT.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote {OUT} (data={'yes' if any_data else 'placeholder'})")

    # also print quick-scan range
    for label, key in STRATEGIES:
        macros = []
        for s in SEEDS:
            p = metrics_path(key, s)
            if p.is_file():
                macros.append(
                    float(json.loads(p.read_text(encoding="utf-8"))["macro_f1"])
                )
        if macros:
            print(f"{label}: n_seeds={len(macros)}, "
                  f"mean_mF1={statistics.fmean(macros):.4f}, "
                  f"sd={statistics.pstdev(macros):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
