"""Aggregate the GroupKFold 5-fold metrics into a JSON summary and a
LaTeX table fragment.

Reads ``paper/outputs/groupkfold/<model>_fold{0..4}_metrics.json``
and emits ``paper/outputs/groupkfold.json`` plus
``paper/outputs/groupkfold_table.tex``.
"""
from __future__ import annotations

import json
import statistics as stats
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "groupkfold"
JSON_OUT = ROOT / "outputs" / "groupkfold.json"
TEX_OUT = ROOT / "outputs" / "groupkfold_table.tex"

MODELS = [
    ("RoBERTa-base", "roberta_base"),
    ("InternLM2.5-7B + LoRA", "internlm2_5_7b"),
]
N_FOLDS = 5


def gather(short: str) -> dict[str, object]:
    fold_metrics: list[dict[str, float]] = []
    for k in range(N_FOLDS):
        p = SRC / f"{short}_fold{k}_metrics.json"
        if not p.is_file():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        fold_metrics.append(
            {
                "fold": k,
                "macro_f1": float(d.get("macro_f1", 0.0)),
                "harm_f1": float(d.get("harmful_f1", 0.0)),
                "accuracy": float(d.get("accuracy", 0.0)),
                "n_test": int(d.get("total", 0)),
            }
        )
    if not fold_metrics:
        return {"n_folds": 0}
    macros = [m["macro_f1"] for m in fold_metrics]
    harms = [m["harm_f1"] for m in fold_metrics]
    return {
        "n_folds": len(fold_metrics),
        "folds": fold_metrics,
        "macro_f1_mean": float(stats.fmean(macros)),
        "macro_f1_sd": float(stats.pstdev(macros)) if len(macros) > 1 else 0.0,
        "macro_f1_min": float(min(macros)),
        "macro_f1_max": float(max(macros)),
        "harm_f1_mean": float(stats.fmean(harms)),
        "harm_f1_sd": float(stats.pstdev(harms)) if len(harms) > 1 else 0.0,
    }


def main() -> int:
    summary: dict[str, object] = {}
    for label, short in MODELS:
        summary[short] = gather(short)
    JSON_OUT.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {JSON_OUT}")

    rows: list[str] = []
    rows.append("\\begin{tabular}{lcccc}")
    rows.append("\\toprule")
    rows.append("Model & Folds & Macro-F1 (s.d.) & Min--Max & Harm-F1 (s.d.) \\\\")
    rows.append("\\midrule")
    for label, short in MODELS:
        d = summary[short]
        if d.get("n_folds", 0) == 0:
            rows.append(f"{label} & 0/{N_FOLDS} & -- & -- & -- \\\\")
            continue
        rows.append(
            f"{label} & {d['n_folds']}/{N_FOLDS} & "
            f"{d['macro_f1_mean']:.3f} ({d['macro_f1_sd']:.3f}) & "
            f"{d['macro_f1_min']:.3f}--{d['macro_f1_max']:.3f} & "
            f"{d['harm_f1_mean']:.3f} ({d['harm_f1_sd']:.3f}) \\\\"
        )
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    TEX_OUT.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote {TEX_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
