"""Compute Expected Calibration Error (ECE) and reliability-diagram
data for every (model, target) pair where prediction CSVs include
soft scores.

Our existing prediction CSVs only carry hard labels (no probabilities).
To produce ECE without re-running expensive evals, we use a *prediction
confidence proxy*: per-class self-consistency across the three random
seeds is treated as a discrete confidence (0/3, 1/3, 2/3, 3/3 agree on
the predicted class). This is the same confidence-proxy used in
Wenzel et al. 2020 for ensemble-based calibration estimation.

For each (target, model):
  * Bin examples by their 3-seed self-consistency
  * In each bin: (1) average confidence (k/3), (2) empirical accuracy
  * ECE = sum_b (|B_b|/n) * |conf_b - acc_b|

We compute ECE for in-domain SynBullying, OLID, and HateXplain.
Output: ``paper/outputs/calibration.json`` + a LaTeX fragment
``paper/outputs/calibration_table.tex``.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"

TARGETS = {
    "synb_w4": (ROOT / "outputs" / "baselines",
                ROOT / "outputs" / "llm",
                "_w4_seed{seed}_predictions.csv"),
    "olid": (ROOT / "outputs" / "olid", ROOT / "outputs" / "olid",
             "_w4_seed{seed}_predictions.csv"),
    "hatexplain": (ROOT / "outputs" / "hatexplain",
                   ROOT / "outputs" / "hatexplain",
                   "_w4_seed{seed}_predictions.csv"),
}

MODELS_ENC = ["bert_base", "roberta_base", "hatebert", "deberta_v3"]
MODELS_LLM = ["internlm2_5_7b", "qwen2_5_7b"]
SEEDS = (41, 42, 43)


def load_predictions(dir_enc: Path, dir_llm: Path, model: str, fmt: str) -> list[list[int]] | None:
    """Return list of 3 per-seed pred vectors, each binary 0/1."""
    direc = dir_llm if model in MODELS_LLM else dir_enc
    seeds_pred: list[list[int]] = []
    for s in SEEDS:
        path = direc / f"{model}{fmt.format(seed=s)}"
        if not path.is_file():
            return None
        df = pd.read_csv(path)
        # Detect label column (some files use 'label', some 'pred')
        pred_col = "pred" if "pred" in df.columns else None
        label_col = "label" if "label" in df.columns else None
        if pred_col is None or label_col is None:
            return None
        m = {"harmful": 1, "not_harmful": 0, "not harmful": 0}
        preds = df[pred_col].astype(str).str.strip().str.lower().str.replace(" ", "_").map(m)
        seeds_pred.append(preds.tolist())
    # Also stash gold from first seed
    df = pd.read_csv(direc / f"{model}{fmt.format(seed=SEEDS[0])}")
    gold = df["label"].astype(str).str.strip().str.lower().str.replace(" ", "_").map(
        {"harmful": 1, "not_harmful": 0, "not harmful": 0}
    ).tolist()
    return [seeds_pred, gold]  # type: ignore[return-value]


def compute_ece(seed_preds: list[list[int]], gold: list[int]) -> dict[str, float]:
    n = len(gold)
    if n == 0 or len(seed_preds) < 2:
        return {"n": 0, "ece": 0.0, "bins": {}}
    # Self-consistency: number of seeds predicting class 1
    counts1 = [sum(seed_preds[s][i] for s in range(len(seed_preds))) for i in range(n)]
    # Confidence for prediction: max(k/S, 1 - k/S)
    S = len(seed_preds)
    bins: dict[str, list[tuple[float, int]]] = defaultdict(list)
    total_ece = 0.0
    bin_summary: dict[str, dict[str, float]] = {}
    for i, k in enumerate(counts1):
        pred = 1 if k > S / 2 else 0
        conf = max(k, S - k) / S  # 0.5..1.0
        bins[f"{conf:.3f}"].append((conf, int(gold[i] == pred)))
    for key, items in bins.items():
        confs = [c for c, _ in items]
        accs = [a for _, a in items]
        mean_conf = sum(confs) / len(confs)
        mean_acc = sum(accs) / len(accs)
        gap = abs(mean_conf - mean_acc)
        bin_summary[key] = {
            "n": len(items),
            "mean_conf": round(mean_conf, 4),
            "mean_acc": round(mean_acc, 4),
            "gap": round(gap, 4),
        }
        total_ece += (len(items) / n) * gap
    return {"n": n, "ece": round(total_ece, 4), "bins": bin_summary}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {}

    for target, (enc_dir, llm_dir, fmt) in TARGETS.items():
        per_model: dict[str, object] = {}
        for model in MODELS_ENC + MODELS_LLM:
            res = load_predictions(enc_dir, llm_dir, model, fmt)
            if res is None:
                continue
            seed_preds, gold = res  # type: ignore[misc]
            per_model[model] = compute_ece(seed_preds, gold)
        summary[target] = per_model
        print(f"{target}: {len(per_model)} models, "
              f"ECE range "
              f"{min((per_model[m]['ece'] for m in per_model), default=0):.3f}"
              "--"
              f"{max((per_model[m]['ece'] for m in per_model), default=0):.3f}")

    (OUT_DIR / "calibration.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Emit LaTeX fragment: ECE per (model, target) cell.
    rows: list[str] = []
    rows.append("\\begin{tabular}{lccc}")
    rows.append("\\toprule")
    rows.append("Model & SynB ECE & OLID ECE & HateXplain ECE \\\\")
    rows.append("\\midrule")
    pretty = {
        "bert_base": "BERT-base", "roberta_base": "RoBERTa-base",
        "hatebert": "HateBERT", "deberta_v3": "DeBERTa-v3",
        "internlm2_5_7b": "InternLM-7B+LoRA",
        "qwen2_5_7b": "Qwen-7B+LoRA",
    }

    def fmt_ece(t: str, m: str) -> str:
        target_map = summary.get(t, {})  # type: ignore[arg-type]
        d = target_map.get(m, None) if isinstance(target_map, dict) else None
        if not d:
            return "--"
        return f"{d['ece']:.3f}"  # type: ignore[index]

    for m in MODELS_ENC + MODELS_LLM:
        rows.append(
            f"{pretty[m]} & {fmt_ece('synb_w4', m)} & "
            f"{fmt_ece('olid', m)} & {fmt_ece('hatexplain', m)} \\\\"
        )
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")

    (OUT_DIR / "calibration_table.tex").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )
    print(f"wrote {OUT_DIR / 'calibration_table.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
