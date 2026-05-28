"""Per-conversation disagreement analysis: do the cluster-tied 6
systems make their mistakes on the same utterances, or on disjoint
ones? If disjoint, an ensemble would beat any individual; if
concentrated, the tie reflects a genuine ceiling.

We compute, **per seed in {41, 42, 43}**:
  1. pairwise prediction agreement (Cohen's-kappa-like, but on
     binary harm labels)
  2. per-utterance "agreement rate" across the 6 systems
  3. an oracle ensemble accuracy: if ANY of 6 systems is correct,
     the ensemble gets it right -- upper bound on what model
     diversity can buy.

Each cell is then reported as a 3-seed mean (s.d.).

Predictions are sourced from
``paper/outputs/calibration_v2/preds/synb_w4_test__{model}__seed{s}.csv``
which exists for every (model, seed) cell after Phase 2B.

Output: ``paper/outputs/disagreement.json`` + a LaTeX fragment
``paper/outputs/disagreement_table.tex``.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
PRED_DIR = ROOT / "outputs" / "calibration_v2" / "preds"

SEEDS = (41, 42, 43)

MODELS: list[tuple[str, str]] = [
    ("BERT-base", "bert_base"),
    ("HateBERT", "hatebert"),
    ("RoBERTa-base", "roberta_base"),
    ("DeBERTa-v3", "deberta_v3"),
    ("InternLM-7B+LoRA", "internlm2_5_7b"),
    ("Qwen-7B+LoRA", "qwen2_5_7b"),
]


def to_binary(s: pd.Series) -> list[int]:
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .map({"harmful": 1, "not_harmful": 0, "not harmful": 0})
        .tolist()
    )


def per_seed_metrics(seed: int) -> dict | None:
    preds: dict[str, list[int]] = {}
    labels: list[int] | None = None
    for name, short in MODELS:
        path = PRED_DIR / f"synb_w4_test__{short}__seed{seed}.csv"
        if not path.is_file():
            print(f"MISS {path.name}")
            return None
        df = pd.read_csv(path)
        preds[name] = to_binary(df["pred"])
        if labels is None:
            labels = to_binary(df["label"])
    n = len(labels)
    n_systems = len(preds)

    indiv = {
        m: sum(1 for i in range(n) if preds[m][i] == labels[i]) / n
        for m in preds
    }

    mv_correct = 0
    for i in range(n):
        ones = sum(preds[m][i] for m in preds)
        mv_pred = 1 if ones > n_systems / 2 else 0
        if mv_pred == labels[i]:
            mv_correct += 1
    mv_acc = mv_correct / n

    oracle = sum(
        1 for i in range(n)
        if any(preds[m][i] == labels[i] for m in preds)
    )
    oracle_acc = oracle / n

    n_unanimous = 0
    for i in range(n):
        ones = sum(preds[m][i] for m in preds)
        if ones in (0, n_systems):
            n_unanimous += 1

    return {
        "seed": seed,
        "n_test": n,
        "n_unanimous": n_unanimous,
        "n_disagreement": n - n_unanimous,
        "frac_unanimous": n_unanimous / n,
        "individual_accuracy": {m: round(a, 4) for m, a in indiv.items()},
        "majority_vote_accuracy": round(mv_acc, 4),
        "oracle_any_correct_accuracy": round(oracle_acc, 4),
    }


def aggregate(per_seed: list[dict]) -> dict:
    out: dict = {"per_seed": per_seed, "n_seeds": len(per_seed)}
    out["n_test"] = per_seed[0]["n_test"]

    # individual accuracy: mean (s.d.) per model
    indiv_stats: dict[str, tuple[float, float]] = {}
    for m, _ in MODELS:
        vals = [s["individual_accuracy"][m] for s in per_seed]
        indiv_stats[m] = (
            float(statistics.fmean(vals)),
            float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
        )
    out["individual_accuracy_3seed"] = {
        m: {"mean": round(v[0], 4), "std": round(v[1], 4)}
        for m, v in indiv_stats.items()
    }

    mv = [s["majority_vote_accuracy"] for s in per_seed]
    ora = [s["oracle_any_correct_accuracy"] for s in per_seed]
    una = [s["frac_unanimous"] for s in per_seed]
    out["majority_vote_3seed"] = {
        "mean": round(float(statistics.fmean(mv)), 4),
        "std": round(float(statistics.pstdev(mv)), 4)
        if len(mv) > 1 else 0.0,
    }
    out["oracle_3seed"] = {
        "mean": round(float(statistics.fmean(ora)), 4),
        "std": round(float(statistics.pstdev(ora)), 4)
        if len(ora) > 1 else 0.0,
    }
    out["frac_unanimous_3seed"] = {
        "mean": round(float(statistics.fmean(una)), 4),
        "std": round(float(statistics.pstdev(una)), 4)
        if len(una) > 1 else 0.0,
    }
    out["headroom_3seed"] = round(
        out["oracle_3seed"]["mean"] -
        max(v["mean"] for v in out["individual_accuracy_3seed"].values()),
        4,
    )
    return out


def main() -> int:
    per_seed = []
    for s in SEEDS:
        r = per_seed_metrics(s)
        if r is not None:
            per_seed.append(r)
    if not per_seed:
        print("no seed data available")
        return 1
    summary = aggregate(per_seed)

    (OUT_DIR / "disagreement.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    rows: list[str] = []
    rows.append("\\begin{tabular}{lc}")
    rows.append("\\toprule")
    rows.append("Setting & Accuracy (3-seed mean, s.d.) \\\\")
    rows.append("\\midrule")
    for m, _ in MODELS:
        v = summary["individual_accuracy_3seed"][m]
        rows.append(
            f"Individual: {m} & {v['mean']:.3f} ({v['std']:.3f}) \\\\"
        )
    rows.append("\\midrule")
    mv = summary["majority_vote_3seed"]
    ora = summary["oracle_3seed"]
    rows.append(
        f"Majority vote of 6 systems & \\textbf{{{mv['mean']:.3f}}} ({mv['std']:.3f}) \\\\"
    )
    rows.append(
        f"Oracle (any-of-6 correct) & {ora['mean']:.3f} ({ora['std']:.3f}) \\\\"
    )
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    (OUT_DIR / "disagreement_table.tex").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )

    print(f"n_seeds={summary['n_seeds']}, n_test={summary['n_test']}")
    print(f"MV mean={mv['mean']:.4f} (s.d. {mv['std']:.4f})")
    print(f"Oracle mean={ora['mean']:.4f} (s.d. {ora['std']:.4f})")
    print(f"headroom (oracle - best individual mean) = "
          f"{summary['headroom_3seed']:.4f}")
    print("Per-system individual 3-seed:")
    for m, v in summary["individual_accuracy_3seed"].items():
        print(f"  {m}: {v['mean']:.4f} (s.d. {v['std']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
