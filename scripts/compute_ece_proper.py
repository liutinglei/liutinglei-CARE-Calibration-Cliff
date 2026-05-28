"""Standard probability-based Expected Calibration Error.

This implements the Guo et al. (2017, ICML) definition: bin examples by
their predicted confidence into 15 equal-width bins on [0, 1], compute
``|conf - acc|`` per bin, weight by bin size, sum.

For binary classifiers the predicted confidence on example i is
``max(p_i, 1 - p_i)`` where ``p_i = P(class=1 | x_i)``, and the empirical
accuracy of a bin is ``mean[pred_i == gold_i]`` over examples in the bin.

We additionally report:
  * Brier score, a strictly proper scoring rule:
    ``Brier = mean[(p_i - gold_i)^2]`` for the harmful class.
  * Maximum Calibration Error (MCE): the worst bin-level gap.
  * The reliability-diagram data (per-bin conf, acc, count) so downstream
    plotting code can render the curve without re-running.

Input: a predictions CSV with columns including ``label`` (string in
{harmful, not_harmful}) and ``prob_harmful`` (float).
Output: a JSON file with the ECE summary plus reliability bins.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


LABEL_MAP = {"harmful": 1, "not_harmful": 0, "not harmful": 0}


@dataclass(frozen=True)
class Bin:
    lo: float
    hi: float
    n: int
    mean_conf: float
    mean_acc: float
    gap: float


def load_probs_and_labels(csv_path: Path) -> tuple[list[float], list[int]]:
    probs: list[float] = []
    golds: list[int] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            label_str = row["label"].strip().lower().replace(" ", "_")
            if label_str not in LABEL_MAP:
                continue
            try:
                p = float(row["prob_harmful"])
            except (KeyError, ValueError):
                continue
            probs.append(p)
            golds.append(LABEL_MAP[label_str])
    return probs, golds


def compute_ece(
    probs: list[float], golds: list[int], n_bins: int = 15
) -> dict[str, object]:
    if not probs:
        return {"n": 0, "ece": 0.0, "mce": 0.0, "brier": 0.0, "bins": []}
    n = len(probs)
    # confidence = max(p, 1-p); prediction = argmax
    confs = [max(p, 1 - p) for p in probs]
    preds = [1 if p >= 0.5 else 0 for p in probs]
    correct = [1 if preds[i] == golds[i] else 0 for i in range(n)]
    brier = sum((probs[i] - golds[i]) ** 2 for i in range(n)) / n

    edges = [i / n_bins for i in range(n_bins + 1)]
    bins: list[Bin] = []
    ece_sum = 0.0
    mce = 0.0
    for k in range(n_bins):
        lo, hi = edges[k], edges[k + 1]
        idxs = [
            i
            for i in range(n)
            if (confs[i] > lo or (k == 0 and confs[i] == lo))
            and confs[i] <= hi
        ]
        if not idxs:
            continue
        bin_n = len(idxs)
        bin_conf = sum(confs[i] for i in idxs) / bin_n
        bin_acc = sum(correct[i] for i in idxs) / bin_n
        gap = abs(bin_conf - bin_acc)
        bins.append(
            Bin(
                lo=round(lo, 4),
                hi=round(hi, 4),
                n=bin_n,
                mean_conf=round(bin_conf, 4),
                mean_acc=round(bin_acc, 4),
                gap=round(gap, 4),
            )
        )
        ece_sum += (bin_n / n) * gap
        mce = max(mce, gap)

    return {
        "n": n,
        "ece": round(ece_sum, 4),
        "mce": round(mce, 4),
        "brier": round(brier, 4),
        "n_bins": n_bins,
        "bins": [b.__dict__ for b in bins],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions_csv", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--n_bins", type=int, default=15)
    args = ap.parse_args()

    probs, golds = load_probs_and_labels(Path(args.predictions_csv))
    if not probs:
        raise SystemExit(
            f"No usable rows in {args.predictions_csv}: "
            "predictions_csv must contain a 'prob_harmful' column."
        )
    res = compute_ece(probs, golds, n_bins=args.n_bins)
    res["source_csv"] = str(args.predictions_csv)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(
        json.dumps(res, indent=2), encoding="utf-8"
    )
    print(
        f"ECE={res['ece']:.4f}  MCE={res['mce']:.4f}  "
        f"Brier={res['brier']:.4f}  n={res['n']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
