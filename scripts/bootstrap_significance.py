#!/usr/bin/env python
"""Paired bootstrap test for Macro-F1 differences between two systems.

Both systems must have been evaluated on the same test set (same row order).
We pool predictions across seeds (concatenated) to obtain a more stable
estimate of the per-example agreement matrix, then resample row indices with
replacement and recompute the per-bootstrap-sample mean Macro-F1 of each
system.

Output: bootstrap mean, 95% CI, p-value(2-sided) for H0: F1_A == F1_B.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


def load_pred_pool(pred_dir, prefix):
    """Load all <prefix>_seed*_predictions.csv files. Return a 2D array of
    predictions of shape (n_seeds, n_examples) plus labels (1D)."""
    paths = sorted(Path(pred_dir).glob(f"{prefix}_seed*_predictions.csv"))
    if not paths:
        raise SystemExit(f"no predictions for prefix {prefix} in {pred_dir}")
    pred_mat = []
    label_vec = None
    for p in paths:
        df = pd.read_csv(p)
        lab = df["label"].map({"harmful": 1, "not_harmful": 0}).to_numpy()
        pr = df["pred"].map({"harmful": 1, "not_harmful": 0}).to_numpy()
        if label_vec is None:
            label_vec = lab
        else:
            if not np.array_equal(label_vec, lab):
                raise SystemExit(f"label mismatch in {p}")
        pred_mat.append(pr)
    return np.stack(pred_mat, axis=0), label_vec, [p.stem for p in paths]


def macro_f1(labels, preds):
    _, _, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], average=None, zero_division=0
    )
    return float(f1.mean())


def bootstrap_diff(a_mat, b_mat, labels, B=10000, rng=None):
    """a_mat, b_mat: (n_seeds, n) arrays of binary preds.
    For each bootstrap iteration: pick a random seed per system, sample
    indices with replacement, compute Macro-F1 for each, return their diff."""
    rng = rng or np.random.default_rng(42)
    n = len(labels)
    diffs = np.zeros(B)
    A_f1 = np.zeros(B); B_f1 = np.zeros(B)
    for i in range(B):
        sa = rng.integers(a_mat.shape[0])
        sb = rng.integers(b_mat.shape[0])
        idx = rng.integers(n, size=n)
        f1a = macro_f1(labels[idx], a_mat[sa][idx])
        f1b = macro_f1(labels[idx], b_mat[sb][idx])
        diffs[i] = f1a - f1b
        A_f1[i] = f1a; B_f1[i] = f1b
    return A_f1, B_f1, diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir_a", required=True)
    ap.add_argument("--prefix_a", required=True)
    ap.add_argument("--pred_dir_b", required=True)
    ap.add_argument("--prefix_b", required=True)
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    A, labels_a, files_a = load_pred_pool(args.pred_dir_a, args.prefix_a)
    B, labels_b, files_b = load_pred_pool(args.pred_dir_b, args.prefix_b)
    if not np.array_equal(labels_a, labels_b):
        raise SystemExit("label vectors differ between A and B")

    print(f"A: {len(files_a)} seed(s), {A.shape}")
    print(f"B: {len(files_b)} seed(s), {B.shape}")

    A_f1, B_f1, diffs = bootstrap_diff(A, B, labels_a, B=args.B)
    p_two_sided = 2 * min(np.mean(diffs >= 0), np.mean(diffs <= 0))

    summary = {
        "system_A": args.prefix_a,
        "system_A_files": files_a,
        "system_B": args.prefix_b,
        "system_B_files": files_b,
        "B": args.B,
        "A_mean_macroF1": float(A_f1.mean()),
        "A_ci95": [float(np.quantile(A_f1, 0.025)),
                   float(np.quantile(A_f1, 0.975))],
        "B_mean_macroF1": float(B_f1.mean()),
        "B_ci95": [float(np.quantile(B_f1, 0.025)),
                   float(np.quantile(B_f1, 0.975))],
        "diff_mean": float(diffs.mean()),
        "diff_ci95": [float(np.quantile(diffs, 0.025)),
                      float(np.quantile(diffs, 0.975))],
        "p_value_two_sided": float(p_two_sided),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
