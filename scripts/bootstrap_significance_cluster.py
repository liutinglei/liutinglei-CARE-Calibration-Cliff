"""Conversation-level cluster paired bootstrap for Macro-F1 differences.

This replaces the utterance-level bootstrap previously used in
``bootstrap_significance.py``. The resampling unit is the *conversation*
(``conv_id``), not the row index, which is the correct independence
assumption when test utterances belong to overlapping context windows
within the same dialogue.

For each bootstrap iteration:
  1. Draw a random seed for system A and system B independently.
  2. Sample ``n_clusters`` conv_ids with replacement.
  3. Concatenate all utterances belonging to those conv_ids.
  4. Compute Macro-F1 for each system on the concatenated indices.

The script also writes per-row label-alignment diagnostics so we can
sanity-check that the recovered conv_id vector matches the order used
by the prediction CSVs.

Usage::

    python bootstrap_significance_cluster.py \\
        --pred_dir_a paper/outputs/baselines \\
        --prefix_a roberta_base_w4 \\
        --pred_dir_b paper/outputs/llm \\
        --prefix_b internlm2_5_7b_w4 \\
        --conv_ids paper/outputs/test_conv_ids.json \\
        --B 10000 \\
        --out paper/outputs/bootstrap_cluster/roberta_vs_internlm_w4.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

LABEL_MAP = {"harmful": 1, "not_harmful": 0, "not harmful": 0}


def _to_binary(series: pd.Series) -> np.ndarray:
    cleaned = series.astype(str).str.strip().str.lower().str.replace(" ", "_")
    return cleaned.map(LABEL_MAP).to_numpy()


def load_pred_pool(
    pred_dir: str | Path, prefix: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    paths = sorted(Path(pred_dir).glob(f"{prefix}_seed*_predictions.csv"))
    if not paths:
        raise SystemExit(f"no predictions for prefix={prefix} in {pred_dir}")
    pred_mat: list[np.ndarray] = []
    label_vec: np.ndarray | None = None
    for path in paths:
        df = pd.read_csv(path)
        lab = _to_binary(df["label"])
        pr = _to_binary(df["pred"])
        if label_vec is None:
            label_vec = lab
        elif not np.array_equal(label_vec, lab):
            raise SystemExit(f"label mismatch between seeds in {path}")
        pred_mat.append(pr)
    assert label_vec is not None
    return np.stack(pred_mat, axis=0), label_vec, [p.stem for p in paths]


def macro_f1(labels: np.ndarray, preds: np.ndarray) -> float:
    _, _, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], average=None, zero_division=0
    )
    return float(f1.mean())


def harm_f1(labels: np.ndarray, preds: np.ndarray) -> float:
    _, _, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], average=None, zero_division=0
    )
    return float(f1[1])


def build_cluster_index(conv_ids: list[str]) -> dict[str, np.ndarray]:
    by_conv: dict[str, list[int]] = defaultdict(list)
    for idx, cid in enumerate(conv_ids):
        by_conv[cid].append(idx)
    return {k: np.array(v, dtype=np.int64) for k, v in by_conv.items()}


def cluster_bootstrap_diff(
    a_mat: np.ndarray,
    b_mat: np.ndarray,
    labels: np.ndarray,
    cluster_index: dict[str, np.ndarray],
    B: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    conv_keys = list(cluster_index.keys())
    n_clusters = len(conv_keys)
    diffs_macro = np.zeros(B)
    diffs_harm = np.zeros(B)
    a_macro = np.zeros(B)
    b_macro = np.zeros(B)
    diffs_harm_only = np.zeros(B)
    for i in range(B):
        sa = rng.integers(a_mat.shape[0])
        sb = rng.integers(b_mat.shape[0])
        sampled = rng.integers(n_clusters, size=n_clusters)
        chunks = [cluster_index[conv_keys[k]] for k in sampled]
        idx = np.concatenate(chunks)
        lab_b = labels[idx]
        pa = a_mat[sa][idx]
        pb = b_mat[sb][idx]
        a_macro[i] = macro_f1(lab_b, pa)
        b_macro[i] = macro_f1(lab_b, pb)
        diffs_macro[i] = a_macro[i] - b_macro[i]
        diffs_harm[i] = harm_f1(lab_b, pa) - harm_f1(lab_b, pb)
        diffs_harm_only[i] = diffs_harm[i]
    return a_macro, b_macro, diffs_macro, diffs_harm_only, np.array([])


def _ci(arr: np.ndarray, q: tuple[float, float] = (0.025, 0.975)) -> list[float]:
    return [float(np.quantile(arr, q[0])), float(np.quantile(arr, q[1]))]


def jackknife_seed_averaged_diff(
    a_mat: np.ndarray,
    b_mat: np.ndarray,
    labels: np.ndarray,
    cluster_index: dict[str, np.ndarray],
    metric,
) -> tuple[float, np.ndarray]:
    """Leave-one-conversation-out jackknife on the seed-averaged
    diff statistic. Returns (overall theta_hat, jackknife array)."""
    conv_keys = list(cluster_index.keys())
    all_idx = np.concatenate([cluster_index[k] for k in conv_keys])
    a_overall = float(
        np.mean(
            [
                metric(labels[all_idx], a_mat[s][all_idx])
                for s in range(a_mat.shape[0])
            ]
        )
    )
    b_overall = float(
        np.mean(
            [
                metric(labels[all_idx], b_mat[s][all_idx])
                for s in range(b_mat.shape[0])
            ]
        )
    )
    theta_overall = a_overall - b_overall

    theta_jack = np.zeros(len(conv_keys))
    for i, drop_key in enumerate(conv_keys):
        keep_idx = np.concatenate(
            [cluster_index[k] for k in conv_keys if k != drop_key]
        )
        a_jk = float(
            np.mean(
                [
                    metric(labels[keep_idx], a_mat[s][keep_idx])
                    for s in range(a_mat.shape[0])
                ]
            )
        )
        b_jk = float(
            np.mean(
                [
                    metric(labels[keep_idx], b_mat[s][keep_idx])
                    for s in range(b_mat.shape[0])
                ]
            )
        )
        theta_jack[i] = a_jk - b_jk
    return theta_overall, theta_jack


def bca_ci(
    boot_samples: np.ndarray,
    theta_overall: float,
    theta_jack: np.ndarray,
    alpha: float = 0.05,
) -> list[float]:
    """Bias-Corrected and accelerated (BCa) percentile CI."""
    from scipy.stats import norm

    # Bias correction z0
    frac_below = float(np.mean(boot_samples < theta_overall))
    if frac_below in (0.0, 1.0):
        # degenerate; fall back to percentile CI
        return [
            float(np.quantile(boot_samples, alpha / 2)),
            float(np.quantile(boot_samples, 1 - alpha / 2)),
        ]
    z0 = float(norm.ppf(frac_below))

    # Acceleration a_hat from jackknife
    theta_jack_mean = float(np.mean(theta_jack))
    diff = theta_jack_mean - theta_jack
    num = float(np.sum(diff ** 3))
    den = 6.0 * float(np.sum(diff ** 2)) ** 1.5
    a_hat = num / den if den != 0.0 else 0.0

    z_lo = float(norm.ppf(alpha / 2))
    z_hi = float(norm.ppf(1 - alpha / 2))
    alpha1 = float(
        norm.cdf(z0 + (z0 + z_lo) / (1 - a_hat * (z0 + z_lo)))
    )
    alpha2 = float(
        norm.cdf(z0 + (z0 + z_hi) / (1 - a_hat * (z0 + z_hi)))
    )
    return [
        float(np.quantile(boot_samples, alpha1)),
        float(np.quantile(boot_samples, alpha2)),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir_a", required=True)
    ap.add_argument("--prefix_a", required=True)
    ap.add_argument("--pred_dir_b", required=True)
    ap.add_argument("--prefix_b", required=True)
    ap.add_argument("--conv_ids", required=True,
                    help="Path to test_conv_ids.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    a_mat, labels_a, files_a = load_pred_pool(args.pred_dir_a, args.prefix_a)
    b_mat, labels_b, files_b = load_pred_pool(args.pred_dir_b, args.prefix_b)
    if not np.array_equal(labels_a, labels_b):
        raise SystemExit("label vectors differ between A and B")

    conv_meta = json.loads(Path(args.conv_ids).read_text(encoding="utf-8"))
    conv_ids = conv_meta[f"{args.split}_conv_ids"]
    ref_labels = np.array(conv_meta[f"{args.split}_labels"])
    if len(conv_ids) != len(labels_a):
        raise SystemExit(
            f"length mismatch: conv_ids={len(conv_ids)} preds={len(labels_a)}"
        )
    if not np.array_equal(ref_labels, labels_a):
        diffs = int(np.sum(ref_labels != labels_a))
        raise SystemExit(
            f"label vector from conv_ids manifest does not match prediction "
            f"labels ({diffs} positions differ); ordering is not aligned."
        )

    cluster_index = build_cluster_index(conv_ids)
    print(
        f"A: {len(files_a)} seed(s) shape={a_mat.shape};  "
        f"B: {len(files_b)} seed(s) shape={b_mat.shape};  "
        f"clusters={len(cluster_index)} rows={len(conv_ids)}"
    )

    rng = np.random.default_rng(args.seed)
    a_macro, b_macro, d_macro, d_harm, _ = cluster_bootstrap_diff(
        a_mat, b_mat, labels_a, cluster_index, args.B, rng
    )
    p_macro_two = 2 * min(np.mean(d_macro >= 0), np.mean(d_macro <= 0))
    p_harm_two = 2 * min(np.mean(d_harm >= 0), np.mean(d_harm <= 0))

    # BCa CI + design-level effective sample size
    theta_macro_overall, jack_macro = jackknife_seed_averaged_diff(
        a_mat, b_mat, labels_a, cluster_index, macro_f1
    )
    theta_harm_overall, jack_harm = jackknife_seed_averaged_diff(
        a_mat, b_mat, labels_a, cluster_index, harm_f1
    )
    bca_macro = bca_ci(d_macro, theta_macro_overall, jack_macro)
    bca_harm = bca_ci(d_harm, theta_harm_overall, jack_harm)

    summary: dict[str, object] = {
        "system_A": args.prefix_a,
        "system_A_files": files_a,
        "system_B": args.prefix_b,
        "system_B_files": files_b,
        "B": args.B,
        "seed": args.seed,
        "resampling_unit": "conversation",
        "n_clusters": len(cluster_index),
        "n_rows": len(conv_ids),
        "effective_sample_size_clusters": len(cluster_index),
        "jackknife_replications": len(cluster_index),
        "A_mean_macroF1": float(a_macro.mean()),
        "A_ci95_macroF1": _ci(a_macro),
        "B_mean_macroF1": float(b_macro.mean()),
        "B_ci95_macroF1": _ci(b_macro),
        "diff_macroF1_overall": theta_macro_overall,
        "diff_macroF1_mean": float(d_macro.mean()),
        "diff_macroF1_ci95": _ci(d_macro),
        "diff_macroF1_bca_ci95": bca_macro,
        "p_value_macroF1_two_sided": float(p_macro_two),
        "diff_harmF1_overall": theta_harm_overall,
        "diff_harmF1_mean": float(d_harm.mean()),
        "diff_harmF1_ci95": _ci(d_harm),
        "diff_harmF1_bca_ci95": bca_harm,
        "p_value_harmF1_two_sided": float(p_harm_two),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
