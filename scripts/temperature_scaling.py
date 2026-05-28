"""Temperature scaling on top of an arbitrary 2-class probability vector.

Given a held-out *validation* CSV with ``label`` and ``prob_harmful``
columns, we recover the implied logit ``z = log(p / (1 - p))`` for each
example and learn a single scalar temperature ``T > 0`` that minimises
the negative log-likelihood on validation. We then apply the same ``T``
to a *target* CSV's probabilities and emit a new CSV whose
``prob_harmful`` column is the temperature-rescaled value.

This is the canonical Guo et al. 2017 recipe restricted to a single
parameter, which is the only setting that does not overfit on a small
validation set. The learned ``T`` (along with raw vs. post-scaling NLL
and ECE on the *validation* set) is recorded as a JSON summary.

The script does NOT recompute ECE on the rescaled target file --- chain
``compute_ece_proper.py`` afterwards for that.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path


LABEL_MAP = {"harmful": 1, "not_harmful": 0, "not harmful": 0}
EPS = 1e-7


@dataclass(frozen=True)
class Row:
    label: int
    prob: float


def load_csv(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ls = row["label"].strip().lower().replace(" ", "_")
            if ls not in LABEL_MAP:
                continue
            try:
                p = float(row["prob_harmful"])
            except (KeyError, ValueError):
                continue
            rows.append(Row(label=LABEL_MAP[ls], prob=min(max(p, EPS), 1 - EPS)))
    return rows


def prob_to_logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def logit_to_prob(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def nll(rows: list[Row], T: float) -> float:
    total = 0.0
    for r in rows:
        z = prob_to_logit(r.prob)
        p1 = logit_to_prob(z / T)
        p1 = min(max(p1, EPS), 1 - EPS)
        total += -math.log(p1 if r.label == 1 else (1.0 - p1))
    return total / max(1, len(rows))


def grad_T(rows: list[Row], T: float) -> float:
    # d/dT of (-1/n sum log p_y(T)) = -1/n sum y_i * d/dT log p1 + ...
    # closed-form via chain rule with sigmoid is messy; use finite diff
    # since this is 1-D and tiny.
    h = 1e-4
    return (nll(rows, T + h) - nll(rows, T - h)) / (2 * h)


def fit_temperature(
    rows: list[Row],
    lo: float = 0.05,
    hi: float = 10.0,
    n_iter: int = 200,
) -> float:
    """Golden-section search on a continuous, unimodal NLL(T)."""
    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - (b - a) * invphi
    d = a + (b - a) * invphi
    fc = nll(rows, c)
    fd = nll(rows, d)
    for _ in range(n_iter):
        if abs(b - a) < 1e-4:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - (b - a) * invphi
            fc = nll(rows, c)
        else:
            a, c, fc = c, d, fd
            d = a + (b - a) * invphi
            fd = nll(rows, d)
    return (a + b) / 2.0


def apply_T_to_csv(in_path: Path, out_path: Path, T: float) -> int:
    with in_path.open("r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    if "prob_harmful" not in fieldnames:
        raise SystemExit(
            f"{in_path} does not have a prob_harmful column; "
            "regenerate via the patched eval script first."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with out_path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            try:
                p = float(row["prob_harmful"])
            except ValueError:
                writer.writerow(row)
                continue
            p_safe = min(max(p, EPS), 1 - EPS)
            z = prob_to_logit(p_safe)
            new_p = logit_to_prob(z / T)
            row["prob_harmful"] = f"{new_p:.6f}"
            writer.writerow(row)
            n_written += 1
    return n_written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--target_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--summary_json", required=True)
    args = ap.parse_args()

    val = load_csv(Path(args.val_csv))
    if not val:
        raise SystemExit(f"empty validation CSV: {args.val_csv}")
    nll_raw = nll(val, 1.0)
    T = fit_temperature(val)
    nll_scaled = nll(val, T)

    n = apply_T_to_csv(
        Path(args.target_csv), Path(args.out_csv), T
    )

    summary = {
        "val_csv": str(args.val_csv),
        "val_n": len(val),
        "target_csv": str(args.target_csv),
        "out_csv": str(args.out_csv),
        "n_rows_rescaled": n,
        "temperature": round(T, 4),
        "val_nll_T_eq_1": round(nll_raw, 4),
        "val_nll_at_T": round(nll_scaled, 4),
        "val_nll_improvement": round(nll_raw - nll_scaled, 4),
    }
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
