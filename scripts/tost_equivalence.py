"""Two-One-Sided-Tests (TOST) equivalence procedure for the
conversation-level cluster bootstrap diffs.

Standard approach in clinical / behavioural science when a paper
claims that two interventions are "statistically indistinguishable":
the null of \\emph{difference} cannot be rejected by a vanilla
two-sided test, but to claim \\emph{equivalence} you must instead
formally reject both one-sided nulls

  H0_-: diff <= -Delta
  H0_+: diff >= +Delta

at level alpha. By the TOST argument, the test is equivalent to
checking that the (1-2*alpha) two-sided CI of the diff fits inside
the equivalence interval (-Delta, +Delta).

We set the Smallest Effect Size Of Interest (SESOI) Delta = 0.02
Macro-F1 by default --- this is roughly 2% absolute, which is the
order of magnitude of inter-seed noise on SynBullying and below the
typical practical-importance threshold for cyberbullying detection.

Reads ``paper/outputs/bootstrap_cluster/*.json`` (produced by
``run_cluster_bootstrap.py``) and writes a markdown table plus a
machine-readable JSON to ``paper/outputs/tost_equivalence.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / "outputs" / "bootstrap_cluster"
OUT = ROOT / "outputs" / "tost_equivalence.json"

PAIRS = [
    ("RoBERTa vs BERT-base",       "roberta_vs_bert_base_w4"),
    ("RoBERTa vs HateBERT",        "roberta_vs_hatebert_w4"),
    ("RoBERTa vs DeBERTa-v3",      "roberta_vs_deberta_v3_w4"),
    ("RoBERTa vs InternLM-7B+LoRA", "roberta_vs_internlm2_5_7b_w4"),
    ("RoBERTa vs Qwen-7B+LoRA",     "roberta_vs_qwen2_5_7b_w4"),
    ("InternLM vs Qwen",           "internlm_vs_qwen_w4"),
    ("RoBERTa w4 vs w0",           "roberta_w4_vs_w0"),
    ("RoBERTa w4 vs w2",           "roberta_w4_vs_w2"),
    ("RoBERTa w8 vs w4",           "roberta_w8_vs_w4"),
]


def tost_from_ci(ci_low: float, ci_hi: float, sesoi: float) -> tuple[str, str]:
    """Return (verdict, reason) where verdict is one of
    'equivalent', 'inconclusive', 'non-equivalent'.

    The 90% bootstrap CI is the right one for TOST at alpha=0.05.
    Our stored ``diff_ci95`` is the 95% CI, which is conservative for
    a TOST decision (wider CI is harder to fit inside the SESOI). We
    therefore use the 95% CI directly: if the 95% CI fits inside
    the SESOI, equivalence at alpha=0.05 is trivially established.
    """
    if ci_low > sesoi or ci_hi < -sesoi:
        return "non-equivalent", (
            f"95% CI ({ci_low:+.4f}, {ci_hi:+.4f}) lies entirely "
            f"outside +/- {sesoi:.3f}"
        )
    if ci_low >= -sesoi and ci_hi <= sesoi:
        return "equivalent", (
            f"95% CI ({ci_low:+.4f}, {ci_hi:+.4f}) fits inside "
            f"+/- {sesoi:.3f} (Delta)"
        )
    return "inconclusive", (
        f"95% CI ({ci_low:+.4f}, {ci_hi:+.4f}) straddles +/- {sesoi:.3f}; "
        "neither difference nor equivalence is established"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sesoi",
        type=float, default=0.02,
        help="Primary SESOI for the main TOST table (default 0.02).",
    )
    ap.add_argument(
        "--sesoi_sweep",
        type=str, default="0.01,0.02,0.03",
        help="Comma-separated SESOIs for the sensitivity sweep.",
    )
    ap.add_argument(
        "--ci_kind",
        choices=("percentile", "bca", "auto"),
        default="auto",
        help="Which CI to use: percentile, bca, or auto (bca if present).",
    )
    args = ap.parse_args()

    sesoi_grid = [float(x) for x in args.sesoi_sweep.split(",") if x.strip()]

    rows: list[dict[str, object]] = []
    sensitivity: list[dict[str, object]] = []

    def pick_ci(d: dict) -> tuple[list[float], str]:
        if args.ci_kind == "percentile" or "diff_macroF1_bca_ci95" not in d:
            return list(d["diff_macroF1_ci95"]), "percentile"
        if args.ci_kind == "bca":
            return list(d["diff_macroF1_bca_ci95"]), "bca"
        # auto: prefer bca when present
        return list(d["diff_macroF1_bca_ci95"]), "bca"

    print(
        f"\nTOST equivalence test against Delta = +/-{args.sesoi:.3f} "
        f"Macro-F1 (primary)\n"
    )
    print("| Pair | diff | 95% CI ({}) | Verdict |".format(
        args.ci_kind if args.ci_kind != "auto" else "bca if present"
    ))
    print("|---|---|---|---|")
    for label, key in PAIRS:
        path = BOOTSTRAP / f"{key}.json"
        if not path.is_file():
            print(f"| {label} | -- | -- | _no bootstrap_ |")
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        ci, ci_kind = pick_ci(d)
        diff = float(d["diff_macroF1_mean"])
        verdict, reason = tost_from_ci(float(ci[0]), float(ci[1]), args.sesoi)
        print(
            f"| {label} | {diff:+.4f} | "
            f"[{ci[0]:+.4f}, {ci[1]:+.4f}] | "
            f"**{verdict}** |"
        )
        rows.append(
            {
                "label": label,
                "key": key,
                "diff_macroF1": diff,
                "ci95": list(ci),
                "ci_kind": ci_kind,
                "sesoi": args.sesoi,
                "verdict": verdict,
                "reason": reason,
            }
        )

        # Sensitivity sweep across multiple SESOIs
        sweep_verdicts: dict[str, str] = {}
        for s in sesoi_grid:
            v, _ = tost_from_ci(float(ci[0]), float(ci[1]), s)
            sweep_verdicts[f"{s:.3f}"] = v
        sensitivity.append(
            {
                "label": label,
                "key": key,
                "ci_kind": ci_kind,
                "ci95": list(ci),
                "verdicts_by_sesoi": sweep_verdicts,
            }
        )

    # Print sensitivity sweep table
    print(
        f"\nSESOI sensitivity sweep over Delta in "
        f"{[f'{s:.3f}' for s in sesoi_grid]}:\n"
    )
    header = "| Pair | " + " | ".join(
        f"Delta={s:.3f}" for s in sesoi_grid
    ) + " |"
    print(header)
    print("|" + "---|" * (len(sesoi_grid) + 1))
    for r in sensitivity:
        cells = " | ".join(
            r["verdicts_by_sesoi"][f"{s:.3f}"] for s in sesoi_grid
        )
        print(f"| {r['label']} | {cells} |")

    OUT.write_text(
        json.dumps(
            {
                "sesoi": args.sesoi,
                "sesoi_sweep": sesoi_grid,
                "rows": rows,
                "sensitivity": sensitivity,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {OUT}")

    # Summary
    n_eq = sum(1 for r in rows if r["verdict"] == "equivalent")
    n_inconc = sum(1 for r in rows if r["verdict"] == "inconclusive")
    n_neq = sum(1 for r in rows if r["verdict"] == "non-equivalent")
    print(
        f"\nprimary-SESOI summary ({args.sesoi}): {n_eq} equivalent, "
        f"{n_inconc} inconclusive, {n_neq} non-equivalent"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
