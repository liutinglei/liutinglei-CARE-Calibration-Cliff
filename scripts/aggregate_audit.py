"""Aggregate the leakage-audit metrics into one JSON and one LaTeX table
fragment, with 3-seed mean (s.d.) for RoBERTa-base (v12) and
single-seed for InternLM (matched to v11 because re-running InternLM
across 3 seeds x 6 cells would cost ~9 GPU-hours; the strict-conv arm
of the main table already provides multi-seed InternLM context).

Reads
``outputs_audit/<split>__<anon>/roberta_base_w4_seed{41,42,43}_metrics.json``
and
``outputs_audit/<split>__<anon>/internlm2_5_7b_w4_seed42_metrics.json``.

Produces:
  * ``outputs/leakage_audit.json``
  * ``outputs/leakage_audit_table.tex``
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "outputs_audit"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ("strict_conv", "raw_utt")
ANONS = ("raw_role", "prefix_anon", "intext_anon")

ROBERTA_SEEDS = (41, 42, 43)

PRETTY_SPLIT = {
    "strict_conv": "Strict (conv-level)",
    "raw_utt": "Naive (utt-level)",
}
PRETTY_ANON = {
    "raw_role": "Raw role tokens",
    "prefix_anon": "Speaker prefix only",
    "intext_anon": "Full in-text anon",
}


def read_metrics(path: Path) -> dict[str, float] | None:
    if not path.is_file():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    return {
        "macro_f1": float(d.get("macro_f1", 0.0)),
        "harm_f1": float(d.get("harmful_f1", 0.0)),
        "accuracy": float(d.get("accuracy", 0.0)),
    }


def roberta_3seed(cell_dir: Path) -> dict[str, dict[str, float]] | None:
    macros: list[float] = []
    harms: list[float] = []
    accs: list[float] = []
    for s in ROBERTA_SEEDS:
        p = cell_dir / f"roberta_base_w4_seed{s}_metrics.json"
        m = read_metrics(p)
        if m is None:
            continue
        macros.append(m["macro_f1"])
        harms.append(m["harm_f1"])
        accs.append(m["accuracy"])
    if not macros:
        return None
    out: dict[str, dict[str, float]] = {}
    for name, vals in (("macro_f1", macros),
                       ("harm_f1", harms),
                       ("accuracy", accs)):
        out[name] = {
            "mean": float(statistics.fmean(vals)),
            "std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
            "n_seeds": len(vals),
        }
    return out


def fmt_3seed(d: dict[str, float] | None, key: str) -> str:
    if not d or key not in d:
        return "--"
    m = d[key]["mean"]
    s = d[key]["std"]
    return f"{m:.3f} ({s:.3f})"


def fmt_1seed(d: dict[str, float] | None, key: str) -> str:
    if not d:
        return "--"
    return f"{d[key]:.3f}"


def main() -> int:
    flat: dict[str, dict[str, object]] = {}
    for split in SPLITS:
        for anon in ANONS:
            cell = f"{split}__{anon}"
            cell_dir = ROOT / cell
            rob = roberta_3seed(cell_dir)
            iln = read_metrics(
                cell_dir / "internlm2_5_7b_w4_seed42_metrics.json"
            )
            flat[cell] = {
                "RoBERTa-base (3-seed)": rob,
                "InternLM2.5-7B+LoRA (seed42)": iln,
            }

    out_json = OUT_DIR / "leakage_audit.json"
    out_json.write_text(
        json.dumps(flat, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {out_json}")

    rows: list[str] = []
    rows.append("\\begin{tabular}{llcccc}")
    rows.append("\\toprule")
    rows.append(
        "& & \\multicolumn{2}{c}{RoBERTa-base (3-seed mean, s.d.)} & "
        "\\multicolumn{2}{c}{InternLM2.5-7B+LoRA (seed 42)} \\\\"
    )
    rows.append("\\cmidrule(lr){3-4} \\cmidrule(lr){5-6}")
    rows.append(
        "Split & Anonymisation & Macro-F1 & Harm-F1 & Macro-F1 & Harm-F1 \\\\"
    )
    rows.append("\\midrule")
    for split in SPLITS:
        for anon in ANONS:
            cell = f"{split}__{anon}"
            r = flat[cell]
            rob = r["RoBERTa-base (3-seed)"]
            iln = r["InternLM2.5-7B+LoRA (seed42)"]
            rows.append(
                f"{PRETTY_SPLIT[split]} & {PRETTY_ANON[anon]} & "
                f"{fmt_3seed(rob, 'macro_f1')} & "
                f"{fmt_3seed(rob, 'harm_f1')} & "
                f"{fmt_1seed(iln, 'macro_f1')} & "
                f"{fmt_1seed(iln, 'harm_f1')} \\\\"
            )
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")

    out_tex = OUT_DIR / "leakage_audit_table.tex"
    out_tex.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote {out_tex}")

    # Quick-scan inflation deltas
    print("\nLeakage inflation (Naive - Strict, RoBERTa 3-seed mean):")
    for anon in ANONS:
        s = flat[f"strict_conv__{anon}"]["RoBERTa-base (3-seed)"]
        n = flat[f"raw_utt__{anon}"]["RoBERTa-base (3-seed)"]
        if s and n:
            delta_m = n["macro_f1"]["mean"] - s["macro_f1"]["mean"]
            delta_h = n["harm_f1"]["mean"] - s["harm_f1"]["mean"]
            print(f"  {anon}: Macro +{delta_m:+.4f}, Harm +{delta_h:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
