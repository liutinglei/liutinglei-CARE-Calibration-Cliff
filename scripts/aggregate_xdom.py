"""Aggregate the 3-seed transfer metrics on OLID and HateXplain
into one JSON + one LaTeX table fragment.

Reads ``paper/outputs/{olid,hatexplain}/<model>_w4_seed{41,42,43}_metrics.json``
for the six standard model rows. Produces:

  * ``paper/outputs/xdom_transfer.json``
  * ``paper/outputs/xdom_transfer_table.tex``

The LaTeX fragment is a self-contained ``tabular`` that the
``06_results.tex`` cross-domain subsection imports via
``\\IfFileExists{outputs/xdom_transfer_table.tex}{\\input{...}}``.
"""
from __future__ import annotations

import json
import statistics as stats
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("BERT-base-uncased", "bert_base"),
    ("HateBERT", "hatebert"),
    ("RoBERTa-base", "roberta_base"),
    ("DeBERTa-v3-base", "deberta_v3"),
    ("Qwen2.5-7B + LoRA", "qwen2_5_7b"),
    ("InternLM2.5-7B + LoRA", "internlm2_5_7b"),
]
TARGETS = ["olid", "hatexplain", "synb", "civilcomments"]
SEEDS = [41, 42, 43]


def read_synb_row(short: str, seed: int) -> dict[str, float] | None:
    """Read in-domain SynBullying strict-w=4 metrics for a model."""
    sub = "llm" if short in ("internlm2_5_7b", "qwen2_5_7b") else "baselines"
    path = ROOT / "outputs" / sub / f"{short}_w4_seed{seed}_metrics.json"
    if not path.is_file():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    return {
        "macro_f1": float(d.get("macro_f1", 0.0)),
        "harm_f1": float(d.get("harmful_f1", 0.0)),
    }


def read_xdom_row(target: str, short: str, seed: int) -> dict[str, float] | None:
    if target == "synb":
        return read_synb_row(short, seed)
    # CivilComments is single-seed (42) only -- silently treat 41/43 as missing
    path = OUT_DIR / target / f"{short}_w4_seed{seed}_metrics.json"
    if not path.is_file():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    harm_key = "harmful_f1" if "harmful_f1" in d else (
        "off_f1" if "off_f1" in d else None
    )
    return {
        "macro_f1": float(d.get("macro_f1", 0.0)),
        "harm_f1": float(d.get(harm_key, 0.0)) if harm_key else 0.0,
    }


def aggregate(short: str) -> dict[str, dict[str, list[float] | float]]:
    out: dict[str, dict[str, list[float] | float]] = {}
    for target in TARGETS:
        macros: list[float] = []
        harms: list[float] = []
        seeds_used: list[int] = []
        for s in SEEDS:
            r = read_xdom_row(target, short, s)
            if r is None:
                continue
            macros.append(r["macro_f1"])
            harms.append(r["harm_f1"])
            seeds_used.append(s)
        if not macros:
            out[target] = {"n_seeds": 0}
            continue
        out[target] = {
            "n_seeds": len(macros),
            "seeds": seeds_used,
            "macro_f1_mean": float(stats.fmean(macros)),
            "macro_f1_sd": float(stats.pstdev(macros)) if len(macros) > 1 else 0.0,
            "harm_f1_mean": float(stats.fmean(harms)),
            "harm_f1_sd": float(stats.pstdev(harms)) if len(harms) > 1 else 0.0,
        }
    return out


def fmt_cell(d: dict[str, list[float] | float]) -> str:
    if d.get("n_seeds", 0) == 0:
        return "--"
    m = d["macro_f1_mean"]  # type: ignore[index]
    s = d["macro_f1_sd"]  # type: ignore[index]
    if d["n_seeds"] == 1:  # type: ignore[index]
        return f"{m:.3f}"
    return f"{m:.3f} ({s:.3f})"


def main() -> int:
    flat: dict[str, object] = {}
    for label, short in MODELS:
        flat[short] = aggregate(short)

    (OUT_DIR / "xdom_transfer.json").write_text(
        json.dumps(flat, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {OUT_DIR / 'xdom_transfer.json'}")

    rows: list[str] = []
    rows.append("\\begin{tabular}{lcccc}")
    rows.append("\\toprule")
    rows.append(
        "Model & SynB Strict-$w{=}4$ & OLID & HateXplain & CivilComments \\\\"
    )
    rows.append(
        "      & Macro-F1 (s.d.) & Macro-F1 (s.d.) & Macro-F1 (s.d.) & Macro-F1 (s.d.) \\\\"
    )
    rows.append("\\midrule")
    for label, short in MODELS:
        cells = flat[short]  # type: ignore[index]
        rows.append(
            f"{label} & "
            f"{fmt_cell(cells['synb'])} & "  # type: ignore[index]
            f"{fmt_cell(cells['olid'])} & "  # type: ignore[index]
            f"{fmt_cell(cells['hatexplain'])} & "  # type: ignore[index]
            f"{fmt_cell(cells['civilcomments'])} \\\\"  # type: ignore[index]
        )
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")

    (OUT_DIR / "xdom_transfer_table.tex").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )
    print(f"wrote {OUT_DIR / 'xdom_transfer_table.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
