"""Aggregate per-cell ECE JSONs into the calibration / cross-domain
calibration LaTeX tables used by Section VI.J of the paper.

Expected on-disk layout (produced by the v10 calibration pipeline)::

    outputs/calibration_v2/
        ece/
            {dataset}__{model}__seed{N}.json    # raw ECE
        ece_temp/
            {dataset}__{model}__seed{N}.json    # temp-scaled ECE
        temp/
            {dataset}__{model}__seed{N}.json    # temperature summary

Datasets:   synb_w4 | civilcomments | olid | hatexplain
Encoders:   bert_base, roberta_base, hatebert, deberta_v3 (seed 42 only)
LLMs:       internlm2_5_7b, qwen2_5_7b (seeds 41, 42, 43)

Emits three LaTeX fragments:

  * ``calibration_v2_table.tex`` -- main 4-dataset x 6-model ECE table.
  * ``calibration_tempscaled_table.tex`` -- raw vs. temp-scaled ECE
    on the three OOD targets only.
  * ``calibration_summary.json`` -- machine-readable summary.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path


DATASETS = ("synb_w4_test", "civilcomments", "olid", "hatexplain")
DATASET_LABELS = {
    "synb_w4_test": "SynB",
    "civilcomments": "CivCom",
    "olid": "OLID",
    "hatexplain": "HateX",
}
ENCODER_MODELS = ("bert_base", "roberta_base", "hatebert", "deberta_v3")
LLM_MODELS = ("internlm2_5_7b", "qwen2_5_7b")
PRETTY = {
    "bert_base": "BERT-base",
    "roberta_base": "RoBERTa-base",
    "hatebert": "HateBERT",
    "deberta_v3": "DeBERTa-v3",
    "internlm2_5_7b": "InternLM-7B+LoRA",
    "qwen2_5_7b": "Qwen-7B+LoRA",
}
FNAME_RE = re.compile(r"^(?P<ds>[a-z0-9_]+)__(?P<model>[a-z0-9_]+)__seed(?P<seed>\d+)\.json$")


@dataclass(frozen=True)
class Cell:
    eces: tuple[float, ...]
    briers: tuple[float, ...]
    mces: tuple[float, ...]

    @property
    def mean_ece(self) -> float:
        return statistics.fmean(self.eces) if self.eces else float("nan")

    @property
    def std_ece(self) -> float:
        return statistics.stdev(self.eces) if len(self.eces) > 1 else 0.0


def collect(ece_dir: Path) -> dict[tuple[str, str], Cell]:
    """Map (dataset, model) -> Cell with all seeds collapsed."""
    seeds: dict[tuple[str, str], list[tuple[float, float, float]]] = {}
    for jf in sorted(ece_dir.glob("*.json")):
        m = FNAME_RE.match(jf.name)
        if not m:
            continue
        ds = m.group("ds")
        model = m.group("model")
        try:
            d = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        key = (ds, model)
        seeds.setdefault(key, []).append(
            (float(d.get("ece", 0.0)),
             float(d.get("brier", 0.0)),
             float(d.get("mce", 0.0)))
        )
    out: dict[tuple[str, str], Cell] = {}
    for k, vs in seeds.items():
        out[k] = Cell(
            eces=tuple(v[0] for v in vs),
            briers=tuple(v[1] for v in vs),
            mces=tuple(v[2] for v in vs),
        )
    return out


def fmt_cell(cell: Cell | None) -> str:
    if cell is None or not cell.eces:
        return "--"
    if len(cell.eces) > 1:
        return f"{cell.mean_ece:.3f} ({cell.std_ece:.3f})"
    return f"{cell.mean_ece:.3f}"


def build_main_table(raw: dict[tuple[str, str], Cell]) -> str:
    """Main ECE table: 6 model rows by 4 dataset columns."""
    lines: list[str] = []
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    cols = " & ".join(DATASET_LABELS[d] + " ECE" for d in DATASETS)
    lines.append(f"Model & {cols} \\\\")
    lines.append("\\midrule")
    for model in ENCODER_MODELS:
        cells = [fmt_cell(raw.get((d, model))) for d in DATASETS]
        lines.append(f"{PRETTY[model]} & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    for model in LLM_MODELS:
        cells = [fmt_cell(raw.get((d, model))) for d in DATASETS]
        lines.append(f"{PRETTY[model]} & " + " & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def build_temp_table(
    raw: dict[tuple[str, str], Cell],
    temp: dict[tuple[str, str], Cell],
) -> str:
    """Raw vs temperature-scaled ECE on the 3 OOD targets only."""
    ood = ("civilcomments", "olid", "hatexplain")
    lines: list[str] = []
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\toprule")
    lines.append(
        " & \\multicolumn{2}{c}{CivCom} & \\multicolumn{2}{c}{OLID} "
        "& \\multicolumn{2}{c}{HateX} \\\\"
    )
    lines.append(
        "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}"
    )
    lines.append(
        "Model & Raw & T-sc. & Raw & T-sc. & Raw & T-sc. \\\\"
    )
    lines.append("\\midrule")
    for model in ENCODER_MODELS + LLM_MODELS:
        cells: list[str] = []
        for d in ood:
            cells.append(fmt_cell(raw.get((d, model))))
            cells.append(fmt_cell(temp.get((d, model))))
        lines.append(f"{PRETTY[model]} & " + " & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="paper/outputs/calibration_v2",
        help="Directory containing ece/, ece_temp/, temp/ subdirs.",
    )
    ap.add_argument(
        "--out_dir",
        default="paper/outputs",
        help="Where to write the LaTeX fragments and JSON summary.",
    )
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = collect(root / "ece") if (root / "ece").is_dir() else {}
    temp = collect(root / "ece_temp") if (root / "ece_temp").is_dir() else {}

    main_tex = build_main_table(raw)
    (out_dir / "calibration_v2_table.tex").write_text(
        main_tex, encoding="utf-8"
    )
    print(f"wrote {out_dir / 'calibration_v2_table.tex'}")

    if temp:
        temp_tex = build_temp_table(raw, temp)
        (out_dir / "calibration_tempscaled_table.tex").write_text(
            temp_tex, encoding="utf-8"
        )
        print(f"wrote {out_dir / 'calibration_tempscaled_table.tex'}")

    summary: dict[str, object] = {
        "raw": {
            f"{ds}__{m}": {
                "n_seeds": len(cell.eces),
                "mean_ece": round(cell.mean_ece, 4),
                "std_ece": round(cell.std_ece, 4),
                "mean_brier": round(
                    statistics.fmean(cell.briers) if cell.briers else 0.0, 4
                ),
            }
            for (ds, m), cell in raw.items()
        },
        "tempscaled": {
            f"{ds}__{m}": {
                "n_seeds": len(cell.eces),
                "mean_ece": round(cell.mean_ece, 4),
                "std_ece": round(cell.std_ece, 4),
            }
            for (ds, m), cell in temp.items()
        },
    }
    (out_dir / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"wrote {out_dir / 'calibration_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
