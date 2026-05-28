"""Local post-processing after the remote `run_calibration_v2.sh` forward
pipeline has completed and the predictions CSVs have been synced back to
``paper/outputs/calibration_v2/preds/``.

Pipeline:
  1. For every predictions CSV, compute raw 15-bin ECE / MCE / Brier.
  2. For every (model, OOD-target), learn a temperature on the matching
     SynBullying dev forward, apply it to the OOD predictions, recompute
     the temperature-scaled ECE.
  3. Run aggregate_calibration_xdom.py to produce the LaTeX fragments.

The script is idempotent: it skips outputs already on disk and prints
``SKIP`` / ``DONE`` per cell.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "outputs" / "calibration_v2" / "preds"
ECE_DIR = ROOT / "outputs" / "calibration_v2" / "ece"
ECE_TEMP_DIR = ROOT / "outputs" / "calibration_v2" / "ece_temp"
TEMP_DIR = ROOT / "outputs" / "calibration_v2" / "temp"
PRED_TEMP_DIR = ROOT / "outputs" / "calibration_v2" / "preds_temp"

OOD_DATASETS = ("civilcomments", "olid", "hatexplain")
ALL_DATASETS = ("synb_w4_dev", "synb_w4_test") + OOD_DATASETS

ENCODERS = ("bert_base", "roberta_base", "hatebert", "deberta_v3")
LLMS = ("internlm2_5_7b", "qwen2_5_7b")
ALL_MODELS = ENCODERS + LLMS
SEEDS = (41, 42, 43)

SCRIPTS = ROOT / "scripts"
COMPUTE_ECE = SCRIPTS / "compute_ece_proper.py"
TEMP_SCALE = SCRIPTS / "temperature_scaling.py"
AGGREGATE = SCRIPTS / "aggregate_calibration_xdom.py"


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd))
    cp = subprocess.run(cmd, capture_output=True, text=True)
    if cp.returncode != 0:
        sys.stdout.write(cp.stdout)
        sys.stderr.write(cp.stderr)
        raise SystemExit(cp.returncode)
    sys.stdout.write(cp.stdout)


def pred_path(dataset: str, model: str, seed: int) -> Path:
    return PRED_DIR / f"{dataset}__{model}__seed{seed}.csv"


def phase1_raw_ece() -> None:
    ECE_DIR.mkdir(parents=True, exist_ok=True)
    for dataset in ALL_DATASETS:
        for model in ALL_MODELS:
            for seed in SEEDS:
                src = pred_path(dataset, model, seed)
                if not src.is_file():
                    print(f"MISS preds {src.name}")
                    continue
                out = ECE_DIR / f"{dataset}__{model}__seed{seed}.json"
                if out.is_file():
                    print(f"SKIP raw_ece {out.name}")
                    continue
                run([
                    sys.executable, str(COMPUTE_ECE),
                    "--predictions_csv", str(src),
                    "--out_json", str(out),
                    "--n_bins", "15",
                ])


def phase2_temperature_scaling() -> None:
    PRED_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    ECE_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    for model in ALL_MODELS:
        for seed in SEEDS:
            val_csv = pred_path("synb_w4_dev", model, seed)
            if not val_csv.is_file():
                print(f"MISS val preds {val_csv.name}")
                continue
            for dataset in OOD_DATASETS:
                tgt_csv = pred_path(dataset, model, seed)
                if not tgt_csv.is_file():
                    print(f"MISS target preds {tgt_csv.name}")
                    continue
                new_csv = PRED_TEMP_DIR / f"{dataset}__{model}__seed{seed}.csv"
                summary_json = TEMP_DIR / f"{dataset}__{model}__seed{seed}.json"
                ece_temp_json = (
                    ECE_TEMP_DIR / f"{dataset}__{model}__seed{seed}.json"
                )
                if new_csv.is_file() and summary_json.is_file():
                    print(f"SKIP temp scale {new_csv.name}")
                else:
                    run([
                        sys.executable, str(TEMP_SCALE),
                        "--val_csv", str(val_csv),
                        "--target_csv", str(tgt_csv),
                        "--out_csv", str(new_csv),
                        "--summary_json", str(summary_json),
                    ])
                if ece_temp_json.is_file():
                    print(f"SKIP temp_ece {ece_temp_json.name}")
                    continue
                run([
                    sys.executable, str(COMPUTE_ECE),
                    "--predictions_csv", str(new_csv),
                    "--out_json", str(ece_temp_json),
                    "--n_bins", "15",
                ])


def phase3_aggregate() -> None:
    run([
        sys.executable, str(AGGREGATE),
        "--root", str(ROOT / "outputs" / "calibration_v2"),
        "--out_dir", str(ROOT / "outputs"),
    ])


def phase4_copy_to_latex() -> None:
    """Copy aggregator outputs into latex/outputs/ for hermetic builds."""
    src_dir = ROOT / "outputs"
    dst_dir = ROOT / "latex" / "outputs"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "calibration_v2_table.tex",
        "calibration_tempscaled_table.tex",
        "calibration_summary.json",
    ):
        src = src_dir / name
        if not src.is_file():
            print(f"skipped (missing): {name}")
            continue
        shutil.copy2(src, dst_dir / name)
        print(f"copied: {name} -> latex/outputs/")


def main() -> int:
    print("=== Phase 1: raw ECE ===")
    phase1_raw_ece()
    print("=== Phase 2: temperature scaling + scaled ECE ===")
    phase2_temperature_scaling()
    print("=== Phase 3: aggregate LaTeX fragments ===")
    phase3_aggregate()
    print("=== Phase 4: copy fragments into latex/outputs ===")
    phase4_copy_to_latex()
    print(
        "\nDONE. Inspect:\n"
        f"  {ROOT / 'outputs' / 'calibration_summary.json'}\n"
        f"  {ROOT / 'outputs' / 'calibration_v2_table.tex'}\n"
        f"  {ROOT / 'outputs' / 'calibration_tempscaled_table.tex'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
