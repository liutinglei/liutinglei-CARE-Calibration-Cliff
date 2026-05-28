# CARE: A Calibration-Aware Framework for Cyberbullying Detection Under Distribution Shift

Reproducibility kit accompanying the manuscript submitted to **IEEE Access** (2026).

> **📖 Citation Notice.** If you use any code, manifest, or calibration
> artefact in this repository — including the per-system prediction CSVs
> and the SHA-256-pinned summary artefact — please cite the accompanying
> paper. GitHub auto-renders a *"Cite this repository"* button in the
> sidebar from `CITATION.cff`; bibliographic details for the published
> version will be added there on acceptance. The full author list and
> affiliations are in `CITATION.cff`.

---

## What this repository contains

```
care_anon_repo/
├── scripts/                              31 Python scripts
│   ├── bootstrap_significance_cluster.py    cluster bootstrap + Holm--Bonferroni
│   ├── compute_calibration.py               15-bin Expected Calibration Error
│   ├── compute_ece_proper.py                ECE + Brier per system
│   ├── tost_equivalence.py                  TOST equivalence verdict
│   ├── temperature_scaling.py               post-hoc T-scaling rescue
│   ├── train_encoder_baseline.py            BERT / RoBERTa / HateBERT / DeBERTa fine-tune
│   ├── train_llm_lora.py                    InternLM / Qwen LoRA adaptation
│   ├── eval_*.py                            evaluation across in-domain + 3 OOD targets
│   ├── aggregate_*.py                       table generation scripts
│   └── ...
├── manifests/                            3 JSON files
│   ├── strict_split_manifest.json           strict conversation-level split
│   ├── test_conv_ids.json                   the 13 held-out test conversations
│   └── olid_manifest.json                   OLID OOD evaluation manifest
└── outputs/                              ~75 MB, 685 files
    ├── calibration_v2/preds/                per-input probabilities (encoders + LoRA LLMs)
    ├── calibration_v2/preds_temp/           same after temperature scaling
    ├── calibration_v2/ece/                  per-(model, dataset, seed) ECE breakdowns
    ├── baselines/*.csv                      per-input encoder predictions
    ├── llm/*.json                           LLM eval metrics
    ├── bootstrap_cluster/                   cluster-bootstrap intermediate output
    ├── groupkfold/, civilcomments/,
    │   hatexplain/, olid/                   subset-specific results
    ├── calibration_summary.json             top-level summary anchor
    └── *.json                               per-table summary outputs
```

The **trained LoRA adapter weights** and a **frozen `environment.yml`** will be released
on a public GitHub repository on paper acceptance.

---

## Reproducing the headline numbers

The Calibration Cliff (14.86× worst-vs-best on Civil Comments) can be recomputed end-to-end
from this repository without re-running any training.

### 1. Set up environment

```bash
pip install -r requirements.txt
```

### 2. Recompute the calibration cliff ratio

```bash
python scripts/compute_ece_proper.py \
    --preds_dir outputs/calibration_v2/preds \
    --output_dir outputs/calibration_v2/ece_recompute
python scripts/aggregate_calibration_xdom.py \
    --ece_dir outputs/calibration_v2/ece_recompute \
    --output_table outputs/calibration_recompute_table.tex
```

Expected output (matching Table 9 in the manuscript): family-mean OOD ECE
encoder panel = 0.140, LLM panel = 0.041.

### 3. Recompute cluster-bootstrap significance

```bash
python scripts/bootstrap_significance_cluster.py \
    --predictions_dir outputs/baselines \
    --conv_ids manifests/test_conv_ids.json \
    --n_resamples 10000 \
    --output outputs/bootstrap_cluster_recompute.json
```

Expected output (matching Table 1): all pairwise Holm-adjusted p-values = 1.00.

### 4. Recompute temperature-scaling residual cliff

```bash
python scripts/temperature_scaling.py \
    --preds_dir outputs/calibration_v2/preds \
    --temp_dir outputs/calibration_v2/temp \
    --output_dir outputs/calibration_v2/preds_temp_recompute
```

Expected output (matching Table 10): worst-vs-best Cliff drops from 14.86× to 10.92×
after T-scaling, with 9.36× residual against un-rescued LLM family.

---

## Anchors

The following SHA-256 anchors freeze the artefacts in this repository at submission time.

```
calibration_summary.json     f4cb1115391e3fe5ba791d64fda5735b22cc8193727eb6745f82617489c78f24
outputs+manifests rollup     40d39f9d24259d86633162600144444cd6ef0f397ae50af19f73080c01bb5683
```

If `sha256sum outputs/calibration_summary.json` matches the first line above,
the calibration outputs are bit-identical to the submitted manuscript.

---

## Data access

| Dataset | Source |
|---|---|
| SynBullying corpus | available from the original release under its access agreement |
| OLID / OffensEval-2019 | `tweet_eval` `offensive` subset (HuggingFace Datasets) |
| HateXplain | original release, 2-of-3 majority on `offensive`∪`hatespeech` |
| Civil Comments / Jigsaw | original release, toxicity threshold ≥ 0.5 |

This repository does not redistribute the raw datasets.
`manifests/test_conv_ids.json` lists the 13 held-out conversation IDs used as the
strict-protocol test set, sufficient to reconstruct the partition once the source
corpus is in hand.

---

## License

This repository is released under the MIT License for the purpose of double-blind
peer review (see `LICENSE`). The permanent post-acceptance release may adopt a
different licence appropriate to the venue.
