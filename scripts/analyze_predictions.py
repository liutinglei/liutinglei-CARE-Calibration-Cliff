#!/usr/bin/env python
"""Slice prediction CSVs by data source (gpt-4o/grok/llama/whatsapp), by
cyberbullying type, and by linguistic-style flags (sarcasm/hate-speech/...).

Inputs:
  --pred           predictions.csv produced by train_encoder_baseline.py or
                   train_llm_lora.py.
  --window         which strict split the predictions came from (0/1/2/4/8).
  --src_csv        original SynBullying CSV with model/scenario_id/...
  --manifest       strict_split_manifest.json (for test conv_ids)

Outputs:
  --out_json       structured metric breakdown per slice.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix)


def metrics_for_slice(labels, preds):
    if len(labels) == 0:
        return None
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], average=None, zero_division=0
    )
    pm, rm, f1m, _ = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], average="macro", zero_division=0
    )
    cm = confusion_matrix(labels, preds, labels=[0, 1]).tolist()
    return {
        "n": int(len(labels)),
        "accuracy": float(acc),
        "macro_f1": float(f1m),
        "not_harmful_f1": float(f1[0]),
        "harmful_f1": float(f1[1]),
        "harmful_precision": float(p[1]),
        "harmful_recall": float(r[1]),
        "confusion_matrix": cm,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--window", type=int, required=True)
    ap.add_argument("--src_csv",
                    default="<WORK_ROOT>/datasets/SynBullying/all_conversations_annotated_CBtypes.csv")
    ap.add_argument("--manifest",
                    default="<DATA_ROOT>/data/strict_split_manifest.json")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    src = pd.read_csv(args.src_csv)
    src["text"] = src["text"].fillna("").astype(str)
    src["CB_types"] = src["CB_types"].fillna("None").astype(str)
    src["conv_id"] = (src["model"].astype(str) + "__" +
                      src["scenario_id"].astype(str) + "__" +
                      src["conversation_num"].astype(str))
    src = src.sort_values(["model", "scenario_id", "conversation_num", "sentence_num"]).reset_index(drop=True)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    test_conv_ids = list(manifest["splits"]["test"])
    src_test = src[src["conv_id"].isin(test_conv_ids)].reset_index(drop=True)
    # the iteration order must match the JSONL generation order
    # which is groupby("conv_id", sort=False) over the sorted df
    # this means rows appear by sorted order in the test conv_id subset
    # this is already what src_test gives us

    pred_df = pd.read_csv(args.pred)
    # We assume 1:1 alignment by row order with src_test
    if len(pred_df) != len(src_test):
        raise SystemExit(f"len(pred)={len(pred_df)} != len(test source)={len(src_test)}")

    src_test = src_test.copy()
    src_test["pred_label"] = pred_df["pred"].map({"harmful": 1, "not_harmful": 0})
    src_test["true_label"] = pred_df["label"].map({"harmful": 1, "not_harmful": 0})

    results = {"window": args.window, "pred_file": args.pred,
               "overall": metrics_for_slice(src_test["true_label"].tolist(),
                                            src_test["pred_label"].tolist()),
               "by_source": {}, "by_cb_type": {}, "by_flag": {}}

    # by data source
    for src_name in sorted(src_test["model"].unique()):
        sub = src_test[src_test["model"] == src_name]
        results["by_source"][src_name] = metrics_for_slice(
            sub["true_label"].tolist(), sub["pred_label"].tolist())

    # by CB_types (split multi-label by comma)
    def explode_cb(row):
        cbs = row.CB_types
        if cbs in ("None", "", None):
            return ["None"]
        return [c.strip() for c in str(cbs).split(",") if c.strip()]

    by_cb = {}
    for _, r in src_test.iterrows():
        for cb in explode_cb(r):
            by_cb.setdefault(cb, {"labels": [], "preds": []})
            by_cb[cb]["labels"].append(int(r.true_label))
            by_cb[cb]["preds"].append(int(r.pred_label))
    for cb, d in by_cb.items():
        results["by_cb_type"][cb] = metrics_for_slice(d["labels"], d["preds"])

    # by flag (is_sarcastic, is_humorous, is_hate_speech). The SynBullying CSV
    # stores these as 'yes'/'no' strings, so normalize first.
    def _normalize_flag(v):
        if isinstance(v, bool):
            return 1 if v else 0
        if isinstance(v, (int, float)):
            return int(v == 1)
        if isinstance(v, str):
            return 1 if v.strip().lower() in {"yes", "true", "1"} else 0
        return 0

    for flag in ["is_sarcastic", "is_humorous", "is_hate_speech"]:
        if flag not in src_test.columns:
            continue
        norm = src_test[flag].map(_normalize_flag)
        for v in [0, 1]:
            sub = src_test[norm == v]
            key = f"{flag}={v}"
            results["by_flag"][key] = metrics_for_slice(
                sub["true_label"].tolist(), sub["pred_label"].tolist())

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out_json}")
    # short summary
    print(json.dumps({"overall_macro_f1": results["overall"]["macro_f1"],
                      "by_source": {k: v["macro_f1"] for k, v in results["by_source"].items() if v},
                      "by_flag": {k: v["macro_f1"] for k, v in results["by_flag"].items() if v}},
                     indent=2))


if __name__ == "__main__":
    main()
