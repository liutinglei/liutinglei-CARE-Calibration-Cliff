#!/usr/bin/env python
"""Run cross-source and linguistic-flag analysis on every baseline prediction
CSV, aggregate across seeds, write a master report + per-model breakdown.
"""
import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)


def metrics(labels, preds):
    if len(labels) == 0:
        return None
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], average=None, zero_division=0
    )
    pm, rm, f1m, _ = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], average="macro", zero_division=0
    )
    return {
        "n": int(len(labels)),
        "acc": float(acc),
        "macro_f1": float(f1m),
        "harm_f1": float(f1[1]),
        "harm_p": float(p[1]),
        "harm_r": float(r[1]),
        "not_harm_f1": float(f1[0]),
    }


def normalize_flag(v):
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, (int, float)):
        return int(v == 1)
    if isinstance(v, str):
        return 1 if v.strip().lower() in {"yes", "true", "1"} else 0
    return 0


def parse_tag(name):
    """encoder pred name: <short>_w<W>[_extra]_seed<S>_predictions.csv"""
    m = re.match(r"^(.+)_w(\d+)(?:_([a-z][a-z0-9]*))?_seed(\d+)_predictions\.csv$",
                 name)
    if not m:
        return None
    model = m.group(1)
    if m.group(3):
        model = f"{model}_{m.group(3)}"
    return {"model": model, "window": int(m.group(2)),
            "seed": int(m.group(4))}


def aggregate(seed_rows):
    """seed_rows: list of dict with {acc, macro_f1, harm_f1, ...}.
    Return mean ± std for each metric."""
    if not seed_rows:
        return None
    out = {"n": seed_rows[0]["n"], "seeds": len(seed_rows)}
    for k in ["acc", "macro_f1", "harm_f1", "harm_p", "harm_r", "not_harm_f1"]:
        vals = [r[k] for r in seed_rows if r and k in r]
        if not vals:
            continue
        mu = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        out[k] = {"mean": mu, "std": sd}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True, nargs="+",
                    help="One or more directories with *_predictions.csv")
    ap.add_argument("--src_csv",
                    default="<WORK_ROOT>/datasets/SynBullying/all_conversations_annotated_CBtypes.csv")
    ap.add_argument("--manifest",
                    default="<DATA_ROOT>/data/strict_split_manifest.json")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load source data and align to strict test ordering
    src = pd.read_csv(args.src_csv)
    src["text"] = src["text"].fillna("").astype(str)
    src["CB_types"] = src["CB_types"].fillna("None").astype(str)
    src["conv_id"] = (src["model"].astype(str) + "__" +
                      src["scenario_id"].astype(str) + "__" +
                      src["conversation_num"].astype(str))
    src = src.sort_values(["model", "scenario_id", "conversation_num",
                           "sentence_num"]).reset_index(drop=True)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    test_ids = set(manifest["splits"]["test"])
    src_test = src[src["conv_id"].isin(test_ids)].reset_index(drop=True)

    # Collect per-run slice metrics (across all provided pred_dirs)
    pred_paths = []
    for d in args.pred_dir:
        pred_paths.extend(sorted(Path(d).glob("*_predictions.csv")))
    rows = []
    for p in pred_paths:
        tag = parse_tag(p.name)
        if tag is None:
            continue
        pdf = pd.read_csv(p)
        if len(pdf) != len(src_test):
            print(f"skip {p.name}: row mismatch "
                  f"({len(pdf)} vs {len(src_test)})")
            continue
        labels = pdf["label"].map({"harmful": 1, "not_harmful": 0}).tolist()
        preds = pdf["pred"].map({"harmful": 1, "not_harmful": 0}).tolist()

        overall = metrics(labels, preds)

        # by source
        by_source = {}
        for src_name in sorted(src_test["model"].unique()):
            mask = (src_test["model"] == src_name).tolist()
            ll = [l for l, m in zip(labels, mask) if m]
            pp = [pr for pr, m in zip(preds, mask) if m]
            by_source[src_name] = metrics(ll, pp)

        # by flag
        by_flag = {}
        for flag in ["is_sarcastic", "is_humorous", "is_hate_speech"]:
            if flag not in src_test.columns:
                continue
            norm = src_test[flag].map(normalize_flag).tolist()
            for v in [0, 1]:
                mask = [x == v for x in norm]
                ll = [l for l, m in zip(labels, mask) if m]
                pp = [pr for pr, m in zip(preds, mask) if m]
                by_flag[f"{flag}={v}"] = metrics(ll, pp)

        # by CB_type
        by_cb = {}
        cb_lists = src_test["CB_types"].fillna("None").astype(str).tolist()
        cb_keys = set()
        for c in cb_lists:
            for t in str(c).split(","):
                t = t.strip() or "None"
                cb_keys.add(t)
        for k in cb_keys:
            mask = [k in str(c) or (k == "None" and str(c) in ("None", ""))
                    for c in cb_lists]
            ll = [l for l, m in zip(labels, mask) if m]
            pp = [pr for pr, m in zip(preds, mask) if m]
            if len(ll) >= 8:
                by_cb[k] = metrics(ll, pp)

        rows.append({"tag": p.stem.replace("_predictions", ""), **tag,
                     "overall": overall, "by_source": by_source,
                     "by_flag": by_flag, "by_cb": by_cb})

    # Aggregate by (model, window) across seeds
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["model"], r["window"])].append(r)

    summary = {"by_model_window": {}, "rows": rows}
    for (model, w), rs in sorted(grouped.items()):
        key = f"{model}_w{w}"
        agg = {"seeds": len(rs)}
        # overall agg
        agg["overall"] = aggregate([r["overall"] for r in rs])
        # by_source
        sources = set()
        for r in rs:
            sources.update(r["by_source"].keys())
        agg["by_source"] = {
            s: aggregate([r["by_source"].get(s) for r in rs
                          if r["by_source"].get(s)])
            for s in sources
        }
        # by_flag
        flags = set()
        for r in rs:
            flags.update(r["by_flag"].keys())
        agg["by_flag"] = {
            f: aggregate([r["by_flag"].get(f) for r in rs
                          if r["by_flag"].get(f)])
            for f in flags
        }
        # by_cb
        cbs = set()
        for r in rs:
            cbs.update(r["by_cb"].keys())
        agg["by_cb"] = {
            c: aggregate([r["by_cb"].get(c) for r in rs
                          if r["by_cb"].get(c)])
            for c in cbs
        }
        summary["by_model_window"][key] = agg

    (out_dir / "analysis_all.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Print human-readable summary
    print("=" * 100)
    print(f"OVERALL — mean macro_f1 ± std over seeds")
    print(f"{'tag':30s}  {'seeds':>5}  {'macro_f1':>14}  {'harm_f1':>14}  {'acc':>14}")
    for key, agg in sorted(summary["by_model_window"].items()):
        ov = agg["overall"]
        if not ov:
            continue
        def fmt(field):
            d = ov.get(field, None)
            return f"{d['mean']:.4f}±{d['std']:.4f}" if d else "----"
        print(f"{key:30s}  {agg['seeds']:>5}  "
              f"{fmt('macro_f1'):>14}  {fmt('harm_f1'):>14}  {fmt('acc'):>14}")
    print()
    print(f"BY SOURCE (macro_f1)")
    headers = sorted({s for agg in summary["by_model_window"].values()
                      for s in agg["by_source"].keys()})
    print(f"{'tag':30s}  " + "  ".join(f"{h[:11]:>11}" for h in headers))
    for key, agg in sorted(summary["by_model_window"].items()):
        cells = []
        for h in headers:
            d = agg["by_source"].get(h, {})
            if d and "macro_f1" in d:
                cells.append(f"{d['macro_f1']['mean']:.3f}")
            else:
                cells.append("---")
        print(f"{key:30s}  " + "  ".join(f"{c:>11}" for c in cells))
    print()
    print(f"BY LINGUISTIC FLAG (macro_f1) — only RoBERTa w4 shown")
    rb = summary["by_model_window"].get("roberta_base_w4", {})
    for fk, fv in sorted(rb.get("by_flag", {}).items()):
        if fv and "macro_f1" in fv:
            print(f"  {fk:25s}  {fv['macro_f1']['mean']:.3f} ± {fv['macro_f1']['std']:.4f}")

    print(f"\nWrote {out_dir/'analysis_all.json'}")


if __name__ == "__main__":
    main()
