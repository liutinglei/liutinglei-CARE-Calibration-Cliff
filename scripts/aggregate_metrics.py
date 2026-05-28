#!/usr/bin/env python
"""Aggregate metric JSONs into a comparison table.

Usage:
  python aggregate_metrics.py /path/to/outputs/baselines [--main_strict]
"""
import argparse
import json
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path


def fmt(x, p=4):
    if x is None:
        return "----"
    return f"{x:.{p}f}"


def short(name: str) -> str:
    n = os.path.basename(name).replace("_metrics.json", "")
    return n


def parse_id(name: str):
    """name pattern: <model>_w<W>[_extra]_seed<S>"""
    m = re.match(r"^(.+)_w(\d+)(?:_([a-z][a-z0-9]*))?_seed(\d+)$", name)
    if not m:
        return name, None, None
    model = m.group(1)
    if m.group(3):
        model = f"{model}_{m.group(3)}"
    return model, int(m.group(2)), int(m.group(4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="+", help="dir(s) with *_metrics.json")
    ap.add_argument("--out", default=None, help="optional CSV out path")
    args = ap.parse_args()

    rows = []
    for d in args.dir:
        for p in sorted(Path(d).glob("*_metrics.json")):
            try:
                m = json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"skip {p}: {e}")
                continue
            sid = short(p.name)
            model, w, seed = parse_id(sid)
            rows.append({
                "file": sid,
                "model": model,
                "window": w,
                "seed": seed,
                "acc": m.get("accuracy"),
                "macro_f1": m.get("macro_f1"),
                "harm_f1": m.get("harmful_f1"),
                "harm_p": m.get("harmful_precision"),
                "harm_r": m.get("harmful_recall"),
            })

    # raw
    print(f"{'tag':50s}  {'w':>2}  {'sd':>3}  {'acc':>7}  {'mF1':>7}  {'hF1':>7}  {'hP':>7}  {'hR':>7}")
    for r in rows:
        print(f"{r['file']:50s}  {str(r['window']):>2}  {str(r['seed']):>3}  "
              f"{fmt(r['acc']):>7}  {fmt(r['macro_f1']):>7}  "
              f"{fmt(r['harm_f1']):>7}  {fmt(r['harm_p']):>7}  {fmt(r['harm_r']):>7}")

    # grouped mean +- std
    groups = defaultdict(list)
    for r in rows:
        if r['model'] is None or r['window'] is None:
            continue
        groups[(r['model'], r['window'])].append(r)

    print("\n=== Mean +- Std over seeds ===")
    print(f"{'model':30s}  {'win':>3}  {'n':>2}  {'macro_f1':>14}  {'harm_f1':>14}  {'acc':>14}")
    for (model, w), rs in sorted(groups.items()):
        mfs = [r['macro_f1'] for r in rs if r['macro_f1'] is not None]
        hfs = [r['harm_f1'] for r in rs if r['harm_f1'] is not None]
        accs = [r['acc'] for r in rs if r['acc'] is not None]
        def ms(xs):
            if len(xs) == 0:
                return "----"
            mu = statistics.mean(xs)
            sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
            return f"{mu:.4f}±{sd:.4f}"
        print(f"{model:30s}  {str(w):>3}  {len(rs):>2}  {ms(mfs):>14}  {ms(hfs):>14}  {ms(accs):>14}")

    if args.out:
        import csv as _csv
        with open(args.out, "w", encoding="utf-8", newline="") as fh:
            wr = _csv.writer(fh)
            wr.writerow(["file", "model", "window", "seed", "accuracy",
                         "macro_f1", "harmful_f1", "harmful_p", "harmful_r"])
            for r in rows:
                wr.writerow([r['file'], r['model'], r['window'], r['seed'],
                             r['acc'], r['macro_f1'], r['harm_f1'],
                             r['harm_p'], r['harm_r']])


if __name__ == "__main__":
    main()
