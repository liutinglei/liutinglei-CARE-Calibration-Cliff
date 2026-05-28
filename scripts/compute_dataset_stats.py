"""Compute the full dataset-statistics table required by the new Section III."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(r"D:\5.13 Awesome论文skill\CYBERBULLYING数据集文章\PAPER2_CODEX")
src = pd.read_csv(ROOT / "cyberbullying-llm" / "data"
                  / "all_conversations_annotated_CBtypes.csv")
src["text"] = src["text"].fillna("").astype(str)
src["CB_types"] = src["CB_types"].fillna("None").astype(str)
src["conv_id"] = (src["model"].astype(str) + "__" +
                  src["scenario_id"].astype(str) + "__" +
                  src["conversation_num"].astype(str))


def stats(slice_df):
    s = {}
    s["n_utt"] = len(slice_df)
    s["n_conv"] = slice_df["conv_id"].nunique()
    s["pct_harm"] = 100.0 * (slice_df["harm_label"].astype(int).mean())
    # tokens
    tok_lens = slice_df["text"].str.split().str.len()
    s["avg_tok"] = float(tok_lens.mean())
    s["med_tok"] = float(tok_lens.median())
    s["max_tok"] = int(tok_lens.max())
    # flags
    for fl in ["is_sarcastic", "is_humorous", "is_hate_speech"]:
        if fl in slice_df.columns:
            ser = slice_df[fl].astype(str).str.lower()
            s[f"pct_{fl}"] = 100.0 * (ser == "yes").mean()
    # source mix
    if "model" in slice_df.columns:
        for src_name in ["gpt-4o", "grok", "llama", "whatsapp"]:
            mask = slice_df["model"] == src_name
            s[f"n_{src_name}"] = int(mask.sum())
    return s


# load strict manifest
manifest = json.loads((ROOT / "paper" / "outputs" / "strict_split_manifest.json").read_text(encoding="utf-8"))
splits = manifest["splits"]

reports = {"all": stats(src)}
for split_name in ["train", "dev", "test"]:
    ids = set(splits[split_name])
    df = src[src["conv_id"].isin(ids)]
    reports[f"strict_{split_name}"] = stats(df)
# also the whatsapp authentic subset of test
auth_ids = [c for c in splits["test"] if c.startswith("whatsapp")]
df_auth = src[src["conv_id"].isin(set(auth_ids))]
reports["whatsapp_auth_test"] = stats(df_auth)

# OLID (loaded directly from jsonl)
olid_test = ROOT / "cyberbullying-llm" / "data" / "olid_test.jsonl"
if olid_test.exists():
    rows = [json.loads(l) for l in olid_test.open(encoding="utf-8")]
    olid_df = pd.DataFrame(rows)
    olid_df["text"] = olid_df["input"].str.replace("Speaker1: ", "", regex=False)
    olid_stat = {
        "n_utt": len(olid_df),
        "n_conv": "n/a",
        "pct_harm": 100.0 * olid_df["label"].astype(int).mean(),
        "avg_tok": olid_df["text"].str.split().str.len().mean(),
        "med_tok": olid_df["text"].str.split().str.len().median(),
        "max_tok": int(olid_df["text"].str.split().str.len().max()),
    }
    reports["olid_test"] = olid_stat


out = ROOT / "paper" / "outputs" / "dataset_stats.json"
out.write_text(json.dumps(reports, indent=2), encoding="utf-8")
print(f"wrote {out}")

# pretty print
for k, v in reports.items():
    print(f"\n=== {k} ===")
    for kk, vv in v.items():
        if isinstance(vv, float):
            print(f"  {kk:20s} {vv:10.2f}")
        else:
            print(f"  {kk:20s} {vv}")
