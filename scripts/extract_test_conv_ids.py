"""Re-run the strict-split pipeline locally and emit the per-row
``conv_id`` vector for each window/test split, plus an aggregated
``test_conv_ids.json`` (window-invariant because the sort+groupby+filter
order is identical across windows).

This script is intentionally a faithful replay of
``prepare_strict_splits.py`` and ``prepare_strict_windows_extra.py``;
it does NOT re-train anything, it just recovers the lost ``conv_id``
metadata that was stripped when writing JSONL.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_CSV = REPO_ROOT / "cyberbullying-llm" / "data" / "all_conversations_annotated_CBtypes.csv"
OUT_DIR = REPO_ROOT / "paper" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
WINDOWS: tuple[int, ...] = (0, 1, 2, 4, 8)


def split_conv_ids(conv_ids: list[str]) -> dict[str, set[str]]:
    train_ids, temp_ids = train_test_split(
        conv_ids, test_size=0.2, random_state=RANDOM_STATE
    )
    dev_ids, test_ids = train_test_split(
        temp_ids, test_size=0.5, random_state=RANDOM_STATE
    )
    return {"train": set(train_ids), "dev": set(dev_ids), "test": set(test_ids)}


def _flag(value: object) -> int:
    if isinstance(value, (int, float)):
        return int(bool(value))
    if value is None:
        return 0
    text = str(value).strip().lower()
    return 1 if text in {"1", "yes", "true", "y"} else 0


def iter_rows(df: pd.DataFrame, ids: Iterable[str]) -> list[dict[str, object]]:
    ids_set = set(ids)
    rows: list[dict[str, object]] = []
    sorted_df = df.sort_values(
        ["model", "scenario_id", "conversation_num", "sentence_num"]
    )
    groups = sorted_df.groupby("conv_id", sort=False)
    for conv_id, group in groups:
        if conv_id not in ids_set:
            continue
        for _, row in group.reset_index(drop=True).iterrows():
            rows.append(
                {
                    "conv_id": conv_id,
                    "label": int(row.harm_label),
                    "is_sarcastic": _flag(row.get("is_sarcastic", 0)),
                    "is_humorous": _flag(row.get("is_humorous", 0)),
                    "is_hate_speech": _flag(row.get("is_hate_speech", 0)),
                    "source": str(row.model),
                }
            )
    return rows


def main() -> None:
    df = pd.read_csv(SRC_CSV)
    df["text"] = df["text"].fillna("").astype(str)
    df["CB_types"] = df["CB_types"].fillna("None").astype(str)
    df["conv_id"] = (
        df["model"].astype(str)
        + "__"
        + df["scenario_id"].astype(str)
        + "__"
        + df["conversation_num"].astype(str)
    )

    conv_ids_all = sorted(df["conv_id"].unique())
    splits = split_conv_ids(conv_ids_all)

    out: dict[str, object] = {
        "random_state": RANDOM_STATE,
        "num_conversations": len(conv_ids_all),
        "windows": list(WINDOWS),
        "splits": {k: sorted(v) for k, v in splits.items()},
    }

    for split_name, ids in splits.items():
        rows = iter_rows(df, ids)
        out[f"{split_name}_n"] = len(rows)
        out[f"{split_name}_conv_ids"] = [r["conv_id"] for r in rows]
        out[f"{split_name}_labels"] = [r["label"] for r in rows]
        out[f"{split_name}_is_sarcastic"] = [r["is_sarcastic"] for r in rows]
        out[f"{split_name}_is_humorous"] = [r["is_humorous"] for r in rows]
        out[f"{split_name}_is_hate_speech"] = [r["is_hate_speech"] for r in rows]
        out[f"{split_name}_source"] = [r["source"] for r in rows]

    out_path = OUT_DIR / "test_conv_ids.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_path}")
    print(
        f"train={out['train_n']}  dev={out['dev_n']}  test={out['test_n']}  "
        f"unique_test_convs={len(set(out['test_conv_ids']))}"  # type: ignore[arg-type]
    )


if __name__ == "__main__":
    main()
