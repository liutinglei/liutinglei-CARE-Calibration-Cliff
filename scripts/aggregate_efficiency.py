"""Aggregate efficiency measurement JSON into the LaTeX table fragment
used by ``Table~\\ref{tab:efficiency}``.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "efficiency.json"
OUT = ROOT / "outputs" / "efficiency_table.tex"


def fmt_params(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1e9:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.0f}M"
    return f"{n}"


def fmt_pct(p: float) -> str:
    return f"{p:.2f}\\%"


def main() -> int:
    if not SRC.is_file():
        OUT.write_text(
            "\\begin{tabular}{lrrrr}\n\\toprule\n"
            "Model & Params & Trainable & VRAM & Latency \\\\\n"
            "\\midrule\n"
            "\\multicolumn{5}{c}{\\emph{efficiency measurement pending}}\\\\\n"
            "\\bottomrule\n\\end{tabular}\n",
            encoding="utf-8",
        )
        print(f"wrote placeholder to {OUT}")
        return 0

    data = json.loads(SRC.read_text(encoding="utf-8"))

    rows: list[str] = []
    rows.append("\\begin{tabular}{lrrrr}")
    rows.append("\\toprule")
    rows.append("Model & Params & Trainable & VRAM (MB) & Latency (ms) \\\\")
    rows.append("\\midrule")
    for row in data:
        if "error" in row:
            rows.append(
                f"{row.get('name','?')} & "
                "\\multicolumn{4}{l}{\\emph{measurement failed}} \\\\"
            )
            continue
        name = row["name"]
        pt = int(row["params_total"])
        pp = int(row["params_trainable"])
        ppct = float(row.get("params_trainable_pct", 0.0))
        vram = row.get("vram_peak_mb_inference", 0.0)
        lat = row.get("latency_ms_per_utterance", 0.0)
        trainable_str = f"{fmt_params(pp)} ({fmt_pct(ppct)})"
        rows.append(
            f"{name} & {fmt_params(pt)} & {trainable_str} & "
            f"{vram:.0f} & {lat:.1f} \\\\"
        )
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    OUT.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
