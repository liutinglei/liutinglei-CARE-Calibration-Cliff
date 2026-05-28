"""Render the TOST equivalence summary as a LaTeX table fragment.

Reads ``paper/outputs/tost_equivalence.json`` and writes
``paper/outputs/tost_table.tex``.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "tost_equivalence.json"
OUT = ROOT / "outputs" / "tost_table.tex"


def main() -> int:
    if not SRC.is_file():
        OUT.write_text(
            "\\begin{tabular}{lccc}\n\\toprule\nPair & diff & 95\\% CI & "
            "Verdict ($\\Delta$=0.02) \\\\\n\\midrule\n"
            "\\multicolumn{4}{c}{\\emph{TOST output pending}}\\\\\n"
            "\\bottomrule\n\\end{tabular}\n",
            encoding="utf-8",
        )
        print(f"wrote placeholder {OUT}")
        return 0

    data = json.loads(SRC.read_text(encoding="utf-8"))
    rows: list[str] = []
    rows.append("\\begin{tabular}{lccc}")
    rows.append("\\toprule")
    rows.append(
        "Pair & $\\Delta$(Macro-F1) & 95\\% CI & "
        "TOST verdict ($\\Delta_{\\text{SESOI}}{=}0.02$) \\\\"
    )
    rows.append("\\midrule")
    verdict_pretty = {
        "equivalent": "\\textbf{equivalent}",
        "inconclusive": "inconclusive",
        "non-equivalent": "\\textbf{non-equivalent}",
    }
    for r in data["rows"]:
        diff = float(r["diff_macroF1"])
        ci = r["ci95"]
        rows.append(
            f"{r['label']} & {diff:+.4f} & "
            f"[{ci[0]:+.4f}, {ci[1]:+.4f}] & "
            f"{verdict_pretty.get(r['verdict'], r['verdict'])} \\\\"
        )
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    OUT.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
