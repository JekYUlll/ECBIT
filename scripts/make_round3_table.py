#!/usr/bin/env python3
"""Generate a held-out station generalization LaTeX table."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


RUNS_CSV = Path("experiments/results/tables/round3_final_runs.csv")
OUT_TEX = Path("paper/tables/tab_round3_heldout.tex")
STATION_RE = re.compile(r"ecbit_full_heldout_(.+)_(short|medium|long)_s\d+")


def station_name(raw: str) -> str:
    return raw.replace("_", " ").title()


def fmt(mean: float, std: float, count: int, bold: bool) -> str:
    value = f"{mean:.3f} $\\pm$ {std:.3f}"
    if bold:
        value = f"\\textbf{{{value}}}"
    return f"{value} ({count})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-csv", type=Path, default=RUNS_CSV)
    parser.add_argument("--out-tex", type=Path, default=OUT_TEX)
    args = parser.parse_args()

    runs = pd.read_csv(args.runs_csv)
    runs = runs.copy()
    runs["station"] = runs["run_name"].map(lambda s: STATION_RE.match(s).group(1))
    summary = runs.groupby("station")["mae_mean"].agg(["mean", "std", "count"]).sort_values("mean")
    best = float(summary["mean"].min())

    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Round 3 held-out station generalization at 40\\% block missingness. Values are mean test MAE across short, medium, and long block regimes and three seeds. Lower is better.}",
        "  \\label{tab:round3-heldout}",
        "  \\begin{tabular}{lc}",
        "    \\toprule",
        "    Held-out station & MAE \\\\",
        "    \\midrule",
    ]
    for station, row in summary.iterrows():
        lines.append(
            f"    {station_name(station)} & "
            + fmt(float(row["mean"]), float(row["std"]), int(row["count"]), float(row["mean"]) == best)
            + " \\\\"
        )
    lines.extend(["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""])
    args.out_tex.parent.mkdir(parents=True, exist_ok=True)
    args.out_tex.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out_tex}")


if __name__ == "__main__":
    main()
