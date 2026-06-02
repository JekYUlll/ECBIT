#!/usr/bin/env python3
"""Generate a Round 2 ECBIT/gated ablation LaTeX table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RUNS_CSV = Path("experiments/results/tables/round2_gated_final_runs.csv")
OUT_TEX = Path("paper/tables/tab_round2_ablations.tex")
VARIANT_ORDER = ["full", "no_cross", "no_era5", "no_blockmask"]
VARIANT_LABELS = {
    "full": "Gated ERA5 injection",
    "no_cross": "Concat/no-cross fusion",
    "no_era5": "No ERA5",
    "no_blockmask": "MCAR training",
}


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
    block_runs = runs[runs["pattern"].ne("mcar")].copy()
    mcar_runs = runs[runs["pattern"].eq("mcar")].copy()

    block_summary = (
        block_runs.groupby("variant")["mae_mean"]
        .agg(["mean", "std", "count"])
        .reindex([v for v in VARIANT_ORDER if v != "no_blockmask"])
    )
    best_block = block_summary["mean"].min()

    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Round 2 ERA5-conditioning ablation results. Block rows use block-missing test masks; the MCAR row is an in-distribution MCAR-test reference and differs from Table~\\ref{tab:curriculum-era5-factorial}'s MCAR-on-block rows. Lower is better.}",
        "  \\label{tab:round2-ablations}",
        "  \\begin{tabular}{lc}",
        "    \\toprule",
        "    Variant & Test MAE \\\\",
        "    \\midrule",
    ]
    for variant in [v for v in VARIANT_ORDER if v != "no_blockmask"]:
        if variant not in block_summary.index or pd.isna(block_summary.loc[variant, "mean"]):
            cell = "--"
        else:
            row = block_summary.loc[variant]
            cell = fmt(
                float(row["mean"]),
                float(row["std"]) if not pd.isna(row["std"]) else 0.0,
                int(row["count"]),
                float(row["mean"]) == float(best_block),
            )
        lines.append(f"    {VARIANT_LABELS[variant]} & {cell} \\\\")

    if not mcar_runs.empty:
        mcar = mcar_runs["mae_mean"].agg(["mean", "std", "count"])
        lines.extend(
            [
                "    \\midrule",
                f"    {VARIANT_LABELS['no_blockmask']} & "
                + fmt(float(mcar["mean"]), float(mcar["std"]), int(mcar["count"]), False)
                + " \\\\",
            ]
        )
    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    args.out_tex.parent.mkdir(parents=True, exist_ok=True)
    args.out_tex.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out_tex}")


if __name__ == "__main__":
    main()
