#!/usr/bin/env python3
"""Generate the Round 1 baseline LaTeX table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RUNS_CSV = Path("experiments/results/tables/round1_core_runs.csv")
OUT_TEX = Path("paper/tables/tab_round1_baselines.tex")
MODEL_ORDER = ["linear_interp", "locf", "era5_direct", "saits", "itransformer"]
MODEL_LABELS = {
    "linear_interp": "Linear interpolation",
    "locf": "LOCF",
    "era5_direct": "ERA5 direct",
    "saits": "SAITS",
    "itransformer": "iTransformer",
}


def fmt(mean: float, std: float, bold: bool) -> str:
    text = f"{mean:.3f} $\\pm$ {std:.3f}"
    return f"\\textbf{{{text}}}" if bold else text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-csv", type=Path, default=RUNS_CSV)
    parser.add_argument("--out-tex", type=Path, default=OUT_TEX)
    args = parser.parse_args()

    runs = pd.read_csv(args.runs_csv)
    rows = []
    for model in MODEL_ORDER:
        part = runs[runs["model"].eq(model)]
        if part.empty:
            continue
        rows.append(
            {
                "model": model,
                "label": MODEL_LABELS[model],
                "mae_mean": part["mae_mean"].mean(),
                "mae_std": part["mae_mean"].std(),
                "rmse_mean": part["rmse_mean"].mean(),
                "rmse_std": part["rmse_mean"].std(),
            }
        )
    best_mae = min(row["mae_mean"] for row in rows)
    best_rmse = min(row["rmse_mean"] for row in rows)

    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Round 1 core baseline results averaged over block-missing patterns, missing rates, and seeds. SAITS is trained without ERA5 inputs. Lower is better.}",
        "  \\label{tab:round1-baselines}",
        "  \\begin{tabular}{lcc}",
        "    \\toprule",
        "    Method & MAE & RMSE \\\\",
        "    \\midrule",
    ]
    for row in rows:
        lines.append(
            "    "
            + row["label"]
            + " & "
            + fmt(row["mae_mean"], row["mae_std"], row["mae_mean"] == best_mae)
            + " & "
            + fmt(row["rmse_mean"], row["rmse_std"], row["rmse_mean"] == best_rmse)
            + " \\\\"
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
