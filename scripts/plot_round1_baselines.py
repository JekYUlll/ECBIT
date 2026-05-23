#!/usr/bin/env python3
"""Plot Round 1 baseline comparison for the ECBIT paper."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RUNS_CSV = Path("experiments/results/tables/round1_core_runs.csv")
OUT_PDF = Path("paper/figures/fig_round1_baselines.pdf")
OUT_PNG = Path("paper/figures/fig_round1_baselines.png")

MODEL_ORDER = ["linear_interp", "locf", "era5_direct", "itransformer"]
MODEL_LABELS = {
    "linear_interp": "Linear",
    "locf": "LOCF",
    "era5_direct": "ERA5 direct",
    "itransformer": "iTransformer",
}
COLORS = {
    "linear_interp": "#56B4E9",
    "locf": "#E69F00",
    "era5_direct": "#009E73",
    "itransformer": "#D55E00",
}
MARKERS = {
    "linear_interp": "o",
    "locf": "s",
    "era5_direct": "^",
    "itransformer": "D",
}


def plot(runs_csv: Path, out_pdf: Path, out_png: Path) -> None:
    runs = pd.read_csv(runs_csv)
    runs = runs[runs["model"].isin(MODEL_ORDER)].copy()
    runs["rate_pct"] = (runs["rate"].astype(float) * 100).astype(int)
    summary = (
        runs.groupby(["model", "pattern", "rate_pct"], as_index=False)
        .agg(mae_mean=("mae_mean", "mean"), mae_std=("mae_mean", "std"))
    )

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), sharey=True, constrained_layout=True)
    for ax, pattern in zip(axes, ["short", "medium", "long"]):
        part = summary[summary["pattern"].eq(pattern)]
        for model in MODEL_ORDER:
            model_part = part[part["model"].eq(model)].sort_values("rate_pct")
            ax.errorbar(
                model_part["rate_pct"],
                model_part["mae_mean"],
                yerr=model_part["mae_std"],
                marker=MARKERS[model],
                color=COLORS[model],
                linewidth=1.5,
                markersize=4.2,
                capsize=2,
                label=MODEL_LABELS[model],
            )
        ax.set_title(pattern.capitalize())
        ax.set_xlabel("Missing rate (%)")
        ax.set_xticks([20, 40, 60])
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6, alpha=0.85)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("MAE (normalized)")
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-csv", type=Path, default=RUNS_CSV)
    parser.add_argument("--out-pdf", type=Path, default=OUT_PDF)
    parser.add_argument("--out-png", type=Path, default=OUT_PNG)
    args = parser.parse_args()
    plot(args.runs_csv, args.out_pdf, args.out_png)


if __name__ == "__main__":
    main()
