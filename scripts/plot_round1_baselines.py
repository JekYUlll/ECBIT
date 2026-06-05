#!/usr/bin/env python3
"""Plot Round 1 baseline comparison for the ECBIT paper."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from paper_plot_style import apply_paper_style, save_figure, style_axis


RUNS_CSV = Path("experiments/results/tables/round1_core_runs.csv")
OUT_PDF = Path("paper/figures/fig_round1_baselines.pdf")
OUT_PNG = Path("paper/figures/fig_round1_baselines.png")

MODEL_ORDER = ["linear_interp", "locf", "era5_direct", "saits", "itransformer"]
MODEL_LABELS = {
    "linear_interp": "Linear",
    "locf": "LOCF",
    "era5_direct": "ERA5 direct",
    "saits": "SAITS",
    "itransformer": "iTransformer",
}
COLORS = {
    "linear_interp": "#56B4E9",
    "locf": "#E69F00",
    "era5_direct": "#009E73",
    "saits": "#CC79A7",
    "itransformer": "#D55E00",
}
MARKERS = {
    "linear_interp": "o",
    "locf": "s",
    "era5_direct": "^",
    "saits": "v",
    "itransformer": "D",
}


def plot(runs_csv: Path, out_pdf: Path, out_png: Path) -> None:
    apply_paper_style(font_size=9.0)
    runs = pd.read_csv(runs_csv)
    runs = runs[runs["model"].isin(MODEL_ORDER)].copy()
    runs["rate_pct"] = (runs["rate"].astype(float) * 100).astype(int)
    summary = (
        runs.groupby(["model", "pattern", "rate_pct"], as_index=False)
        .agg(mae_mean=("mae_mean", "mean"), mae_std=("mae_mean", "std"))
    )

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), sharey=True, constrained_layout=False)
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
        ax.set_xlabel(r"Missing rate (\%)")
        ax.set_xticks([20, 40, 60])
        style_axis(ax, grid_axis="y")
    axes[0].set_ylabel("MAE (normalized)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=8.4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=5,
        columnspacing=1.0,
        handlelength=1.8,
    )
    fig.subplots_adjust(top=0.78, bottom=0.20, left=0.08, right=0.99, wspace=0.08)

    save_figure(fig, out_pdf, out_png)
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
