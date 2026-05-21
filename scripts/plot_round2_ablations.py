#!/usr/bin/env python3
"""Plot Round 2 ERA5-conditioning ablation trends for the paper."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RUNS_CSV = Path("experiments/results/tables/round2_partial_runs.csv")
OUT_PDF = Path("paper/figures/fig_round2_ablations.pdf")
OUT_PNG = Path("paper/figures/fig_round2_ablations.png")

VARIANT_ORDER = ["full", "no_cross", "no_era5"]
VARIANT_LABELS = {
    "full": "Gated injection",
    "no_cross": "Concat/no-cross",
    "no_era5": "No ERA5",
}
COLORS = {
    "full": "#D55E00",
    "no_cross": "#0072B2",
    "no_era5": "#009E73",
    "no_blockmask": "#CC79A7",
}
MARKERS = {
    "full": "o",
    "no_cross": "s",
    "no_era5": "^",
    "no_blockmask": "D",
}


def plot_block_ablations(ax: plt.Axes, runs: pd.DataFrame, pattern: str) -> None:
    part = runs[runs["pattern"].eq(pattern)].copy()
    for variant in VARIANT_ORDER:
        variant_part = (
            part[part["variant"].eq(variant)]
            .groupby("rate_pct", as_index=False)
            .agg(mae_mean=("mae_mean", "mean"), mae_std=("mae_mean", "std"), count=("mae_mean", "count"))
            .sort_values("rate_pct")
        )
        if variant_part.empty:
            continue
        ax.errorbar(
            variant_part["rate_pct"],
            variant_part["mae_mean"],
            yerr=variant_part["mae_std"].fillna(0.0),
            marker=MARKERS[variant],
            color=COLORS[variant],
            linewidth=1.5,
            markersize=4.2,
            capsize=2,
            label=VARIANT_LABELS[variant],
        )
    ax.set_title(pattern.capitalize())
    ax.set_xlabel("Missing rate (%)")
    ax.set_xticks([20, 40, 60])
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6, alpha=0.85)
    ax.set_axisbelow(True)


def plot_mcar(ax: plt.Axes, runs: pd.DataFrame) -> None:
    part = (
        runs[runs["variant"].eq("no_blockmask")]
        .groupby("rate_pct", as_index=False)
        .agg(mae_mean=("mae_mean", "mean"), mae_std=("mae_mean", "std"))
        .sort_values("rate_pct")
    )
    if part.empty:
        ax.text(0.5, 0.5, "MCAR ablation pending", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.errorbar(
            part["rate_pct"],
            part["mae_mean"],
            yerr=part["mae_std"].fillna(0.0),
            marker=MARKERS["no_blockmask"],
            color=COLORS["no_blockmask"],
            linewidth=1.5,
            markersize=4.2,
            capsize=2,
            label="MCAR training",
        )
    ax.set_title("MCAR")
    ax.set_xlabel("Missing rate (%)")
    ax.set_xticks([20, 40, 60])
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6, alpha=0.85)
    ax.set_axisbelow(True)


def plot(runs_csv: Path, out_pdf: Path, out_png: Path) -> None:
    runs = pd.read_csv(runs_csv)
    runs = runs.copy()
    runs["rate_pct"] = (runs["rate"].astype(float) * 100).astype(int)

    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.45), sharey=False, constrained_layout=True)
    for ax, pattern in zip(axes[:3], ["short", "medium", "long"]):
        plot_block_ablations(ax, runs, pattern)
    axes[0].set_ylabel("MAE (normalized)")
    plot_mcar(axes[3], runs)
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    axes[3].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("Round 2 ERA5-conditioning ablations", fontsize=10, fontweight="bold")

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
