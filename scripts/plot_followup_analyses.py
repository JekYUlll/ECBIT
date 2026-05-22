#!/usr/bin/env python3
"""Plot compact follow-up analyses for the ECBIT paper."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BLOCKLEN_CSV = Path("experiments/results/tables/followup_blocklen_final_summary.csv")
ROBUSTNESS_CSV = Path("experiments/results/analysis/era5_robustness/era5_robustness_summary.csv")
OUT_PDF = Path("paper/figures/fig_followup_analyses.pdf")
OUT_PNG = Path("paper/figures/fig_followup_analyses.png")

COLORS = {
    "full": "#0072B2",
    "no_era5": "#D55E00",
    "bar": "#009E73",
    "down": "#CC79A7",
    "grid": "#DDDDDD",
}


def plot_block_length(ax: plt.Axes, summary: pd.DataFrame) -> None:
    data = summary.copy()
    data["hours"] = data["pattern"].str.extract(r"(\d+)h").astype(int)
    data = data.sort_values(["variant", "hours"])

    for variant, label in [("full", "ERA5-conditioned"), ("no_era5", "No ERA5")]:
        part = data[data["variant"].eq(variant)]
        ax.errorbar(
            part["hours"],
            part["mae_mean_mean"],
            yerr=part["mae_mean_std"].fillna(0.0),
            marker="o" if variant == "full" else "s",
            color=COLORS[variant],
            linewidth=1.7,
            markersize=4.5,
            capsize=2,
            label=label,
        )

    pivot = data.pivot(index="hours", columns="variant", values="mae_mean_mean")
    for hours, row in pivot.iterrows():
        if {"full", "no_era5"}.issubset(row.index):
            gap = row["no_era5"] - row["full"]
            ax.annotate(
                f"+{gap:.3f}",
                xy=(hours, row["no_era5"]),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color="#444444",
            )

    ax.set_title("(a) Outage length", fontsize=8.2, pad=4)
    ax.set_xlabel("Maximum block length (hours)")
    ax.set_ylabel("MAE")
    ax.set_xticks([24, 72, 216])
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6, alpha=0.9)
    ax.legend(frameon=False, fontsize=6.8, loc="upper left")


def plot_variable_importance(ax: plt.Axes, robust: pd.DataFrame) -> None:
    variables = robust[robust["perturbation"].str.startswith("mask_")].copy()
    variables["label"] = variables["perturbation"].str.replace("mask_", "", regex=False)
    variables = variables.sort_values("delta_mae", ascending=True)

    ax.barh(variables["label"], variables["delta_mae"], color=COLORS["bar"], alpha=0.9)
    for y, value in enumerate(variables["delta_mae"]):
        ax.text(value + 0.001, y, f"{value:.3f}", va="center", fontsize=7)

    ax.set_title("(b) ERA5 channel importance", fontsize=8.2, pad=4)
    ax.set_xlabel("MAE increase when masked")
    ax.set_xlim(0, max(variables["delta_mae"].max() * 1.22, 0.065))
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.6, alpha=0.9)


def plot_temporal_robustness(ax: plt.Axes, robust: pd.DataFrame) -> None:
    labels = ["6h", "12h", "24h"]
    rows = []
    for label in labels:
        row = robust[robust["perturbation"].eq(f"downsample_{label}")].iloc[0]
        rows.append((label, row["mean"], row["delta_mae"]))
    x = np.arange(len(rows))
    means = [row[1] for row in rows]
    deltas = [row[2] for row in rows]

    ax.bar(x, deltas, color=COLORS["down"], alpha=0.9, width=0.58)
    for idx, (label, mean, delta) in enumerate(rows):
        ax.text(idx, delta + 0.001, f"+{delta:.3f}\nMAE {mean:.3f}", ha="center", va="bottom", fontsize=6.8)

    ax.set_title("(c) ERA5 temporal robustness", fontsize=8.2, pad=4)
    ax.set_xlabel("ERA5 downsampling")
    ax.set_ylabel("MAE increase")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(deltas) * 1.32)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6, alpha=0.9)


def plot(blocklen_csv: Path, robustness_csv: Path, out_pdf: Path, out_png: Path) -> None:
    blocklen = pd.read_csv(blocklen_csv)
    robust = pd.read_csv(robustness_csv)

    plt.rcParams.update(
        {
            "font.size": 7.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.25), constrained_layout=True)
    plot_block_length(axes[0], blocklen)
    plot_variable_importance(axes[1], robust)
    plot_temporal_robustness(axes[2], robust)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocklen-csv", type=Path, default=BLOCKLEN_CSV)
    parser.add_argument("--robustness-csv", type=Path, default=ROBUSTNESS_CSV)
    parser.add_argument("--out-pdf", type=Path, default=OUT_PDF)
    parser.add_argument("--out-png", type=Path, default=OUT_PNG)
    args = parser.parse_args()
    plot(args.blocklen_csv, args.robustness_csv, args.out_pdf, args.out_png)


if __name__ == "__main__":
    main()
