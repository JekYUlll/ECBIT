#!/usr/bin/env python3
"""Plot AntAWS missing-pattern statistics for the ECBIT paper."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from paper_plot_style import apply_paper_style, save_figure, style_axis


STATS_CSV = Path("results/missing_analysis/pattern_stats.csv")
OUT_PDF = Path("paper/figures/fig_missing_patterns.pdf")
OUT_PNG = Path("paper/figures/fig_missing_patterns.png")
VAR_ORDER = ["T", "RH", "wspd", "P"]


def plot(stats_csv: Path, out_pdf: Path, out_png: Path) -> None:
    apply_paper_style(font_size=9.0)
    stats = pd.read_csv(stats_csv)
    stats = stats[stats["variable"].isin(VAR_ORDER)].copy()
    summary = (
        stats.groupby(["split", "variable"], as_index=False)
        .agg(
            missing_rate=("missing_rate", "mean"),
            p90_block_steps=("p90_block_steps", "median"),
            max_block_steps=("max_block_steps", "max"),
        )
    )
    summary["variable"] = pd.Categorical(summary["variable"], categories=VAR_ORDER, ordered=True)
    summary = summary.sort_values(["split", "variable"])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), constrained_layout=True)
    width = 0.36
    x = range(len(VAR_ORDER))
    for ax, metric, ylabel, scale in [
        (axes[0], "missing_rate", "Mean missing rate", 100.0),
        (axes[1], "p90_block_steps", "Median p90 block length (days)", 3.0 / 24.0),
    ]:
        for j, split in enumerate(["main", "heldout"]):
            part = summary[summary["split"].eq(split)].set_index("variable").reindex(VAR_ORDER)
            offset = (j - 0.5) * width
            values = part[metric].to_numpy() * scale
            ax.bar(
                [i + offset for i in x],
                values,
                width=width,
                label=split,
                color=["#56B4E9", "#E69F00"][j],
                edgecolor="black",
                linewidth=0.4,
            )
        ax.set_xticks(list(x), VAR_ORDER)
        ax.set_ylabel(ylabel)
        style_axis(ax, grid_axis="y")
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(bottom=0)
    axes[0].legend(frameon=False, loc="upper left")

    save_figure(fig, out_pdf, out_png)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-csv", type=Path, default=STATS_CSV)
    parser.add_argument("--out-pdf", type=Path, default=OUT_PDF)
    parser.add_argument("--out-png", type=Path, default=OUT_PNG)
    args = parser.parse_args()
    plot(args.stats_csv, args.out_pdf, args.out_png)


if __name__ == "__main__":
    main()
