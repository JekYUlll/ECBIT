#!/usr/bin/env python3
"""Plot the main result comparison across baselines and ECBIT variants."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROUND1_RUNS = Path("experiments/results/tables/round1_core_runs.csv")
ROUND2_RUNS = Path("experiments/results/tables/round2_gated_final_runs.csv")
OUT_PDF = Path("paper/figures/fig_main_results.pdf")
OUT_PNG = Path("paper/figures/fig_main_results.png")


def collect_rows(round1_runs: Path, round2_runs: Path) -> pd.DataFrame:
    r1 = pd.read_csv(round1_runs)
    r2 = pd.read_csv(round2_runs)
    rows = []
    for model, label in [
        ("linear_interp", "Linear"),
        ("locf", "LOCF"),
        ("era5_direct", "ERA5 direct"),
        ("itransformer", "iTransformer"),
    ]:
        vals = r1.loc[r1["model"].eq(model), "mae_mean"]
        rows.append({"label": label, "mean": vals.mean(), "std": vals.std(), "count": len(vals), "group": "Baseline"})

    block = r2[r2["pattern"].ne("mcar")]
    for variant, label in [
        ("no_era5", "ECBIT w/o ERA5"),
        ("no_cross", "ECBIT concat"),
        ("full", "ECBIT + ERA5"),
    ]:
        vals = block.loc[block["variant"].eq(variant), "mae_mean"]
        rows.append({"label": label, "mean": vals.mean(), "std": vals.std(), "count": len(vals), "group": "ECBIT"})
    return pd.DataFrame(rows)


def plot(round1_runs: Path, round2_runs: Path, out_pdf: Path, out_png: Path) -> None:
    data = collect_rows(round1_runs, round2_runs)
    colors = ["#9ECAE1", "#FDD0A2", "#A1D99B", "#FC9272", "#BCBDDC", "#BDBDBD", "#3182BD"]

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
        }
    )
    fig, ax = plt.subplots(figsize=(7.35, 2.65), constrained_layout=True)
    x = range(len(data))
    ax.bar(x, data["mean"], yerr=data["std"], capsize=2.5, color=colors, edgecolor="#333333", linewidth=0.4)
    for idx, row in data.iterrows():
        ax.text(idx, row["mean"] + row["std"] + 0.01, f"{row['mean']:.3f}", ha="center", va="bottom", fontsize=7)

    ax.axvline(3.5, color="#666666", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.text(1.5, 0.535, "Core baselines", ha="center", va="center", fontsize=8, color="#444444")
    ax.text(5.0, 0.535, "ECBIT variants", ha="center", va="center", fontsize=8, color="#444444")
    ax.set_xticks(list(x), data["label"], rotation=22, ha="right")
    ax.set_ylabel("Mean test MAE")
    ax.set_ylim(0.22, 0.56)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6, alpha=0.85)
    ax.set_axisbelow(True)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round1-runs", type=Path, default=ROUND1_RUNS)
    parser.add_argument("--round2-runs", type=Path, default=ROUND2_RUNS)
    parser.add_argument("--out-pdf", type=Path, default=OUT_PDF)
    parser.add_argument("--out-png", type=Path, default=OUT_PNG)
    args = parser.parse_args()
    plot(args.round1_runs, args.round2_runs, args.out_pdf, args.out_png)


if __name__ == "__main__":
    main()
