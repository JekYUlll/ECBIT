#!/usr/bin/env python3
"""Analyze held-out station error against observed ERA5-AWS mismatch."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_plot_style import apply_paper_style, save_figure, style_axis


MANIFEST_CSV = Path("data/antaws_impute_manifest.csv")
RUNS_CSV = Path("experiments/results/tables/round3_final_runs.csv")
OUT_DIR = Path("experiments/results/revision")
OUT_FIG = Path("paper/figures/fig_heldout_bias_vs_mae.pdf")
OUT_PNG = Path("paper/figures/fig_heldout_bias_vs_mae.png")
OUT_TEX = Path("paper/tables/tab_heldout_station_detail.tex")
STATION_RE = re.compile(r"ecbit_full_heldout_(.+)_(short|medium|long)_s\d+")
VARIABLES = ["T", "RH", "wspd", "P", "q"]


def station_title(station_id: str) -> str:
    return station_id.replace("_", " ").title()


def heldout_runs(path: Path) -> pd.DataFrame:
    runs = pd.read_csv(path).copy()
    parsed = runs["run_name"].map(lambda s: STATION_RE.match(s).groups())
    runs["station_id"] = parsed.map(lambda x: x[0])
    runs["regime"] = parsed.map(lambda x: x[1])
    return runs


def mismatch_summary(manifest_csv: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_csv)
    rows = []
    for row in manifest[manifest["station_group"].eq("heldout")].itertuples(index=False):
        data = np.load(row.path, allow_pickle=True)
        keep = data["window_split"].astype(str) == "train"
        x = data["X"][keep]
        era5 = data["E_3h"][keep]
        mask = data["obs_mask"][keep].astype(bool)
        record: dict[str, float | str] = {
            "station": row.station,
            "station_id": row.station_id,
        }
        vals = []
        for i, var in enumerate(VARIABLES):
            valid = mask[:, :, i]
            mismatch = float(np.mean(np.abs(x[:, :, i][valid] - era5[:, :, i][valid]))) if valid.any() else np.nan
            record[f"era5_mismatch_{var}"] = mismatch
            vals.append(mismatch)
        record["era5_mismatch_mean"] = float(np.nanmean(vals))
        rows.append(record)
    return pd.DataFrame(rows)


def write_detail_table(detail: pd.DataFrame, out_tex: Path) -> None:
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Held-out station detail at 40\\% block missingness. Errors are normalized MAE averaged over the three block regimes and three seeds. ERA5 mismatch is the mean absolute AWS-minus-ERA5 difference on the held-out station's training-segment observed positions; it is a diagnostic only and is not used for model training.}",
        "  \\label{tab:heldout-station-detail}",
        "  \\resizebox{\\textwidth}{!}{%",
        "  \\begin{tabular}{lcccccccc}",
        "    \\toprule",
        "    Station & MAE & RMSE & T & RH & Wind & P & q & ERA5 mismatch \\\\",
        "    \\midrule",
    ]
    for row in detail.sort_values("mae_mean").itertuples(index=False):
        lines.append(
            f"    {station_title(row.station_id)} & {row.mae_mean:.3f} & {row.rmse_mean:.3f} & "
            f"{row.mae_T:.3f} & {row.mae_RH:.3f} & {row.mae_wspd:.3f} & "
            f"{row.mae_P:.3f} & {row.mae_q:.3f} & {row.era5_mismatch_mean:.3f} \\\\"
        )
    lines.extend(["    \\bottomrule", "  \\end{tabular}}", "\\end{table}", ""])
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines), encoding="utf-8")


def plot_bias(detail: pd.DataFrame, out_fig: Path, out_png: Path) -> None:
    apply_paper_style(font_size=9.0)
    fig, ax = plt.subplots(figsize=(4.4, 1.95), constrained_layout=True)
    ax.scatter(
        detail["era5_mismatch_mean"],
        detail["mae_mean"],
        s=42,
        color="#2C7FB8",
        edgecolor="#222222",
        linewidth=0.5,
        zorder=3,
    )
    for row in detail.itertuples(index=False):
        ax.annotate(
            station_title(row.station_id),
            (row.era5_mismatch_mean, row.mae_mean),
            xytext=(4, 2),
            textcoords="offset points",
            fontsize=7,
            zorder=4,
        )
    ax.set_xlabel("Held-out ERA5-AWS mismatch")
    ax.set_ylabel("Held-out imputation MAE")
    style_axis(ax, grid_axis="both")
    save_figure(fig, out_fig, out_png)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", type=Path, default=MANIFEST_CSV)
    parser.add_argument("--runs-csv", type=Path, default=RUNS_CSV)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--out-fig", type=Path, default=OUT_FIG)
    parser.add_argument("--out-png", type=Path, default=OUT_PNG)
    parser.add_argument("--out-tex", type=Path, default=OUT_TEX)
    args = parser.parse_args()

    runs = heldout_runs(args.runs_csv)
    mismatch = mismatch_summary(args.manifest_csv)
    station_summary = (
        runs.groupby("station_id", as_index=False)
        .agg(
            mae_mean=("mae_mean", "mean"),
            rmse_mean=("rmse_mean", "mean"),
            mae_T=("mae_T", "mean"),
            mae_RH=("mae_RH", "mean"),
            mae_wspd=("mae_wspd", "mean"),
            mae_P=("mae_P", "mean"),
            mae_q=("mae_q", "mean"),
        )
        .merge(mismatch, on="station_id", how="left")
    )
    regime_summary = (
        runs.groupby(["station_id", "regime"], as_index=False)
        .agg(
            mae_mean=("mae_mean", "mean"),
            rmse_mean=("rmse_mean", "mean"),
            mae_T=("mae_T", "mean"),
            mae_RH=("mae_RH", "mean"),
            mae_wspd=("mae_wspd", "mean"),
            mae_P=("mae_P", "mean"),
            mae_q=("mae_q", "mean"),
        )
    )
    corr = float(station_summary["era5_mismatch_mean"].corr(station_summary["mae_mean"]))
    corr_row = pd.DataFrame([{"n": len(station_summary), "pearson_r": corr}])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    station_summary.to_csv(args.out_dir / "heldout_era5_bias_correlation.csv", index=False)
    station_summary.to_csv(args.out_dir / "heldout_per_station_results.csv", index=False)
    regime_summary.to_csv(args.out_dir / "heldout_per_station_regime_results.csv", index=False)
    corr_row.to_csv(args.out_dir / "heldout_era5_bias_correlation_summary.csv", index=False)
    write_detail_table(station_summary, args.out_tex)
    plot_bias(station_summary, args.out_fig, args.out_png)
    print(station_summary[["station_id", "mae_mean", "era5_mismatch_mean"]].to_string(index=False))
    print(f"Pearson r={corr:.3f} over n={len(station_summary)} held-out stations")


if __name__ == "__main__":
    main()
