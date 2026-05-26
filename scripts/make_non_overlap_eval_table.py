#!/usr/bin/env python3
"""Summarize non-overlap test-window sensitivity for iTransformer+ERA5."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


FULL_RUNS = Path("experiments/results/tables/fair_era5_baselines_runs.csv")
NONOVERLAP_RUNS = Path("experiments/results/tables/non_overlap_eval_runs.csv")
OUT_CSV = Path("experiments/results/revision/non_overlap_itransformer_sensitivity.csv")
OUT_TEX = Path("paper/tables/tab_non_overlap_sensitivity.tex")


def paired_frame(full_runs: Path, nonoverlap_runs: Path) -> pd.DataFrame:
    full = pd.read_csv(full_runs)
    full = full[full["model"].eq("itransformer_era5")].copy()
    non = pd.read_csv(nonoverlap_runs)
    cols = ["run_name", "pattern", "rate", "seed", "mae_mean", "rmse_mean"]
    paired = non[cols].merge(
        full[cols],
        on=["run_name", "pattern", "rate", "seed"],
        suffixes=("_nonoverlap", "_full"),
    )
    if len(paired) != len(non):
        raise SystemExit(f"Paired only {len(paired)} of {len(non)} non-overlap runs")
    paired["mae_diff"] = paired["mae_mean_nonoverlap"] - paired["mae_mean_full"]
    paired["rmse_diff"] = paired["rmse_mean_nonoverlap"] - paired["rmse_mean_full"]
    return paired


def summary_row(label: str, frame: pd.DataFrame) -> dict[str, float | int | str]:
    diffs = frame["mae_diff"].to_numpy(dtype=float)
    n = int(len(frame))
    mean_diff = float(diffs.mean())
    sd_diff = float(diffs.std(ddof=1)) if n > 1 else 0.0
    se = sd_diff / np.sqrt(n) if n > 1 else 0.0
    if n > 1 and se > 0:
        ci_low, ci_high = stats.t.interval(0.95, df=n - 1, loc=mean_diff, scale=se)
        p_value = float(stats.ttest_1samp(diffs, 0.0).pvalue)
    else:
        ci_low = ci_high = mean_diff
        p_value = float("nan")
    return {
        "subset": label,
        "n": n,
        "full_mae_mean": float(frame["mae_mean_full"].mean()),
        "full_mae_std": float(frame["mae_mean_full"].std(ddof=1)) if n > 1 else 0.0,
        "nonoverlap_mae_mean": float(frame["mae_mean_nonoverlap"].mean()),
        "nonoverlap_mae_std": float(frame["mae_mean_nonoverlap"].std(ddof=1)) if n > 1 else 0.0,
        "mean_diff": mean_diff,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "p": p_value,
    }


def fmt_pm(mean: float, std: float) -> str:
    return f"{mean:.4f} $\\pm$ {std:.4f}"


def fmt_p(value: float) -> str:
    if np.isnan(value):
        return "--"
    if value < 1e-4:
        return "$<10^{-4}$"
    return f"{value:.3f}"


def write_tex(summary: pd.DataFrame, out_tex: Path) -> None:
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Non-overlap test-window sensitivity for retained iTransformer+ERA5 checkpoints. Full-test and non-overlap scores use the same trained checkpoints and matched block-missing configurations; differences are non-overlap minus full-test normalized MAE.}",
        "  \\label{tab:non-overlap-sensitivity}",
        "  \\footnotesize",
        "  \\setlength{\\tabcolsep}{3pt}",
        "  \\begin{tabular}{lcccc}",
        "    \\toprule",
        "    Split & Full MAE & Non-overlap MAE & $\\Delta$ & $p$ \\\\",
        "    \\midrule",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"    {row.subset} & {fmt_pm(row.full_mae_mean, row.full_mae_std)} & "
            f"{fmt_pm(row.nonoverlap_mae_mean, row.nonoverlap_mae_std)} & "
            f"{row.mean_diff:+.4f} & {fmt_p(row.p)} \\\\"
        )
    lines.extend(["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""])
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-runs", type=Path, default=FULL_RUNS)
    parser.add_argument("--nonoverlap-runs", type=Path, default=NONOVERLAP_RUNS)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--out-tex", type=Path, default=OUT_TEX)
    args = parser.parse_args()

    paired = paired_frame(args.full_runs, args.nonoverlap_runs)
    rows = [summary_row("Overall", paired)]
    for pattern in ["short", "medium", "long"]:
        rows.append(summary_row(pattern.capitalize(), paired[paired["pattern"].eq(pattern)]))
    summary = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_csv, index=False)
    write_tex(summary, args.out_tex)
    print(summary.to_string(index=False))
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_tex}")


if __name__ == "__main__":
    main()
