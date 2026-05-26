#!/usr/bin/env python3
"""Paired tests between station-month ERA5 direct and ERA5-augmented neural models."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


CALIBRATION_RUNS = Path("experiments/results/revision/era5_calibration_comparison.csv")
FAIR_RUNS = Path("experiments/results/tables/fair_era5_baselines_runs.csv")
ROUND2_RUNS = Path("experiments/results/tables/round2_gated_final_runs.csv")
OUT_CSV = Path("experiments/results/revision/station_month_vs_neural_paired_tests.csv")
OUT_TEX = Path("paper/tables/tab_station_month_vs_neural_tests.tex")

METHODS = [
    ("SAITS+ERA5", "fair", "saits_era5_concat"),
    ("iTransformer+ERA5", "fair", "itransformer_era5"),
    ("ECBIT concat", "round2", "no_cross"),
    ("ECBIT gated", "round2", "full"),
]


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running_max = 0.0
    m = len(p_values)
    for rank, idx in enumerate(order):
        value = min((m - rank) * p_values[idx], 1.0)
        running_max = max(running_max, value)
        adjusted[idx] = running_max
    return adjusted.tolist()


def neural_frame(fair: pd.DataFrame, round2: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for label, source, key in METHODS:
        if source == "fair":
            part = fair[fair["model"].eq(key)].copy()
        elif source == "round2":
            part = round2[round2["variant"].eq(key) & round2["pattern"].ne("mcar")].copy()
        else:
            raise ValueError(source)
        part["method"] = label
        frames.append(part[["method", "pattern", "rate", "seed", "mae_mean"]])
    return pd.concat(frames, ignore_index=True)


def paired_row(baseline: pd.DataFrame, neural: pd.DataFrame, method: str) -> dict[str, float | int | str]:
    idx_cols = ["pattern", "rate", "seed"]
    left = neural[neural["method"].eq(method)][idx_cols + ["mae_mean"]].rename(columns={"mae_mean": "neural"})
    right = baseline[idx_cols + ["mae_mean"]].rename(columns={"mae_mean": "station_month"})
    paired = left.merge(right, on=idx_cols, how="inner")
    diffs = paired["neural"].to_numpy(dtype=float) - paired["station_month"].to_numpy(dtype=float)
    n = int(diffs.size)
    mean = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1))
    se = sd / np.sqrt(n)
    t_stat = mean / se
    p_value = float(stats.ttest_1samp(diffs, 0.0).pvalue)
    ci_low, ci_high = stats.t.interval(0.95, df=n - 1, loc=mean, scale=se)
    return {
        "comparison": f"{method} -- ERA5 station-month",
        "n": n,
        "mean_diff": mean,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "t": float(t_stat),
        "p": p_value,
        "cohen_dz": mean / sd,
        "baseline_mae": float(right["station_month"].mean()),
        "method_mae": float(left["neural"].mean()),
    }


def fmt_p(value: float) -> str:
    if value < 1e-4:
        return "$<10^{-4}$"
    return f"{value:.3f}"


def write_tex(tests: pd.DataFrame, out_tex: Path) -> None:
    lines = [
        "\\begin{table*}[t]",
        "  \\centering",
        "  \\caption{Paired tests against station-month calibrated ERA5 direct substitution. Pairs are matched by block regime, missing rate, and random seed ($n=27$). Differences are neural model minus station-month ERA5 in normalized MAE; negative values favor the neural model. Holm-adjusted $p$ values correct this four-comparison family.}",
        "  \\label{tab:station-month-neural-tests}",
        "  \\begin{tabular}{lccccc}",
        "    \\toprule",
        "    Comparison & Mean diff. & 95\\% CI & $t$ & $d_z$ & Holm $p$ \\\\",
        "    \\midrule",
    ]
    for row in tests.itertuples(index=False):
        lines.append(
            f"    {row.comparison} & {row.mean_diff:.4f} & "
            f"[{row.ci95_low:.4f}, {row.ci95_high:.4f}] & "
            f"{row.t:.2f} & {row.cohen_dz:.2f} & {fmt_p(row.p_holm)} \\\\"
        )
    lines.extend(["    \\bottomrule", "  \\end{tabular}", "\\end{table*}", ""])
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-runs", type=Path, default=CALIBRATION_RUNS)
    parser.add_argument("--fair-runs", type=Path, default=FAIR_RUNS)
    parser.add_argument("--round2-runs", type=Path, default=ROUND2_RUNS)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--out-tex", type=Path, default=OUT_TEX)
    args = parser.parse_args()

    calibration = pd.read_csv(args.calibration_runs)
    baseline = calibration[calibration["mode"].eq("station_month_bias")].copy()
    neural = neural_frame(pd.read_csv(args.fair_runs), pd.read_csv(args.round2_runs))
    rows = [paired_row(baseline, neural, method) for method, _, _ in METHODS]
    adjusted = holm_adjust([float(row["p"]) for row in rows])
    for row, p_holm in zip(rows, adjusted, strict=True):
        row["p_holm"] = p_holm
    tests = pd.DataFrame(rows)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    tests.to_csv(args.out_csv, index=False)
    write_tex(tests, args.out_tex)
    print(tests.to_string(index=False))
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_tex}")


if __name__ == "__main__":
    main()
