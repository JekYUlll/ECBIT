#!/usr/bin/env python3
"""Generate fair ERA5-augmented baseline tables and paired tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROUND1_RUNS = Path("experiments/results/tables/round1_core_runs.csv")
FAIR_RUNS = Path("experiments/results/tables/fair_era5_baselines_runs.csv")
ROUND2_RUNS = Path("experiments/results/tables/round2_gated_final_runs.csv")
OUT_SUMMARY = Path("experiments/results/tables/fair_era5_comparison_summary.csv")
OUT_TESTS = Path("experiments/results/tables/fair_era5_paired_tests.csv")
OUT_TEX = Path("paper/tables/tab_fair_era5_baselines.tex")
OUT_TESTS_TEX = Path("paper/tables/tab_fair_era5_tests.tex")

METHODS = [
    ("SAITS", "round1", "saits"),
    ("SAITS+ERA5", "fair", "saits_era5_concat"),
    ("iTransformer", "round1", "itransformer"),
    ("iTransformer+ERA5", "fair", "itransformer_era5"),
    ("ECBIT concat", "round2", "no_cross"),
    ("ECBIT gated", "round2", "full"),
]

COMPARISONS = [
    ("SAITS+ERA5 -- SAITS", "SAITS+ERA5", "SAITS"),
    ("iTransformer+ERA5 -- iTransformer", "iTransformer+ERA5", "iTransformer"),
    ("SAITS+ERA5 -- ECBIT gated", "SAITS+ERA5", "ECBIT gated"),
    ("iTransformer+ERA5 -- ECBIT gated", "iTransformer+ERA5", "ECBIT gated"),
    ("ECBIT concat -- ECBIT gated", "ECBIT concat", "ECBIT gated"),
    ("SAITS+ERA5 -- iTransformer+ERA5", "SAITS+ERA5", "iTransformer+ERA5"),
]


def method_frame(round1: pd.DataFrame, fair: pd.DataFrame, round2: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for label, source, key in METHODS:
        if source == "round1":
            part = round1[round1["model"].eq(key)].copy()
        elif source == "fair":
            part = fair[fair["model"].eq(key)].copy()
        elif source == "round2":
            part = round2[round2["variant"].eq(key) & round2["pattern"].ne("mcar")].copy()
        else:
            raise ValueError(source)
        part["method"] = label
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    overall = data.groupby("method", as_index=False).agg(
        mae_mean=("mae_mean", "mean"),
        mae_std=("mae_mean", "std"),
        rmse_mean=("rmse_mean", "mean"),
        rmse_std=("rmse_mean", "std"),
        count=("mae_mean", "size"),
    )
    pattern = data.pivot_table(index="method", columns="pattern", values="mae_mean", aggfunc="mean").reset_index()
    summary = overall.merge(pattern, on="method", how="left")
    order = {label: idx for idx, (label, _, _) in enumerate(METHODS)}
    summary["_order"] = summary["method"].map(order)
    return summary.sort_values("_order").drop(columns="_order")


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


def paired_row(data: pd.DataFrame, label: str, left: str, right: str) -> dict[str, float | int | str]:
    idx_cols = ["pattern", "rate", "seed"]
    left_df = data[data["method"].eq(left)][idx_cols + ["mae_mean"]].rename(columns={"mae_mean": "left"})
    right_df = data[data["method"].eq(right)][idx_cols + ["mae_mean"]].rename(columns={"mae_mean": "right"})
    paired = left_df.merge(right_df, on=idx_cols, how="inner")
    diffs = paired["left"].to_numpy(dtype=float) - paired["right"].to_numpy(dtype=float)
    n = int(diffs.size)
    mean = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1))
    se = sd / np.sqrt(n)
    t_stat = mean / se
    p_value = float(stats.ttest_1samp(diffs, 0.0).pvalue)
    ci_low, ci_high = stats.t.interval(0.95, df=n - 1, loc=mean, scale=se)
    return {
        "comparison": label,
        "n": n,
        "mean_diff": mean,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "t": float(t_stat),
        "p": p_value,
        "cohen_dz": mean / sd,
    }


def fmt(mean: float, std: float, bold: bool = False) -> str:
    text = f"{mean:.3f} $\\pm$ {std:.3f}"
    return f"\\textbf{{{text}}}" if bold else text


def fmt_p(value: float) -> str:
    if value < 1e-4:
        return "$<10^{-4}$"
    return f"{value:.3f}"


def write_summary_table(summary: pd.DataFrame, out_tex: Path) -> None:
    best = float(summary["mae_mean"].min())
    lines = [
        "\\begin{table*}[t]",
        "  \\centering",
        "  \\caption{Fair ERA5-augmented baseline comparison on the 27 matched block-missing configurations. SAITS+ERA5 uses simple channel concatenation of AWS, masks, time features, and ERA5 covariates; iTransformer+ERA5 uses the same variate-token backbone with ERA5 concatenated to the token features. Lower is better.}",
        "  \\label{tab:fair-era5-baselines}",
        "  \\begin{tabular}{lccccc}",
        "    \\toprule",
        "    Method & Overall MAE & RMSE & Short MAE & Medium MAE & Long MAE \\\\",
        "    \\midrule",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"    {row.method} & "
            f"{fmt(row.mae_mean, row.mae_std, abs(row.mae_mean - best) < 1e-12)} & "
            f"{fmt(row.rmse_mean, row.rmse_std)} & "
            f"{row.short:.3f} & {row.medium:.3f} & {row.long:.3f} \\\\"
        )
    lines.extend(["    \\bottomrule", "  \\end{tabular}", "\\end{table*}", ""])
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines), encoding="utf-8")


def write_tests_table(tests: pd.DataFrame, out_tex: Path) -> None:
    lines = [
        "\\begin{table*}[t]",
        "  \\centering",
        "  \\caption{Paired tests for fair ERA5-augmented baselines. Pairs are matched by block regime, missing rate, and random seed ($n=27$). Differences are first method minus second method in normalized MAE; negative values favor the first method. Holm-adjusted $p$ values correct this six-comparison family.}",
        "  \\label{tab:fair-era5-tests}",
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
    parser.add_argument("--round1-runs", type=Path, default=ROUND1_RUNS)
    parser.add_argument("--fair-runs", type=Path, default=FAIR_RUNS)
    parser.add_argument("--round2-runs", type=Path, default=ROUND2_RUNS)
    parser.add_argument("--out-summary", type=Path, default=OUT_SUMMARY)
    parser.add_argument("--out-tests", type=Path, default=OUT_TESTS)
    parser.add_argument("--out-tex", type=Path, default=OUT_TEX)
    parser.add_argument("--out-tests-tex", type=Path, default=OUT_TESTS_TEX)
    args = parser.parse_args()

    data = method_frame(pd.read_csv(args.round1_runs), pd.read_csv(args.fair_runs), pd.read_csv(args.round2_runs))
    summary = summarize(data)
    rows = [paired_row(data, label, left, right) for label, left, right in COMPARISONS]
    adjusted = holm_adjust([float(row["p"]) for row in rows])
    for row, p_holm in zip(rows, adjusted, strict=True):
        row["p_holm"] = p_holm
    tests = pd.DataFrame(rows)

    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_summary, index=False)
    tests.to_csv(args.out_tests, index=False)
    write_summary_table(summary, args.out_tex)
    write_tests_table(tests, args.out_tests_tex)
    print(f"wrote {args.out_summary}")
    print(f"wrote {args.out_tests}")
    print(f"wrote {args.out_tex}")
    print(f"wrote {args.out_tests_tex}")


if __name__ == "__main__":
    main()
