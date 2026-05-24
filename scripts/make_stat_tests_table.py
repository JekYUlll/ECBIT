#!/usr/bin/env python3
"""Generate paired-test table with effect sizes, CIs, and Holm correction."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


RUNS_CSV = Path("experiments/results/tables/round2_gated_final_runs.csv")
OUT_CSV = Path("experiments/results/tables/round2_gated_paired_tests.csv")
OUT_TEX = Path("paper/tables/tab_stat_tests.tex")

COMPARISONS = [
    ("Gated -- concat", "full", "no_cross"),
    ("Gated -- no ERA5", "full", "no_era5"),
    ("Concat -- no ERA5", "no_cross", "no_era5"),
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


def fmt_p(value: float) -> str:
    if value < 1e-4:
        return "$<10^{-4}$"
    return f"{value:.3f}"


def paired_row(runs: pd.DataFrame, label: str, left: str, right: str) -> dict[str, float | str | int]:
    idx_cols = ["pattern", "rate", "seed"]
    left_df = runs[runs["variant"].eq(left)][idx_cols + ["mae_mean"]].rename(columns={"mae_mean": "left"})
    right_df = runs[runs["variant"].eq(right)][idx_cols + ["mae_mean"]].rename(columns={"mae_mean": "right"})
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-csv", type=Path, default=RUNS_CSV)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--out-tex", type=Path, default=OUT_TEX)
    args = parser.parse_args()

    runs = pd.read_csv(args.runs_csv)
    block_runs = runs[runs["pattern"].ne("mcar")].copy()
    rows = [paired_row(block_runs, label, left, right) for label, left, right in COMPARISONS]
    p_holm = holm_adjust([float(row["p"]) for row in rows])
    for row, adjusted in zip(rows, p_holm, strict=True):
        row["p_holm"] = adjusted

    out = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    lines = [
        "\\begin{table*}[t]",
        "  \\centering",
        "  \\caption{Paired tests for Round 2 block-missing ECBIT ablations. Pairs are matched by block regime, missing rate, and random seed ($n=27$). Differences are first variant minus second variant in normalized MAE; negative values favor the first variant. Holm-adjusted $p$ values correct the three comparisons in this family.}",
        "  \\label{tab:stat-tests}",
        "  \\begin{tabular}{lccccc}",
        "    \\toprule",
        "    Comparison & Mean diff. & 95\\% CI & $t$ & $d_z$ & Holm $p$ \\\\",
        "    \\midrule",
    ]
    for row in rows:
        lines.append(
            "    "
            + f"{row['comparison']} & {row['mean_diff']:.4f} & "
            + f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] & "
            + f"{row['t']:.2f} & {row['cohen_dz']:.2f} & {fmt_p(float(row['p_holm']))} \\\\"
        )
    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table*}",
            "",
        ]
    )
    args.out_tex.parent.mkdir(parents=True, exist_ok=True)
    args.out_tex.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_tex}")


if __name__ == "__main__":
    main()
