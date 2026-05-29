#!/usr/bin/env python3
"""Create raw-unit error tables from normalized ECBIT result CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SCALER_NPZ = Path("data/antaws_impute_scaler.npz")
ROUND2_RUNS = Path("experiments/results/tables/round2_gated_final_runs.csv")
ROUND3_RUNS = Path("experiments/results/tables/round3_final_runs.csv")
OUT_DIR = Path("experiments/results/tables")
PAPER_TABLE = Path("paper/tables/tab_raw_unit_errors.tex")

VAR_LABELS = {
    "T": "Temperature (deg C)",
    "RH": "Relative humidity (\\%)",
    "wspd": "Wind speed (m s$^{-1}$)",
    "P": "Pressure (hPa)",
    "q": "Specific humidity (g kg$^{-1}$)",
}


def load_scale(scaler_npz: Path) -> dict[str, float]:
    data = np.load(scaler_npz, allow_pickle=True)
    variables = [str(v) for v in data["variables"]]
    std = data["std"].astype(float)
    return dict(zip(variables, std))


def raw_variable_summary(runs: pd.DataFrame, scale: dict[str, float], group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, part in runs.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        for var, std in scale.items():
            mae_col = f"mae_{var}"
            rmse_col = f"rmse_{var}"
            if mae_col not in part.columns:
                continue
            rows.append(
                {
                    **base,
                    "variable": var,
                    "variable_label": VAR_LABELS[var],
                    "mae_raw_mean": float((part[mae_col] * std).mean()),
                    "mae_raw_std": float((part[mae_col] * std).std()),
                    "rmse_raw_mean": float((part[rmse_col] * std).mean()),
                    "rmse_raw_std": float((part[rmse_col] * std).std()),
                    "n": int(len(part)),
                }
            )
    return pd.DataFrame(rows)


def write_latex(summary: pd.DataFrame, out_tex: Path) -> None:
    focus = summary[summary["variant"].eq("full")].copy()
    order = ["T", "RH", "wspd", "P", "q"]
    focus["variable"] = pd.Categorical(focus["variable"], order, ordered=True)
    focus = focus.sort_values("variable")
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Raw-unit ECBIT errors for the gated ERA5 model over the 27 block-missing test configurations. Values convert normalized MAE/RMSE using training-window standard deviations.}",
        "  \\label{tab:raw-unit-errors}",
        "  \\begin{tabular}{lcc}",
        "    \\toprule",
        "    Variable & MAE & RMSE \\\\",
        "    \\midrule",
    ]
    for _, row in focus.iterrows():
        fmt = ".3f" if row["variable"] == "q" else ".2f"
        lines.append(
            "    "
            + f"{row['variable_label']} & "
            + f"{row['mae_raw_mean']:{fmt}} $\\pm$ {row['mae_raw_std']:{fmt}} & "
            + f"{row['rmse_raw_mean']:{fmt}} $\\pm$ {row['rmse_raw_std']:{fmt}} \\\\"
        )
    lines.extend(["    \\bottomrule", "  \\end{tabular}", "\\end{table}"])
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scaler-npz", type=Path, default=SCALER_NPZ)
    parser.add_argument("--round2-runs", type=Path, default=ROUND2_RUNS)
    parser.add_argument("--round3-runs", type=Path, default=ROUND3_RUNS)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--paper-table", type=Path, default=PAPER_TABLE)
    args = parser.parse_args()

    scale = load_scale(args.scaler_npz)
    round2 = pd.read_csv(args.round2_runs)
    round3 = pd.read_csv(args.round3_runs)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    r2_summary = raw_variable_summary(round2[round2["variant"].isin(["full", "no_era5"])], scale, ["variant"])
    r3_summary = raw_variable_summary(round3, scale, ["run_name"])
    r3_summary["station_id"] = r3_summary["run_name"].str.extract(r"heldout_(.+)_(?:short|medium|long)_s\\d+")[0]
    r3_station = (
        r3_summary.groupby(["station_id", "variable", "variable_label"], as_index=False)
        .agg(
            mae_raw_mean=("mae_raw_mean", "mean"),
            rmse_raw_mean=("rmse_raw_mean", "mean"),
            n=("n", "sum"),
        )
        .sort_values(["station_id", "variable"])
    )

    r2_summary.to_csv(args.out_dir / "round2_raw_unit_variable_summary.csv", index=False)
    r3_station.to_csv(args.out_dir / "round3_raw_unit_station_variable_summary.csv", index=False)
    write_latex(r2_summary, args.paper_table)
    print(r2_summary.to_string(index=False))
    print(f"Wrote {args.paper_table}")


if __name__ == "__main__":
    main()
