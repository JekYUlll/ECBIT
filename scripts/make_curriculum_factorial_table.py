#!/usr/bin/env python3
"""Build the block-curriculum by ERA5-conditioning factorial table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROUND2_RUNS = Path("experiments/results/tables/round2_gated_final_runs.csv")
MCAR_ERA5_RUNS = Path("experiments/results/analysis/mcar_on_block/mcar_on_block_runs.csv")
MCAR_NOERA5_RUNS = Path("experiments/results/analysis/mcar_noera5_on_block/mcar_on_block_runs.csv")
OUT_CSV = Path("experiments/results/tables/curriculum_era5_factorial.csv")
OUT_TEX = Path("paper/tables/tab_curriculum_era5_factorial.tex")

PATTERN_ORDER = ["short", "medium", "long"]


def mean_or_missing(values: pd.Series) -> float | None:
    values = values.dropna()
    if values.empty:
        return None
    return float(values.mean())


def append_cell(
    rows: list[dict[str, object]],
    label: str,
    train_mask: str,
    era5: str,
    runs: pd.DataFrame | None,
) -> None:
    row: dict[str, object] = {
        "training_mask": train_mask,
        "era5_conditioning": era5,
        "setting": label,
    }
    if runs is None or runs.empty:
        row.update({"short": None, "medium": None, "long": None, "overall": None, "count": 0})
    else:
        for pattern in PATTERN_ORDER:
            row[pattern] = mean_or_missing(runs.loc[runs["test_pattern"].eq(pattern), "mae_mean"])
        row["overall"] = mean_or_missing(runs["mae_mean"])
        row["count"] = int(runs["mae_mean"].dropna().shape[0])
    rows.append(row)


def load_round2(path: Path) -> pd.DataFrame:
    runs = pd.read_csv(path)
    runs = runs[runs["pattern"].isin(PATTERN_ORDER)].copy()
    runs["test_pattern"] = runs["pattern"]
    return runs


def load_optional(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    runs = pd.read_csv(path)
    if "test_pattern" not in runs.columns:
        raise ValueError(f"{path} must contain a test_pattern column")
    return runs[runs["test_pattern"].isin(PATTERN_ORDER)].copy()


def fmt(value: object) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value):.3f}"


def write_tex(table: pd.DataFrame, out_tex: Path) -> None:
    lines = [
        "\\begin{table*}[t]",
        "  \\centering",
        "  \\caption{Cross-training-paradigm comparison of block-missing curriculum and ERA5 conditioning. All values are mean MAE on block-missing test masks; lower is better. MCAR rows are MCAR-trained models evaluated out-of-distribution on block-missing test masks, so row differences measure block-test transfer rather than standard factorial main effects.}",
        "  \\label{tab:curriculum-era5-factorial}",
        "  \\begin{tabular}{llcccc}",
        "    \\toprule",
        "    Training mask & ERA5 conditioning & Short & Medium & Long & Overall \\\\",
        "    \\midrule",
    ]
    for _, row in table.iterrows():
        lines.append(
            "    "
            + f"{row['training_mask']} & {row['era5_conditioning']} & "
            + f"{fmt(row['short'])} & {fmt(row['medium'])} & {fmt(row['long'])} & {fmt(row['overall'])} \\\\"
        )
    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table*}",
            "",
        ]
    )
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round2-runs", type=Path, default=ROUND2_RUNS)
    parser.add_argument("--mcar-era5-runs", type=Path, default=MCAR_ERA5_RUNS)
    parser.add_argument("--mcar-noera5-runs", type=Path, default=MCAR_NOERA5_RUNS)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--out-tex", type=Path, default=OUT_TEX)
    args = parser.parse_args()

    round2 = load_round2(args.round2_runs)
    mcar_era5 = load_optional(args.mcar_era5_runs)
    mcar_noera5 = load_optional(args.mcar_noera5_runs)

    rows: list[dict[str, object]] = []
    append_cell(rows, "block_era5", "Block", "Yes", round2[round2["variant"].eq("full")])
    append_cell(rows, "block_noera5", "Block", "No", round2[round2["variant"].eq("no_era5")])
    append_cell(rows, "mcar_era5", "MCAR", "Yes", mcar_era5)
    append_cell(rows, "mcar_noera5", "MCAR", "No", mcar_noera5)
    table = pd.DataFrame(rows)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_csv, index=False)
    write_tex(table, args.out_tex)
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_tex}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
