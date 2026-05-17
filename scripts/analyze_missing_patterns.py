#!/usr/bin/env python3
"""Analyze real missing block patterns in selected AntAWS station CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


META_CSV = Path("data/station_meta_ecbit.csv")
OUTPUT_DIR = Path("results/missing_analysis")
ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "latin1")

VARIABLES = {
    "T": "Temperature(℃)",
    "P": "Pressure(hPa)",
    "wspd": "Wind Speed(m/s)",
    "RH": "Relative Humidity(%)",
}


def read_csv_with_fallback(path: Path, **kwargs) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError(f"Could not read {path}") from last_error


def missing_blocks(mask: np.ndarray) -> list[int]:
    """Return consecutive missing block lengths for a 1D boolean mask."""
    if mask.size == 0:
        return []
    padded = np.r_[False, mask.astype(bool), False]
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return (ends - starts).astype(int).tolist()


def summarize_blocks(blocks: list[int], n_steps: int) -> dict[str, float | int]:
    arr = np.asarray(blocks, dtype=float)
    missing_steps = int(arr.sum()) if arr.size else 0
    if arr.size == 0:
        return {
            "missing_rate": 0.0,
            "block_count": 0,
            "mean_block_steps": 0.0,
            "median_block_steps": 0.0,
            "p90_block_steps": 0.0,
            "max_block_steps": 0,
            "mean_block_hours": 0.0,
            "max_block_days": 0.0,
            "long_block_count_ge_72": 0,
            "long_block_missing_share": 0.0,
        }
    long = arr[arr >= 72]
    return {
        "missing_rate": missing_steps / n_steps,
        "block_count": int(arr.size),
        "mean_block_steps": float(arr.mean()),
        "median_block_steps": float(np.median(arr)),
        "p90_block_steps": float(np.percentile(arr, 90)),
        "max_block_steps": int(arr.max()),
        "mean_block_hours": float(arr.mean() * 3),
        "max_block_days": float(arr.max() * 3 / 24),
        "long_block_count_ge_72": int(long.size),
        "long_block_missing_share": float(long.sum() / missing_steps) if missing_steps else 0.0,
    }


def analyze(meta_csv: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = pd.read_csv(meta_csv)
    selected = meta[meta["split"].isin(["main", "heldout"])].copy()
    rows = []
    for _, station in selected.iterrows():
        csv_path = Path(station["file"])
        df = read_csv_with_fallback(csv_path, usecols=list(VARIABLES.values()))
        n_steps = len(df)
        for var_name, col in VARIABLES.items():
            values = pd.to_numeric(df[col], errors="coerce")
            blocks = missing_blocks(values.isna().to_numpy())
            row = {
                "station": station["station"],
                "station_id": station["station_id"],
                "split": station["split"],
                "variable": var_name,
                "n_steps": n_steps,
                **summarize_blocks(blocks, n_steps),
            }
            rows.append(row)

    stats = pd.DataFrame(rows)
    summary = (
        stats.groupby(["split", "variable"], as_index=False)
        .agg(
            stations=("station", "nunique"),
            mean_missing_rate=("missing_rate", "mean"),
            median_missing_rate=("missing_rate", "median"),
            mean_block_steps=("mean_block_steps", "mean"),
            median_p90_block_steps=("p90_block_steps", "median"),
            max_block_steps=("max_block_steps", "max"),
            total_blocks=("block_count", "sum"),
            total_long_blocks_ge_72=("long_block_count_ge_72", "sum"),
            mean_long_block_missing_share=("long_block_missing_share", "mean"),
        )
        .sort_values(["split", "variable"])
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stats.to_csv(output_dir / "pattern_stats.csv", index=False)
    summary.to_csv(output_dir / "pattern_summary.csv", index=False)
    return stats, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-csv", type=Path, default=META_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    stats, summary = analyze(args.meta_csv, args.output_dir)
    print(f"Wrote {args.output_dir / 'pattern_stats.csv'} ({len(stats)} rows)")
    print(f"Wrote {args.output_dir / 'pattern_summary.csv'} ({len(summary)} rows)")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
