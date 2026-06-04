#!/usr/bin/env python3
"""Generate train-period ERA5-AWS residual summaries for SABC v2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.impute_dataset import RESIDUAL_FEATURE_COLUMNS
from src.metrics import VARIABLES


DEFAULT_OUT = Path("experiments/results/exploration_sabc_v2/sabc_residual_features.csv")


def split_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def month_array(data: np.lib.npyio.NpzFile, keep: np.ndarray) -> np.ndarray:
    if "timestamps" not in data.files:
        return np.ones(data["X"][keep].shape[:2], dtype=np.int64)
    return data["timestamps"][keep].astype("datetime64[M]").astype(int) % 12 + 1


def residual_stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"bias": float("nan"), "std": float("nan"), "mae": float("nan"), "count": 0.0}
    return {
        "bias": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "mae": float(np.abs(values).mean()),
        "count": float(values.size),
    }


def scaled_log_count(count: float) -> float:
    return float(np.log1p(max(count, 0.0)) / 12.0)


def feature_row(
    station_id: str,
    month: int,
    variable_index: int,
    variable: str,
    station_stats: dict[str, float],
    month_stats: dict[str, float],
) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "station_id": station_id,
        "month": int(month),
        "variable_index": int(variable_index),
        "variable": variable,
        "station_bias": float(station_stats["bias"]),
        "station_std": float(station_stats["std"]),
        "station_mae": float(station_stats["mae"]),
        "station_log_count": scaled_log_count(station_stats["count"]),
        "station_month_bias": float(month_stats["bias"]),
        "station_month_std": float(month_stats["std"]),
        "station_month_mae": float(month_stats["mae"]),
        "station_month_log_count": scaled_log_count(month_stats["count"]),
    }
    return row


def collect_station_arrays(row: pd.Series, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(Path(row["path"]), allow_pickle=True)
    window_split = data["window_split"].astype(str)
    keep = window_split == split
    if not keep.any():
        n_vars = int(data["X"].shape[-1])
        return (
            np.empty((0, 0, n_vars), dtype=np.float32),
            np.empty((0, 0, n_vars), dtype=np.float32),
            np.empty((0, 0, n_vars), dtype=bool),
            np.empty((0, 0), dtype=np.int64),
        )
    x = data["X"][keep].astype(np.float32)
    era5 = data["E_3h"][keep].astype(np.float32)
    mask = data["obs_mask"][keep].astype(bool)
    months = month_array(data, keep).astype(np.int64)
    finite = np.isfinite(x) & np.isfinite(era5)
    return x, era5, mask & finite, months


def global_fallbacks(items: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> dict[int, dict[str, float]]:
    n_vars = max((x.shape[-1] for _, x, _, _, _ in items if x.ndim == 3 and x.shape[-1] > 0), default=len(VARIABLES))
    out: dict[int, dict[str, float]] = {}
    for c in range(n_vars):
        parts = []
        for _, x, era5, mask, _ in items:
            if x.size == 0:
                continue
            valid = mask[:, :, c]
            if valid.any():
                parts.append((x[:, :, c] - era5[:, :, c])[valid])
        values = np.concatenate(parts) if parts else np.asarray([], dtype=np.float32)
        stats = residual_stats(values)
        if stats["count"] == 0:
            stats = {"bias": 0.0, "std": 0.0, "mae": 0.0, "count": 0.0}
        out[c] = stats
    return out


def generate_features(
    manifest_csv: Path,
    out_csv: Path,
    station_groups: set[str],
    split: str,
    min_month_samples: int,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_csv)
    manifest = manifest[manifest["station_group"].isin(station_groups)].copy()
    if manifest.empty:
        raise ValueError(f"No stations match groups: {sorted(station_groups)}")

    loaded = [
        (str(row["station_id"]), *collect_station_arrays(row, split))
        for _, row in manifest.iterrows()
    ]
    global_stats = global_fallbacks(loaded)
    rows: list[dict[str, Any]] = []
    for station_id, x, era5, mask, months in loaded:
        n_vars = int(x.shape[-1]) if x.ndim == 3 and x.shape[-1] else len(VARIABLES)
        for c in range(n_vars):
            valid = mask[:, :, c] if x.size else np.zeros((0, 0), dtype=bool)
            station_values = (x[:, :, c] - era5[:, :, c])[valid] if valid.any() else np.asarray([], dtype=np.float32)
            station_stats = residual_stats(station_values)
            if station_stats["count"] == 0:
                station_stats = dict(global_stats[c])

            for month in range(1, 13):
                month_valid = valid & (months == month) if x.size else np.zeros((0, 0), dtype=bool)
                month_values = (
                    (x[:, :, c] - era5[:, :, c])[month_valid]
                    if month_valid.any()
                    else np.asarray([], dtype=np.float32)
                )
                month_stats = residual_stats(month_values)
                if month_stats["count"] < min_month_samples:
                    month_stats = {
                        "bias": station_stats["bias"],
                        "std": station_stats["std"],
                        "mae": station_stats["mae"],
                        "count": month_stats["count"],
                    }
                rows.append(
                    feature_row(
                        station_id=station_id,
                        month=month,
                        variable_index=c,
                        variable=VARIABLES[c] if c < len(VARIABLES) else f"var{c}",
                        station_stats=station_stats,
                        month_stats=month_stats,
                    )
                )

    out = pd.DataFrame(rows)
    out = out[["station_id", "month", "variable_index", "variable", *RESIDUAL_FEATURE_COLUMNS]]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    summary = {
        "rows": int(len(out)),
        "stations": int(out["station_id"].nunique()),
        "variables": int(out["variable_index"].nunique()),
        "feature_columns": list(RESIDUAL_FEATURE_COLUMNS),
        "manifest_csv": str(manifest_csv),
        "station_groups": sorted(station_groups),
        "split": split,
        "min_month_samples": int(min_month_samples),
    }
    with open(out_csv.with_suffix(".summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", type=Path, default=Path("data/antaws_impute_manifest.csv"))
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--station-groups", default="main,heldout")
    parser.add_argument("--split", default="train")
    parser.add_argument("--min-month-samples", type=int, default=128)
    args = parser.parse_args()

    table = generate_features(
        manifest_csv=args.manifest_csv,
        out_csv=args.out_csv,
        station_groups=set(split_list(args.station_groups)),
        split=args.split,
        min_month_samples=args.min_month_samples,
    )
    print(f"Wrote {len(table)} residual-feature rows to {args.out_csv}")


if __name__ == "__main__":
    main()
