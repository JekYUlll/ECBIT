#!/usr/bin/env python3
"""Validate aligned ERA5 3h files for selected ECBIT stations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def validate(meta_csv: Path, era5_dir: Path, manifest_path: Path | None) -> pd.DataFrame:
    meta = pd.read_csv(meta_csv)
    selected = meta[meta["split"].isin(["main", "heldout"])].copy()
    rows = []
    failures = []

    for _, row in selected.iterrows():
        sid = row["station_id"]
        path = era5_dir / f"{sid}_era5_3h.npz"
        record = {
            "station": row["station"],
            "station_id": sid,
            "split": row["split"],
            "expected_steps": int(row["observed_steps"]),
            "path": str(path),
            "exists": path.exists(),
            "shape": "",
            "finite": False,
            "nan_count": "",
        }
        if not path.exists():
            failures.append(f"{sid}: missing {path}")
            rows.append(record)
            continue

        data = np.load(path, allow_pickle=True)
        era5_met = data["era5_met"]
        timestamps = data["timestamps"]
        record["shape"] = f"{era5_met.shape[0]}x{era5_met.shape[1]}"
        record["finite"] = bool(np.isfinite(era5_met).all())
        record["nan_count"] = int(np.isnan(era5_met).sum())

        if era5_met.shape != (int(row["observed_steps"]), 5):
            failures.append(f"{sid}: shape {era5_met.shape}, expected {(int(row['observed_steps']), 5)}")
        if len(timestamps) != int(row["observed_steps"]):
            failures.append(f"{sid}: timestamp length {len(timestamps)}, expected {int(row['observed_steps'])}")
        if not record["finite"]:
            failures.append(f"{sid}: non-finite ERA5 values")
        rows.append(record)

    out = pd.DataFrame(rows)
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(manifest_path, index=False)
    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        raise SystemExit(f"ERA5 validation failed for {len(failures)} issue(s)")
    print(f"ERA5 validation passed for {len(selected)} selected stations")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-csv", type=Path, default=Path("data/station_meta_ecbit.csv"))
    parser.add_argument("--era5-dir", type=Path, default=Path("data/era5_3h"))
    parser.add_argument("--manifest", type=Path, default=Path("data/era5_3h_manifest.csv"))
    args = parser.parse_args()
    validate(args.meta_csv, args.era5_dir, args.manifest)


if __name__ == "__main__":
    main()
