#!/usr/bin/env python3
"""Create a compact inventory of reproducibility artifacts."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


ARTIFACTS = [
    ("station_metadata", "data/station_meta_ecbit.csv", "Selected station metadata, completeness, and held-out flag."),
    ("preprocess_manifest", "data/antaws_impute_manifest.csv", "Per-station preprocessed window counts and file paths."),
    ("era5_manifest", "data/era5_3h_manifest.csv", "Validation manifest for aligned station-level ERA5 files."),
    ("window_index", "experiments/results/analysis/benchmark_protocol/window_index.csv", "Station, split, and timestamp index for every retained window."),
    ("station_window_summary", "experiments/results/analysis/benchmark_protocol/station_window_summary.csv", "Per-station train/validation/test window counts."),
    ("block_length_diagnostics", "experiments/results/analysis/benchmark_protocol/realized_block_length_summary.csv", "Requested, clipped, and label-segment block-length diagnostics."),
    ("era5_direct_calibration", "experiments/results/analysis/era5_direct_calibration/era5_direct_calibration_runs.csv", "No-bias, mean-bias, and linear train-only ERA5-direct calibration runs."),
    ("real_gap_plausibility", "experiments/results/analysis/real_gap_plausibility/real_gap_plausibility_gaps.csv", "Historical-gap boundary plausibility metrics."),
    ("observed_consistency", "experiments/results/analysis/observed_consistency_mcar_noera5/observed_consistency_summary.csv", "Visible-position pre-copy and post-copy consistency audit."),
    ("station_metadata_revision", "experiments/results/revision/station_metadata_table.csv", "Selected station metadata table used by the revision."),
    ("era5_calibration_revision", "experiments/results/revision/era5_calibration_comparison.csv", "ERA5 direct calibration comparison including station-month train-only bias correction."),
    ("heldout_station_revision", "experiments/results/revision/heldout_per_station_results.csv", "Held-out station and per-variable imputation results."),
    ("heldout_bias_revision", "experiments/results/revision/heldout_era5_bias_correlation.csv", "Held-out ERA5-AWS mismatch diagnostic by station."),
    ("round1_runs", "experiments/results/tables/round1_core_runs.csv", "Core baseline per-run test metrics."),
    ("fair_era5_runs", "experiments/results/tables/fair_era5_baselines_runs.csv", "SAITS+ERA5 and iTransformer+ERA5 per-run test metrics."),
    ("fair_era5_comparison", "experiments/results/tables/fair_era5_comparison_summary.csv", "Fair ERA5-augmented baseline summary table."),
    ("fair_era5_tests", "experiments/results/tables/fair_era5_paired_tests.csv", "Paired tests for fair ERA5-augmented baselines."),
    ("round2_gated_runs", "experiments/results/tables/round2_gated_final_runs.csv", "ECBIT gated, concat, no-ERA5, and MCAR per-run test metrics."),
    ("round3_heldout_runs", "experiments/results/tables/round3_final_runs.csv", "Held-out station per-run test metrics."),
    ("curriculum_factorial", "experiments/results/tables/curriculum_era5_factorial.csv", "Block-vs-MCAR and ERA5-vs-no-ERA5 factorial summary."),
    ("raw_unit_variables", "experiments/results/tables/round2_raw_unit_variable_summary.csv", "Variable-level physical-unit error summary."),
    ("local_environment", "experiments/results/analysis/environment/local_environment.txt", "Local preprocessing, plotting, and LaTeX environment summary."),
    ("remote_environment", "experiments/results/analysis/environment/remote_environment.txt", "Remote GPU training and evaluation environment summary."),
]


def sha256_12(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def count_files(path: Path, pattern: str) -> int:
    return sum(1 for _ in path.glob(pattern)) if path.exists() else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("experiments/results/tables/reproducibility_inventory.csv"))
    args = parser.parse_args()

    rows = []
    for name, rel_path, description in ARTIFACTS:
        path = Path(rel_path)
        exists = path.exists()
        rows.append(
            {
                "artifact": name,
                "path": rel_path,
                "exists": exists,
                "bytes": path.stat().st_size if exists else 0,
                "sha256_12": sha256_12(path) if exists and path.is_file() else "",
                "description": description,
            }
        )

    for config_dir in sorted(Path("experiments/configs").iterdir()):
        if config_dir.is_dir():
            rows.append(
                {
                    "artifact": f"configs_{config_dir.name}",
                    "path": str(config_dir),
                    "exists": True,
                    "bytes": 0,
                    "sha256_12": "",
                    "description": f"{count_files(config_dir, '*.yaml')} YAML experiment configurations.",
                }
            )

    metric_root = Path("experiments/results/metrics")
    if metric_root.exists():
        for result_dir in sorted(p for p in metric_root.iterdir() if p.is_dir()):
            rows.append(
                {
                    "artifact": f"metrics_{result_dir.name}",
                    "path": str(result_dir),
                    "exists": True,
                    "bytes": 0,
                    "sha256_12": "",
                    "description": f"{count_files(result_dir, '*/result.json')} per-run result JSON files.",
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
