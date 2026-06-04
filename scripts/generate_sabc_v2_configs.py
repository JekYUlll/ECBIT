#!/usr/bin/env python3
"""Generate residual-aware SABC v2 pilot configs for remote training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_experiment_configs import PATTERNS, SEEDS, base_config, missing_config
from src.data.impute_dataset import RESIDUAL_FEATURE_DIM, STATION_FEATURE_DIM


OUT_DIR = Path("experiments/configs/exploration_sabc_v2")
OUTPUT_ROOT = Path("experiments/results/metrics/exploration_sabc_v2")
RESIDUAL_FEATURE_CSV = Path("experiments/results/exploration_sabc_v2/sabc_residual_features.csv")
META_CSV = Path("data/station_meta_ecbit.csv")
DEFAULT_STATIONS = ("mount_sidley", "nico", "zhongshan")


def write_yaml(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def selected_heldout_stations(meta_csv: Path, requested: list[str]) -> list[str]:
    meta = pd.read_csv(meta_csv)
    heldout = set(meta.loc[meta["split"].eq("heldout"), "station_id"].astype(str))
    missing = [station for station in requested if station not in heldout]
    if missing:
        raise ValueError(f"Requested stations are not held-out stations in {meta_csv}: {missing}")
    return requested


def ecbit_sabc_residual_config() -> dict:
    return {
        "name": "ecbit_sabc",
        "variant": "sabc_residual",
        "seq_len": 168,
        "n_vars": 5,
        "n_time": 4,
        "d_model": 128,
        "n_heads": 8,
        "n_layers": 3,
        "d_ff": 256,
        "dropout": 0.1,
        "fusion_type": "gated",
        "n_station_features": STATION_FEATURE_DIM,
        "n_residual_features": RESIDUAL_FEATURE_DIM,
        "sabc": {
            "hidden_dim": 64,
            "dropout": 0.1,
            "residual_scale_init": 0.1,
        },
    }


def generate_configs(
    out_dir: Path,
    output_root: Path,
    residual_feature_csv: Path,
    meta_csv: Path,
    stations: list[str],
) -> int:
    stations = selected_heldout_stations(meta_csv, stations)
    count = 0
    for station_id in stations:
        for pattern in PATTERNS:
            for seed in SEEDS:
                run_name = f"sabc_residual_heldout_{station_id}_{pattern}_r40_s{seed}"
                cfg = base_config(seed, str(output_root / run_name))
                cfg["runner"] = "train"
                cfg["data"]["station_meta_csv"] = str(meta_csv)
                cfg["data"]["residual_feature_csv"] = str(residual_feature_csv)
                cfg["data"]["test_station_groups"] = ["heldout"]
                cfg["data"]["test_station_ids"] = [station_id]
                cfg["model"] = ecbit_sabc_residual_config()
                cfg["missing"] = missing_config(pattern, 0.4)
                cfg["run_name"] = run_name
                write_yaml(out_dir / f"{run_name}.yaml", cfg)
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--residual-feature-csv", type=Path, default=RESIDUAL_FEATURE_CSV)
    parser.add_argument("--meta-csv", type=Path, default=META_CSV)
    parser.add_argument("--stations", default=",".join(DEFAULT_STATIONS))
    args = parser.parse_args()

    stations = [part.strip() for part in args.stations.split(",") if part.strip()]
    count = generate_configs(
        out_dir=args.out_dir,
        output_root=args.output_root,
        residual_feature_csv=args.residual_feature_csv,
        meta_csv=args.meta_csv,
        stations=stations,
    )
    print(f"Wrote {count} SABC v2 configs to {args.out_dir}")


if __name__ == "__main__":
    main()
