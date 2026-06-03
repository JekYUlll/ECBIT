#!/usr/bin/env python3
"""Generate isolated SABC exploration configs for remote execution."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.impute_dataset import STATION_FEATURE_DIM
from scripts.generate_experiment_configs import PATTERNS, SEEDS, base_config, missing_config


OUT_DIR = Path("experiments/configs/exploration_sabc")
META_CSV = Path("data/station_meta_ecbit.csv")
STATION_META_CSV = Path("data/station_meta_ecbit.csv")


def write_yaml(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def ecbit_gated_config() -> dict:
    return {
        "name": "ecbit",
        "variant": "gated_baseline",
        "seq_len": 168,
        "n_vars": 5,
        "n_time": 4,
        "d_model": 128,
        "n_heads": 8,
        "n_layers": 3,
        "d_ff": 256,
        "dropout": 0.1,
        "use_era5": True,
        "use_cross": True,
        "fusion_type": "gated",
    }


def ecbit_sabc_metadata_config() -> dict:
    return {
        "name": "ecbit_sabc",
        "variant": "sabc_metadata",
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
        "sabc": {
            "hidden_dim": 64,
            "dropout": 0.1,
            "residual_scale_init": 0.1,
        },
    }


def heldout_station_ids(meta_csv: Path) -> list[str]:
    meta = pd.read_csv(meta_csv)
    if "split" not in meta.columns or "station_id" not in meta.columns:
        raise ValueError("metadata CSV must contain split and station_id columns")
    heldout = meta.loc[meta["split"].eq("heldout"), "station_id"].astype(str).tolist()
    if not heldout:
        raise ValueError("metadata CSV contains no held-out stations")
    return heldout


def generate_sabc_configs(out_dir: Path = OUT_DIR, meta_csv: Path = META_CSV, include_baseline: bool = True) -> int:
    count = 0
    model_variants: list[tuple[str, dict]] = [("sabc_metadata", ecbit_sabc_metadata_config())]
    if include_baseline:
        model_variants.insert(0, ("gated_baseline", ecbit_gated_config()))

    for station_id in heldout_station_ids(meta_csv):
        for pattern in PATTERNS:
            for seed in SEEDS:
                for variant_name, model_cfg in model_variants:
                    run_name = f"{variant_name}_heldout_{station_id}_{pattern}_r40_s{seed}"
                    cfg = base_config(seed, f"experiments/results/metrics/exploration_sabc/{run_name}")
                    cfg["runner"] = "train"
                    cfg["data"]["station_meta_csv"] = str(STATION_META_CSV)
                    cfg["data"]["test_station_groups"] = ["heldout"]
                    cfg["data"]["test_station_ids"] = [station_id]
                    cfg["model"] = model_cfg
                    cfg["missing"] = missing_config(pattern, 0.4)
                    cfg["run_name"] = run_name
                    write_yaml(out_dir / f"{run_name}.yaml", cfg)
                    count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--meta-csv", type=Path, default=META_CSV)
    parser.add_argument("--sabc-only", action="store_true", help="Do not regenerate matched gated-baseline configs")
    args = parser.parse_args()
    count = generate_sabc_configs(args.out_dir, args.meta_csv, include_baseline=not args.sabc_only)
    print(f"Wrote {count} configs to {args.out_dir}")


if __name__ == "__main__":
    main()
