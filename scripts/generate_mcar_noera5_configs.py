#!/usr/bin/env python3
"""Generate MCAR-trained no-ERA5 configs for the curriculum factorial check."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from generate_experiment_configs import SEEDS, PATTERNS, RATES, base_config, model_config, missing_config


OUT_DIR = Path("experiments/configs/followup_mcar_noera5")


def write_yaml(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def generate(out_dir: Path) -> int:
    count = 0
    for pattern in PATTERNS:
        for rate in RATES:
            for seed in SEEDS:
                run_name = f"ecbit_mcar_no_era5_{pattern}_r{int(rate * 100)}_s{seed}"
                cfg = base_config(seed, f"experiments/results/metrics/followup_mcar_noera5/{run_name}")
                cfg["runner"] = "train"
                cfg["model"] = model_config("ecbit", "no_era5")
                cfg["missing"] = missing_config(pattern, rate, force_mcar=True)
                cfg["run_name"] = run_name
                write_yaml(out_dir / f"{run_name}.yaml", cfg)
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    count = generate(args.out_dir)
    print(f"wrote {count} configs to {args.out_dir}")


if __name__ == "__main__":
    main()
