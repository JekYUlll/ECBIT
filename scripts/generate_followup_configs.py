#!/usr/bin/env python3
"""Generate rapid follow-up experiment configs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_experiment_configs import base_config, model_config, write_yaml


SEEDS = [42, 43, 44]
BLOCK_LENGTHS_H = {
    "24h": (2, 8),
    "72h": (6, 24),
    "216h": (24, 72),
}


def generate_blocklen(out_root: Path) -> int:
    count = 0
    for length_label, (min_block, max_block) in BLOCK_LENGTHS_H.items():
        for variant in ["full", "no_era5"]:
            for seed in SEEDS:
                run_name = f"ecbit_blocklen_{length_label}_{variant}_s{seed}"
                cfg = base_config(seed, f"experiments/results/metrics/followup_blocklen/{run_name}")
                cfg["runner"] = "train"
                cfg["model"] = model_config("ecbit", variant, fusion_type="gated")
                cfg["missing"] = {
                    "pattern": f"blocklen_{length_label}",
                    "rate": 0.4,
                    "min_block": min_block,
                    "max_block": max_block,
                    "variable_wise": True,
                }
                cfg["analysis"] = {
                    "block_max_hours": int(length_label.rstrip("h")),
                    "block_min_steps": min_block,
                    "block_max_steps": max_block,
                }
                cfg["run_name"] = run_name
                write_yaml(out_root / "followup_blocklen" / f"{run_name}.yaml", cfg)
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=Path("experiments/configs"))
    args = parser.parse_args()
    counts = {"followup_blocklen": generate_blocklen(args.out_root)}
    print(yaml.safe_dump(counts, sort_keys=True))


if __name__ == "__main__":
    main()
