#!/usr/bin/env python3
"""Generate isolated physical-consistency regularization configs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_experiment_configs import PATTERNS, SEEDS, base_config, missing_config, model_config, write_yaml


OUT_DIR = Path("experiments/configs/exploration_pcr")
OUTPUT_ROOT = Path("experiments/results/metrics/exploration_pcr")


def generate(out_dir: Path = OUT_DIR, output_root: Path = OUTPUT_ROOT, weight: float = 0.05) -> int:
    count = 0
    for pattern in PATTERNS:
        for seed in SEEDS:
            for variant, use_pcr in [("gated_baseline", False), ("pcr_w005", True)]:
                run_name = f"{variant}_{pattern}_r40_s{seed}"
                cfg = base_config(seed, str(output_root / run_name))
                cfg["runner"] = "train"
                cfg["model"] = model_config("ecbit", "full", fusion_type="gated")
                cfg["model"]["variant"] = variant
                cfg["missing"] = missing_config(pattern, 0.4)
                cfg["run_name"] = run_name
                if use_pcr:
                    cfg["loss"] = {
                        "physical_consistency": {
                            "enabled": True,
                            "weight": float(weight),
                            "delta": 1.0,
                        }
                    }
                write_yaml(out_dir / f"{run_name}.yaml", cfg)
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--weight", type=float, default=0.05)
    args = parser.parse_args()
    count = generate(args.out_dir, args.output_root, args.weight)
    print(f"Wrote {count} configs to {args.out_dir}")


if __name__ == "__main__":
    main()
