#!/usr/bin/env python3
"""Generate fair ERA5-augmented baseline configs for the major revision."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_experiment_configs import PATTERNS, RATES, SEEDS, base_config, missing_config


OUT_DIR = Path("experiments/configs/fair_era5_baselines")


def write_yaml(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def itransformer_era5_config() -> dict:
    return {
        "name": "itransformer_era5",
        "seq_len": 168,
        "n_vars": 5,
        "n_time": 4,
        "d_model": 128,
        "n_heads": 8,
        "n_layers": 3,
        "d_ff": 256,
        "dropout": 0.1,
    }


def saits_era5_config() -> dict:
    return {
        "name": "saits_era5_concat",
        "seq_len": 168,
        "n_vars": 5,
        "n_features": 10,
        "model_kwargs": {
            "n_layers": 2,
            "d_model": 128,
            "n_heads": 4,
            "d_k": 32,
            "d_v": 32,
            "d_ffn": 256,
            "dropout": 0.1,
            "attn_dropout": 0.1,
            "batch_size": 32,
            "epochs": 50,
            "patience": 10,
            "num_workers": 0,
            "verbose": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    count = 0
    for model_name, model_cfg, runner in [
        ("itransformer_era5", itransformer_era5_config(), "train"),
        ("saits_era5_concat", saits_era5_config(), "evaluate"),
    ]:
        for pattern in PATTERNS:
            for rate in RATES:
                for seed in SEEDS:
                    run_name = f"{model_name}_{pattern}_r{int(rate * 100)}_s{seed}"
                    cfg = base_config(seed, f"experiments/results/metrics/fair_era5_baselines/{run_name}")
                    cfg["runner"] = runner
                    cfg["model"] = model_cfg
                    cfg["missing"] = missing_config(pattern, rate)
                    cfg["run_name"] = run_name
                    write_yaml(args.out_dir / f"{run_name}.yaml", cfg)
                    count += 1
    print(f"Wrote {count} configs to {args.out_dir}")


if __name__ == "__main__":
    main()
