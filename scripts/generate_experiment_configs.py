#!/usr/bin/env python3
"""Generate ECBIT experiment YAML configs for remote execution."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


OUT_ROOT = Path("experiments/configs")
META_CSV = Path("data/station_meta_ecbit.csv")
SEEDS = [42, 43, 44]
PATTERNS = ["short", "medium", "long"]
RATES = [0.2, 0.4, 0.6]


def base_config(seed: int, output_dir: str) -> dict:
    return {
        "seed": seed,
        "device": "cuda",
        "data": {
            "manifest_csv": "data/antaws_impute_manifest.csv",
            "train_station_groups": ["main"],
            "val_station_groups": ["main"],
            "test_station_groups": ["main"],
        },
        "training": {
            "epochs": 50,
            "batch_size": 32,
            "num_workers": 4,
            "lr": 1.0e-4,
            "weight_decay": 1.0e-4,
            "patience": 10,
        },
        "eval": {"batch_size": 64, "num_workers": 4},
        "output_dir": output_dir,
    }


def model_config(model_name: str, variant: str | None = None, fusion_type: str | None = None) -> dict:
    common = {
        "seq_len": 168,
        "n_vars": 5,
        "n_time": 4,
        "d_model": 128,
        "n_heads": 8,
        "n_layers": 3,
        "d_ff": 256,
        "dropout": 0.1,
    }
    if model_name == "ecbit":
        variant = variant or "full"
        settings = {
            "full": {"use_era5": True, "use_cross": fusion_type != "concat"},
            "no_cross": {"use_era5": True, "use_cross": False, "fusion_type": "concat"},
            "no_era5": {"use_era5": False, "use_cross": False},
            "no_blockmask": {"use_era5": True, "use_cross": fusion_type != "concat"},
        }[variant]
        if fusion_type and variant in {"full", "no_blockmask"}:
            settings["fusion_type"] = fusion_type
        return {"name": "ecbit", "variant": variant, **common, **settings}
    if model_name == "itransformer":
        return {"name": "itransformer", **common}
    if model_name in {"linear_interp", "locf", "era5_direct", "saits", "brits"}:
        return {"name": model_name, "seq_len": 168, "n_vars": 5}
    raise ValueError(f"Unknown model {model_name}")


def missing_config(pattern: str, rate: float, force_mcar: bool = False) -> dict:
    ranges = {"short": (6, 24), "medium": (24, 72), "long": (72, 240)}
    min_block, max_block = ranges[pattern]
    if force_mcar:
        return {"pattern": "mcar", "target_pattern": pattern, "rate": rate}
    return {"pattern": pattern, "rate": rate, "min_block": min_block, "max_block": max_block, "variable_wise": True}


def write_yaml(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def generate_round1(out_root: Path) -> int:
    models = ["linear_interp", "locf", "era5_direct", "saits", "brits", "itransformer"]
    count = 0
    for model in models:
        for pattern in PATTERNS:
            for rate in RATES:
                for seed in SEEDS:
                    run_name = f"{model}_{pattern}_r{int(rate*100)}_s{seed}"
                    cfg = base_config(seed, f"experiments/results/metrics/round1/{run_name}")
                    cfg["runner"] = "train" if model == "itransformer" else "evaluate"
                    cfg["model"] = model_config(model)
                    cfg["missing"] = missing_config(pattern, rate)
                    cfg["run_name"] = run_name
                    write_yaml(out_root / "round1" / f"{run_name}.yaml", cfg)
                    count += 1
    return count


def generate_round2(out_root: Path) -> int:
    variants = ["full", "no_cross", "no_era5", "no_blockmask"]
    count = 0
    for variant in variants:
        for pattern in PATTERNS:
            for rate in RATES:
                for seed in SEEDS:
                    run_name = f"ecbit_{variant}_{pattern}_r{int(rate*100)}_s{seed}"
                    cfg = base_config(seed, f"experiments/results/metrics/round2/{run_name}")
                    cfg["runner"] = "train"
                    cfg["model"] = model_config("ecbit", variant)
                    cfg["missing"] = missing_config(pattern, rate, force_mcar=(variant == "no_blockmask"))
                    cfg["run_name"] = run_name
                    write_yaml(out_root / "round2" / f"{run_name}.yaml", cfg)
                    count += 1
    return count


def generate_round2_gated(out_root: Path) -> int:
    variants = ["full", "no_cross", "no_era5", "no_blockmask"]
    count = 0
    for variant in variants:
        for pattern in PATTERNS:
            for rate in RATES:
                for seed in SEEDS:
                    run_name = f"ecbit_{variant}_{pattern}_r{int(rate*100)}_s{seed}"
                    cfg = base_config(seed, f"experiments/results/metrics/round2_gated/{run_name}")
                    cfg["runner"] = "train"
                    cfg["model"] = model_config("ecbit", variant, fusion_type="gated")
                    cfg["missing"] = missing_config(pattern, rate, force_mcar=(variant == "no_blockmask"))
                    cfg["run_name"] = run_name
                    write_yaml(out_root / "round2_gated" / f"{run_name}.yaml", cfg)
                    count += 1
    return count


def generate_round3(out_root: Path, meta_csv: Path) -> int:
    meta = pd.read_csv(meta_csv)
    heldout = meta.loc[meta["split"].eq("heldout"), "station_id"].astype(str).tolist()
    count = 0
    for station_id in heldout:
        for pattern in PATTERNS:
            for seed in SEEDS:
                run_name = f"ecbit_full_heldout_{station_id}_{pattern}_s{seed}"
                cfg = base_config(seed, f"experiments/results/metrics/round3/{run_name}")
                cfg["runner"] = "train"
                cfg["data"]["test_station_groups"] = ["heldout"]
                cfg["data"]["test_station_ids"] = [station_id]
                cfg["model"] = model_config("ecbit", "full")
                cfg["missing"] = missing_config(pattern, 0.4)
                cfg["run_name"] = run_name
                write_yaml(out_root / "round3" / f"{run_name}.yaml", cfg)
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--meta-csv", type=Path, default=META_CSV)
    args = parser.parse_args()
    counts = {
        "round1": generate_round1(args.out_root),
        "round2": generate_round2(args.out_root),
        "round2_gated": generate_round2_gated(args.out_root),
        "round3": generate_round3(args.out_root, args.meta_csv),
    }
    print(counts)


if __name__ == "__main__":
    main()
