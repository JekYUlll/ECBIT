#!/usr/bin/env python3
"""Recover neural result.json files from best.pt checkpoints.

The recovered result stores validation metrics under ``best`` and test metrics
under ``test``. Aggregation uses ``test`` when present.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate_impute import evaluate_neural


def default_root() -> Path:
    if os.environ.get("ECBIT_ROOT"):
        return Path(os.environ["ECBIT_ROOT"]).expanduser()
    home_copy = Path.home() / "ecbit"
    if home_copy.exists():
        return home_copy
    return Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root())
    parser.add_argument("--metrics-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Rewrite result.json even when it already exists")
    parser.add_argument("--device", default=None, help="Temporarily override config device, e.g. cpu")
    parser.add_argument(
        "--completed-only",
        action="store_true",
        help="Only recover runs that already have result.json; useful while workers are still training",
    )
    args = parser.parse_args()

    metrics_dir = args.metrics_dir or args.root / "experiments/results/metrics/round1"
    recovered = 0
    for run_dir in sorted(metrics_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        best_pt = run_dir / "best.pt"
        result_json = run_dir / "result.json"
        if not best_pt.exists():
            continue
        if args.completed_only and not result_json.exists():
            continue
        if result_json.exists() and not args.overwrite:
            continue
        try:
            checkpoint = torch.load(str(best_pt), map_location="cpu", weights_only=True)
            config = checkpoint["config"]
            if args.device:
                config = dict(config)
                config["device"] = args.device
            test_metrics = evaluate_neural(config, best_pt, "test")
            result = {"best": checkpoint["metrics"], "test": test_metrics, "config": config}
            with open(result_json, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            mae = result["test"].get("mae_mean", "N/A")
            print(f"RECOVERED: {run_dir.name} (test_mae_mean={mae})")
            recovered += 1
        except Exception as exc:
            print(f"FAILED: {run_dir.name}: {type(exc).__name__}: {exc}")

    print(f"Total recovered: {recovered}")


if __name__ == "__main__":
    main()
