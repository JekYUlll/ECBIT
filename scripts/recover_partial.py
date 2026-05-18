#!/usr/bin/env python3
"""Recover result.json from best.pt for partially completed neural runs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


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
    args = parser.parse_args()

    metrics_dir = args.metrics_dir or args.root / "experiments/results/metrics/round1"
    recovered = 0
    for run_dir in sorted(metrics_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        best_pt = run_dir / "best.pt"
        result_json = run_dir / "result.json"
        if not best_pt.exists() or result_json.exists():
            continue
        try:
            checkpoint = torch.load(str(best_pt), map_location="cpu", weights_only=True)
            result = {"best": checkpoint["metrics"], "config": checkpoint["config"]}
            with open(result_json, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            mae = result["best"].get("mae_mean", "N/A")
            print(f"RECOVERED: {run_dir.name} (mae_mean={mae})")
            recovered += 1
        except Exception as exc:
            print(f"FAILED: {run_dir.name}: {type(exc).__name__}: {exc}")

    print(f"Total recovered: {recovered}")


if __name__ == "__main__":
    main()
