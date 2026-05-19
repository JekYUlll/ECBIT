#!/usr/bin/env python3
"""Aggregate ECBIT experiment result.json files into run and summary tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


METRIC_KEYS = [
    "mae_mean",
    "rmse_mean",
    "mae_T",
    "mae_RH",
    "mae_wspd",
    "mae_P",
    "mae_q",
    "rmse_T",
    "rmse_RH",
    "rmse_wspd",
    "rmse_P",
    "rmse_q",
]


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    if "test" in result:
        metrics = dict(result["test"])
        if "best" in result and isinstance(result["best"], dict):
            metrics["best_epoch"] = result["best"].get("epoch")
            metrics["val_mae_mean"] = result["best"].get("mae_mean")
            metrics["val_rmse_mean"] = result["best"].get("rmse_mean")
        return metrics
    if "best" in result:
        metrics = dict(result["best"])
        metrics["best_epoch"] = metrics.pop("epoch", None)
        return metrics
    return dict(result)


def row_from_result(result_path: Path, config_dir: Path) -> dict[str, Any]:
    run_name = result_path.parent.name
    config_path = config_dir / f"{run_name}.yaml"
    config = load_yaml(config_path) if config_path.exists() else {}
    result = load_json(result_path)
    metrics = flatten_result(result)
    model_cfg = config.get("model", result.get("config", {}).get("model", {}))
    missing_cfg = config.get("missing", result.get("config", {}).get("missing", {}))
    row = {
        "run_name": run_name,
        "model": model_cfg.get("name", run_name.split("_")[0]),
        "variant": model_cfg.get("variant", ""),
        "pattern": missing_cfg.get("pattern", ""),
        "target_pattern": missing_cfg.get("target_pattern", ""),
        "rate": missing_cfg.get("rate", ""),
        "seed": config.get("seed", result.get("config", {}).get("seed", "")),
        "result_path": str(result_path),
    }
    for key in METRIC_KEYS:
        row[key] = metrics.get(key)
    if "best_epoch" in metrics:
        row["best_epoch"] = metrics["best_epoch"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/results/tables"))
    parser.add_argument("--prefix", default=None)
    args = parser.parse_args()

    rows = [
        row_from_result(path, args.config_dir)
        for path in sorted(args.metrics_dir.glob("*/result.json"))
    ]
    if not rows:
        raise SystemExit(f"No result.json files found under {args.metrics_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or args.metrics_dir.name
    run_df = pd.DataFrame(rows).sort_values(["model", "variant", "pattern", "rate", "seed"])
    run_csv = args.out_dir / f"{prefix}_runs.csv"
    run_df.to_csv(run_csv, index=False)

    group_cols = ["model", "variant", "pattern", "rate"]
    summary = (
        run_df.groupby(group_cols, dropna=False)[METRIC_KEYS]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in col if part)
        if isinstance(col, tuple)
        else str(col)
        for col in summary.columns
    ]
    summary_csv = args.out_dir / f"{prefix}_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"wrote {run_csv} ({len(run_df)} runs)")
    print(f"wrote {summary_csv} ({len(summary)} groups)")


if __name__ == "__main__":
    main()
