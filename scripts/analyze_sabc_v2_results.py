#!/usr/bin/env python3
"""Aggregate residual-aware SABC v2 results against existing gated baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats


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
        if isinstance(result.get("best"), dict):
            metrics["best_epoch"] = result["best"].get("epoch")
            metrics["val_mae_mean"] = result["best"].get("mae_mean")
        return metrics
    return dict(result.get("best", result))


def row_from_result(result_path: Path, config_dir: Path) -> dict[str, Any]:
    run_name = result_path.parent.name
    config_path = config_dir / f"{run_name}.yaml"
    result = load_json(result_path)
    config = load_yaml(config_path) if config_path.exists() else result.get("config", {})
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    missing_cfg = config.get("missing", {})
    metrics = flatten_result(result)
    row = {
        "run_name": run_name,
        "model": model_cfg.get("name", ""),
        "variant": model_cfg.get("variant", ""),
        "station_id": str((data_cfg.get("test_station_ids") or [""])[0]),
        "pattern": missing_cfg.get("pattern", ""),
        "rate": float(missing_cfg.get("rate", np.nan)),
        "seed": int(config.get("seed", -1)),
        "result_path": str(result_path),
    }
    for key in METRIC_KEYS:
        row[key] = metrics.get(key)
    if "best_epoch" in metrics:
        row["best_epoch"] = metrics["best_epoch"]
    if "val_mae_mean" in metrics:
        row["val_mae_mean"] = metrics["val_mae_mean"]
    return row


def collect(metrics_dir: Path, config_dir: Path) -> pd.DataFrame:
    paths = sorted(metrics_dir.glob("*/result.json"))
    if not paths:
        return pd.DataFrame()
    return pd.DataFrame(row_from_result(path, config_dir) for path in paths)


def paired_test(values: np.ndarray) -> dict[str, float | int]:
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        return {"n": 0, "mean_improvement": np.nan, "ci95_low": np.nan, "ci95_high": np.nan, "t": np.nan, "p": np.nan}
    mean = float(values.mean())
    if n < 2:
        return {"n": n, "mean_improvement": mean, "ci95_low": np.nan, "ci95_high": np.nan, "t": np.nan, "p": np.nan}
    sd = float(values.std(ddof=1))
    se = sd / np.sqrt(n)
    if se == 0.0:
        return {
            "n": n,
            "mean_improvement": mean,
            "ci95_low": mean,
            "ci95_high": mean,
            "t": 0.0,
            "p": 1.0,
        }
    ci_low, ci_high = stats.t.interval(0.95, df=n - 1, loc=mean, scale=se)
    return {
        "n": n,
        "mean_improvement": mean,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "t": float(mean / se),
        "p": float(stats.ttest_1samp(values, 0.0).pvalue),
    }


def paired_tests(paired: pd.DataFrame) -> pd.DataFrame:
    rows = [{"group": "overall", **paired_test(paired["improvement"].to_numpy(dtype=float))}]
    for station_id, part in paired.groupby("station_id"):
        rows.append({"group": f"station:{station_id}", **paired_test(part["improvement"].to_numpy(dtype=float))})
    for pattern, part in paired.groupby("pattern"):
        rows.append({"group": f"pattern:{pattern}", **paired_test(part["improvement"].to_numpy(dtype=float))})
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for row in frame[columns].itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append("nan" if not np.isfinite(value) else f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep, *rows])


def write_report(out_path: Path, paired: pd.DataFrame, tests: pd.DataFrame) -> None:
    if paired.empty:
        text = "# SABC v2 Gate Report\n\nNo matched residual-SABC and gated-baseline pairs are available yet.\n"
    else:
        by_station = paired.groupby("station_id")["improvement"].mean().sort_index()
        by_pattern = paired.groupby("pattern")["improvement"].mean().sort_index()
        overall = float(paired["improvement"].mean())
        text = "\n".join(
            [
                "# SABC v2 Gate Report",
                "",
                "Improvement is matched gated baseline MAE minus residual-aware SABC v2 MAE.",
                "",
                f"Matched pairs: {len(paired)}",
                f"Overall mean improvement: {overall:.4f} normalized MAE",
                "",
                "## Paired Tests",
                "",
                markdown_table(tests, ["group", "n", "mean_improvement", "ci95_low", "ci95_high", "t", "p"]),
                "",
                "## Station Mean Improvements",
                "",
                markdown_table(by_station.reset_index(name="mean_improvement"), ["station_id", "mean_improvement"]),
                "",
                "## Pattern Mean Improvements",
                "",
                markdown_table(by_pattern.reset_index(name="mean_improvement"), ["pattern", "mean_improvement"]),
                "",
            ]
        )
    out_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, default=Path("experiments/results/metrics/exploration_sabc_v2"))
    parser.add_argument("--config-dir", type=Path, default=Path("experiments/configs/exploration_sabc_v2"))
    parser.add_argument("--baseline-metrics-dir", type=Path, default=Path("experiments/results/metrics/exploration_sabc"))
    parser.add_argument("--baseline-config-dir", type=Path, default=Path("experiments/configs/exploration_sabc"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/results/exploration_sabc_v2"))
    args = parser.parse_args()

    runs = collect(args.metrics_dir, args.config_dir)
    baseline = collect(args.baseline_metrics_dir, args.baseline_config_dir)
    if runs.empty:
        raise SystemExit(f"No result.json files found under {args.metrics_dir}")
    baseline = baseline[baseline["variant"].eq("gated_baseline")].copy()
    idx = ["station_id", "pattern", "rate", "seed"]
    paired = runs[idx + ["mae_mean"]].rename(columns={"mae_mean": "sabc_v2_mae"}).merge(
        baseline[idx + ["mae_mean"]].rename(columns={"mae_mean": "baseline_mae"}),
        on=idx,
        how="inner",
    )
    paired["improvement"] = paired["baseline_mae"] - paired["sabc_v2_mae"]
    tests = paired_tests(paired)
    summary = runs.groupby(["variant", "station_id", "pattern"], dropna=False)[METRIC_KEYS].agg(["mean", "std", "count"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in col if part) if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs.sort_values(["station_id", "pattern", "seed"]).to_csv(args.out_dir / "sabc_v2_runs.csv", index=False)
    summary.to_csv(args.out_dir / "sabc_v2_summary.csv", index=False)
    paired.to_csv(args.out_dir / "sabc_v2_paired_runs.csv", index=False)
    tests.to_csv(args.out_dir / "sabc_v2_paired_tests.csv", index=False)
    write_report(args.out_dir / "sabc_v2_gate_report.md", paired, tests)
    print(f"wrote {args.out_dir / 'sabc_v2_runs.csv'} ({len(runs)} runs)")
    print(f"matched pairs: {len(paired)}")
    print(tests.to_string(index=False))


if __name__ == "__main__":
    main()
