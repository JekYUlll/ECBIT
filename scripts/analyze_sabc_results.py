#!/usr/bin/env python3
"""Aggregate SABC exploration results and evaluate decision gates."""

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
    if "best" in result:
        metrics = dict(result["best"])
        metrics["best_epoch"] = metrics.pop("epoch", None)
        return metrics
    return dict(result)


def row_from_result(result_path: Path, config_dir: Path) -> dict[str, Any]:
    run_name = result_path.parent.name
    config_path = config_dir / f"{run_name}.yaml"
    result = load_json(result_path)
    config = load_yaml(config_path) if config_path.exists() else result.get("config", {})
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    missing_cfg = config.get("missing", {})
    test_station_ids = data_cfg.get("test_station_ids") or [""]
    metrics = flatten_result(result)

    row = {
        "run_name": run_name,
        "model": model_cfg.get("name", ""),
        "variant": model_cfg.get("variant", ""),
        "station_id": str(test_station_ids[0]),
        "pattern": missing_cfg.get("pattern", ""),
        "rate": missing_cfg.get("rate", ""),
        "seed": config.get("seed", ""),
        "result_path": str(result_path),
    }
    for key in METRIC_KEYS:
        row[key] = metrics.get(key)
    if "best_epoch" in metrics:
        row["best_epoch"] = metrics["best_epoch"]
    if "val_mae_mean" in metrics:
        row["val_mae_mean"] = metrics["val_mae_mean"]
    return row


def paired_frame(runs: pd.DataFrame) -> pd.DataFrame:
    idx_cols = ["station_id", "pattern", "rate", "seed"]
    baseline = runs[runs["variant"].eq("gated_baseline")][idx_cols + ["mae_mean"]].rename(
        columns={"mae_mean": "baseline_mae"}
    )
    sabc = runs[runs["variant"].eq("sabc_metadata")][idx_cols + ["mae_mean"]].rename(
        columns={"mae_mean": "sabc_mae"}
    )
    paired = sabc.merge(baseline, on=idx_cols, how="inner")
    paired["improvement"] = paired["baseline_mae"] - paired["sabc_mae"]
    return paired


def paired_test(values: np.ndarray) -> dict[str, float | int]:
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        return {"n": 0, "mean_improvement": np.nan, "ci95_low": np.nan, "ci95_high": np.nan, "t": np.nan, "p": np.nan}
    mean = float(np.mean(values))
    if n < 2:
        return {"n": n, "mean_improvement": mean, "ci95_low": np.nan, "ci95_high": np.nan, "t": np.nan, "p": np.nan}
    sd = float(np.std(values, ddof=1))
    se = sd / np.sqrt(n)
    if se == 0.0:
        t_stat = np.inf if mean > 0 else -np.inf if mean < 0 else 0.0
        p_value = 0.0 if mean != 0 else 1.0
        ci_low = ci_high = mean
    else:
        t_stat = float(mean / se)
        p_value = float(stats.ttest_1samp(values, 0.0).pvalue)
        ci_low, ci_high = stats.t.interval(0.95, df=n - 1, loc=mean, scale=se)
    return {
        "n": n,
        "mean_improvement": mean,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "t": t_stat,
        "p": p_value,
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


def gate_report(paired: pd.DataFrame, tests: pd.DataFrame) -> str:
    if paired.empty:
        return "# SABC Gate Report\n\nNo matched SABC/baseline result pairs are available yet.\n"

    overall = float(paired["improvement"].mean())
    by_station = paired.groupby("station_id")["improvement"].mean().sort_index()
    mount_sidley = float(by_station.get("mount_sidley", np.nan))
    zhongshan = float(by_station.get("zhongshan", np.nan))
    nico = float(by_station.get("nico", np.nan))
    hard_candidates = [value for value in [mount_sidley, zhongshan] if np.isfinite(value)]
    hard_station_gain = max(hard_candidates) if hard_candidates else np.nan

    overall_pass = overall >= 0.015
    hard_station_pass = np.isfinite(hard_station_gain) and hard_station_gain >= 0.030
    nico_pass = (not np.isfinite(nico)) or nico >= -0.010
    gate = overall_pass and hard_station_pass and nico_pass

    lines = [
        "# SABC Gate Report",
        "",
        f"Matched pairs: {len(paired)}",
        f"Overall mean improvement (gated baseline - SABC): {overall:.4f} normalized MAE",
        f"Mount Sidley mean improvement: {mount_sidley:.4f}",
        f"Zhongshan mean improvement: {zhongshan:.4f}",
        f"Nico mean improvement: {nico:.4f}",
        "",
        "| Gate | Threshold | Value | Pass |",
        "|---|---:|---:|:---:|",
        f"| Held-out mean improvement | >= 0.015 | {overall:.4f} | {'yes' if overall_pass else 'no'} |",
        f"| Mount Sidley or Zhongshan improvement | >= 0.030 | {hard_station_gain:.4f} | {'yes' if hard_station_pass else 'no'} |",
        f"| Nico degradation guard | >= -0.010 | {nico:.4f} | {'yes' if nico_pass else 'no'} |",
        "",
        f"Decision: {'pass' if gate else 'do not pass yet'}",
        "",
        "## Paired Tests",
        "",
        markdown_table(tests, ["group", "n", "mean_improvement", "ci95_low", "ci95_high", "t", "p"]),
        "",
        "## Station Mean Improvements",
        "",
        markdown_table(by_station.reset_index(name="mean_improvement"), ["station_id", "mean_improvement"]),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, default=Path("experiments/results/metrics/exploration_sabc"))
    parser.add_argument("--config-dir", type=Path, default=Path("experiments/configs/exploration_sabc"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/results/exploration_sabc"))
    args = parser.parse_args()

    result_paths = sorted(args.metrics_dir.glob("*/result.json"))
    if not result_paths:
        raise SystemExit(f"No result.json files found under {args.metrics_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(row_from_result(path, args.config_dir) for path in result_paths)
    runs = runs.sort_values(["station_id", "pattern", "seed", "variant"])
    summary = (
        runs.groupby(["variant", "station_id", "pattern"], dropna=False)[METRIC_KEYS]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in col if part) if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]
    paired = paired_frame(runs)
    tests = paired_tests(paired)

    runs.to_csv(args.out_dir / "sabc_runs.csv", index=False)
    summary.to_csv(args.out_dir / "sabc_summary.csv", index=False)
    paired.to_csv(args.out_dir / "sabc_paired_runs.csv", index=False)
    tests.to_csv(args.out_dir / "sabc_paired_tests.csv", index=False)
    (args.out_dir / "sabc_gate_report.md").write_text(gate_report(paired, tests), encoding="utf-8")

    print(f"wrote {args.out_dir / 'sabc_runs.csv'} ({len(runs)} runs)")
    print(f"matched pairs: {len(paired)}")
    if not tests.empty:
        print(tests.to_string(index=False))


if __name__ == "__main__":
    main()
