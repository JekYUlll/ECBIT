#!/usr/bin/env python3
"""Evaluate station-month ERA5 plus endpoint anchoring hybrid baselines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_era5_direct_calibration import (
    collect_train,
    fit_station_month_bias,
    load_config,
    station_month_correct,
)
from src.baselines.era5_hybrid import endpoint_residual_anchor
from src.evaluate_impute import make_loader
from src.metrics import masked_mae_rmse
from src.train_impute import artificial_mask_batch
from src.utils.block_missing import apply_mask


CONFIG_DIR = Path("experiments/configs/round1")
OUT_DIR = Path("experiments/results/exploration_hybrid")
ROUND2_RUNS = Path("experiments/results/tables/round2_gated_final_runs.csv")
FAIR_RUNS = Path("experiments/results/tables/fair_era5_baselines_runs.csv")
MODES = ("station_month_bias", "endpoint_anchor")
MODE_LABELS = {
    "station_month_bias": "Station-month ERA5",
    "endpoint_anchor": "Station-month ERA5 + endpoint anchoring",
}


def evaluate_config(config: dict[str, Any], config_path: Path, mode: str, calibration: dict[str, Any]) -> dict[str, float | str]:
    loader = make_loader(config, "test", force_num_workers=0)
    missing_cfg = config.get("missing", {})
    seed = int(config.get("seed", 42))
    preds, targets, masks = [], [], []
    for step, batch in enumerate(loader):
        x = batch["x"]
        obs_mask = batch["obs_mask"]
        artificial = artificial_mask_batch(obs_mask, missing_cfg, seed + step * 100_000)
        model_obs = torch.clamp(obs_mask - artificial, 0.0, 1.0)
        x_obs = apply_mask(x, 1.0 - model_obs)
        calibrated = station_month_correct(batch["era5"], batch["month"], list(batch["station_id"]), calibration)
        if mode == "station_month_bias":
            pred = torch.where(model_obs.bool(), x_obs, calibrated)
        elif mode == "endpoint_anchor":
            pred = endpoint_residual_anchor(x_obs, calibrated, model_obs)
        else:
            raise ValueError(f"Unknown hybrid mode: {mode}")
        preds.append(pred)
        targets.append(x)
        masks.append(artificial)

    metrics = masked_mae_rmse(torch.cat(preds), torch.cat(targets), torch.cat(masks))
    return {
        "config": str(config_path),
        "run_name": str(config.get("run_name", config_path.stem)),
        "pattern": str(missing_cfg.get("pattern", "")),
        "rate": float(missing_cfg.get("rate", float("nan"))),
        "seed": int(seed),
        "mode": mode,
        **metrics,
    }


def paired_stats(values: pd.Series) -> dict[str, float | int]:
    diffs = values.to_numpy(dtype=float)
    n = int(len(diffs))
    mean = float(diffs.mean()) if n else float("nan")
    if n > 1:
        sem = float(diffs.std(ddof=1) / (n ** 0.5))
        ci = float(stats.t.ppf(0.975, n - 1) * sem)
        t_stat, p_value = stats.ttest_1samp(diffs, 0.0)
    else:
        ci = float("nan")
        t_stat, p_value = float("nan"), float("nan")
    return {
        "n": n,
        "mean_diff": mean,
        "ci95_low": mean - ci,
        "ci95_high": mean + ci,
        "t": float(t_stat),
        "p": float(p_value),
    }


def summarize_runs(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        runs.groupby("mode", as_index=False)
        .agg(
            mae_mean=("mae_mean", "mean"),
            mae_std=("mae_mean", "std"),
            rmse_mean=("rmse_mean", "mean"),
            rmse_std=("rmse_mean", "std"),
            n=("mae_mean", "count"),
        )
        .sort_values("mae_mean")
    )
    pattern_summary = (
        runs.groupby(["mode", "pattern"], as_index=False)
        .agg(mae_mean=("mae_mean", "mean"), rmse_mean=("rmse_mean", "mean"), n=("mae_mean", "count"))
    )
    return summary, pattern_summary


def mode_paired_tests(runs: pd.DataFrame) -> pd.DataFrame:
    idx = ["pattern", "rate", "seed"]
    base = runs[runs["mode"].eq("station_month_bias")][idx + ["mae_mean"]].rename(columns={"mae_mean": "station_month"})
    hybrid = runs[runs["mode"].eq("endpoint_anchor")][idx + ["mae_mean"]].rename(columns={"mae_mean": "endpoint_anchor"})
    paired = base.merge(hybrid, on=idx, how="inner")
    rows: list[dict[str, float | str | int]] = []
    for group, data in [("overall", paired)] + [(f"pattern:{p}", g) for p, g in paired.groupby("pattern")]:
        stats_row = paired_stats(data["station_month"] - data["endpoint_anchor"])
        rows.append({"comparison": "station_month_minus_endpoint_anchor", "group": group, **stats_row})
    return pd.DataFrame(rows)


def load_neural_runs(round2_runs: Path, fair_runs: Path) -> pd.DataFrame:
    frames = []
    if round2_runs.exists():
        round2 = pd.read_csv(round2_runs)
        round2 = round2[round2["pattern"].isin(["short", "medium", "long"])].copy()
        label_map = {"full": "ECBIT gated", "no_cross": "ECBIT concat"}
        round2 = round2[round2["variant"].isin(label_map)].copy()
        round2["method"] = round2["variant"].map(label_map)
        frames.append(round2[["method", "pattern", "rate", "seed", "mae_mean"]])
    if fair_runs.exists():
        fair = pd.read_csv(fair_runs)
        label_map = {"itransformer_era5": "iTransformer+ERA5", "saits_era5_concat": "SAITS+ERA5"}
        fair = fair[fair["model"].isin(label_map)].copy()
        fair["method"] = fair["model"].map(label_map)
        frames.append(fair[["method", "pattern", "rate", "seed", "mae_mean"]])
    if not frames:
        return pd.DataFrame(columns=["method", "pattern", "rate", "seed", "mae_mean"])
    return pd.concat(frames, ignore_index=True)


def neural_comparison(runs: pd.DataFrame, neural: pd.DataFrame) -> pd.DataFrame:
    idx = ["pattern", "rate", "seed"]
    hybrid = runs[runs["mode"].eq("endpoint_anchor")][idx + ["mae_mean"]].rename(columns={"mae_mean": "endpoint_anchor"})
    rows: list[dict[str, float | str | int]] = []
    for method, group in neural.groupby("method"):
        paired = group[idx + ["mae_mean"]].rename(columns={"mae_mean": "method_mae"}).merge(hybrid, on=idx, how="inner")
        stats_row = paired_stats(paired["method_mae"] - paired["endpoint_anchor"])
        rows.append(
            {
                "comparison": f"{method} minus endpoint_anchor",
                "method": method,
                "method_mae": float(paired["method_mae"].mean()),
                "endpoint_anchor_mae": float(paired["endpoint_anchor"].mean()),
                **stats_row,
            }
        )
    return pd.DataFrame(rows).sort_values("method_mae")


def markdown_table(frame: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Render a small dataframe as GitHub-style Markdown without tabulate."""
    if frame.empty:
        return "_No rows._"
    cols = list(frame.columns)
    rendered_rows: list[list[str]] = []
    for _, row in frame.iterrows():
        rendered = []
        for col in cols:
            value = row[col]
            if pd.isna(value):
                rendered.append("")
            elif isinstance(value, float):
                rendered.append(format(value, floatfmt))
            else:
                rendered.append(str(value))
        rendered_rows.append(rendered)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rendered_rows)
    return "\n".join(lines)


def write_report(summary: pd.DataFrame, pattern_summary: pd.DataFrame, mode_tests: pd.DataFrame, neural_tests: pd.DataFrame, out_path: Path) -> None:
    endpoint_mae = float(summary.loc[summary["mode"].eq("endpoint_anchor"), "mae_mean"].iloc[0])
    station_mae = float(summary.loc[summary["mode"].eq("station_month_bias"), "mae_mean"].iloc[0])
    best_neural = float(neural_tests["method_mae"].min()) if not neural_tests.empty else float("nan")
    gap_to_best = endpoint_mae - best_neural
    if pd.isna(gap_to_best):
        gate = "unknown"
    elif gap_to_best <= 0.005:
        gate = "competitive with the best neural ERA5 variants"
    elif gap_to_best >= 0.010:
        gate = "clearly behind the best neural ERA5 variants"
    else:
        gate = "close but not within the strict 0.005 competitiveness band"

    lines = [
        "# Operational Hybrid Baseline Report",
        "",
        f"Station-month ERA5 MAE: {station_mae:.4f}.",
        f"Endpoint-anchored station-month ERA5 MAE: {endpoint_mae:.4f}.",
        f"Gap from endpoint-anchored hybrid to best neural ERA5 method: {gap_to_best:.4f}.",
        f"Gate decision: {gate}.",
        "",
        "## Summary",
        "",
        markdown_table(summary),
        "",
        "## Pattern Summary",
        "",
        markdown_table(pattern_summary),
        "",
        "## Paired Test: Station-Month vs Endpoint Anchor",
        "",
        markdown_table(mode_tests),
        "",
        "## Paired Tests Against Neural ERA5 Variants",
        "",
        markdown_table(neural_tests) if not neural_tests.empty else "No neural comparison files were available.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--round2-runs", type=Path, default=ROUND2_RUNS)
    parser.add_argument("--fair-runs", type=Path, default=FAIR_RUNS)
    parser.add_argument("--report-only", action="store_true", help="Regenerate the Markdown report from existing CSV outputs.")
    args = parser.parse_args()

    if args.report_only:
        runs = pd.read_csv(args.out_dir / "operational_hybrid_runs.csv")
        summary = pd.read_csv(args.out_dir / "operational_hybrid_summary.csv")
        pattern_summary = pd.read_csv(args.out_dir / "operational_hybrid_pattern_summary.csv")
        mode_tests = pd.read_csv(args.out_dir / "operational_hybrid_mode_paired_tests.csv")
        neural_tests = neural_comparison(runs, load_neural_runs(args.round2_runs, args.fair_runs))
        neural_tests.to_csv(args.out_dir / "operational_hybrid_vs_neural.csv", index=False)
        write_report(summary, pattern_summary, mode_tests, neural_tests, args.out_dir / "operational_hybrid_report.md")
        print(summary.to_string(index=False))
        if not neural_tests.empty:
            print(neural_tests.to_string(index=False))
        return

    configs = sorted(args.config_dir.glob("era5_direct_*.yaml"))
    if not configs:
        raise FileNotFoundError(f"No ERA5-direct configs found in {args.config_dir}")

    rows: list[dict[str, float | str]] = []
    for config_path in configs:
        config = load_config(config_path)
        train_x, train_era5, train_mask, train_months, train_stations = collect_train(config)
        calibration = fit_station_month_bias(train_x, train_era5, train_mask, train_months, train_stations)
        for mode in MODES:
            print(f"{config_path.name}: {mode}", flush=True)
            rows.append(evaluate_config(config, config_path, mode, calibration))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(rows)
    summary, pattern_summary = summarize_runs(runs)
    mode_tests = mode_paired_tests(runs)
    neural_tests = neural_comparison(runs, load_neural_runs(args.round2_runs, args.fair_runs))

    runs.to_csv(args.out_dir / "operational_hybrid_runs.csv", index=False)
    summary.to_csv(args.out_dir / "operational_hybrid_summary.csv", index=False)
    pattern_summary.to_csv(args.out_dir / "operational_hybrid_pattern_summary.csv", index=False)
    mode_tests.to_csv(args.out_dir / "operational_hybrid_mode_paired_tests.csv", index=False)
    neural_tests.to_csv(args.out_dir / "operational_hybrid_vs_neural.csv", index=False)
    write_report(summary, pattern_summary, mode_tests, neural_tests, args.out_dir / "operational_hybrid_report.md")
    print(summary.to_string(index=False))
    if not neural_tests.empty:
        print(neural_tests.to_string(index=False))


if __name__ == "__main__":
    main()
