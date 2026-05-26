#!/usr/bin/env python3
"""Evaluate ERA5-direct calibration choices under the Round 1 masks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate_impute import make_loader
from src.metrics import masked_mae_rmse
from src.train_impute import artificial_mask_batch
from src.utils.block_missing import apply_mask


CONFIG_DIR = Path("experiments/configs/round1")
OUT_DIR = Path("experiments/results/analysis/era5_direct_calibration")
OUT_TEX = Path("paper/tables/tab_era5_direct_calibration.tex")
MODES = ("none", "mean_bias", "linear", "station_month_bias")
MODE_LABELS = {
    "none": "None",
    "mean_bias": "Global mean bias",
    "linear": "Global linear",
    "station_month_bias": "Station-month mean bias",
}


def load_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def collect_train(config: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    loader = make_loader(config, "train", force_num_workers=0)
    xs, es, ms, months, stations = [], [], [], [], []
    for batch in loader:
        xs.append(batch["x"])
        es.append(batch["era5"])
        ms.append(batch["obs_mask"])
        months.append(batch["month"])
        stations.extend(list(batch["station_id"]))
    return torch.cat(xs), torch.cat(es), torch.cat(ms).bool(), torch.cat(months), stations


def fit_global_calibration(x: torch.Tensor, era5: torch.Tensor, mask: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    n_vars = x.shape[-1]
    slope = torch.ones(n_vars, dtype=torch.float32)
    intercept = torch.zeros(n_vars, dtype=torch.float32)
    if mode == "none":
        return slope, intercept
    for c in range(n_vars):
        valid = mask[:, :, c]
        if not valid.any():
            continue
        xv = x[:, :, c][valid]
        ev = era5[:, :, c][valid]
        if mode == "mean_bias":
            intercept[c] = (xv - ev).mean()
        elif mode == "linear":
            e_mean = ev.mean()
            x_mean = xv.mean()
            denom = ((ev - e_mean) ** 2).sum()
            if float(denom) > 1.0e-8:
                slope[c] = ((ev - e_mean) * (xv - x_mean)).sum() / denom
                intercept[c] = x_mean - slope[c] * e_mean
        else:
            raise ValueError(f"Unknown global calibration mode: {mode}")
    return slope, intercept


def fit_station_month_bias(
    x: torch.Tensor,
    era5: torch.Tensor,
    mask: torch.Tensor,
    months: torch.Tensor,
    stations: list[str],
) -> dict[str, torch.Tensor | dict[tuple[str, int, int], float] | dict[tuple[str, int], float] | dict[int, float]]:
    n_vars = x.shape[-1]
    station_codes, station_names = pd.factorize(pd.Index(stations))
    station_arr = torch.tensor(station_codes, dtype=torch.long)
    station_names = list(station_names)
    global_bias: dict[int, float] = {}
    station_bias: dict[tuple[str, int], float] = {}
    station_month_bias: dict[tuple[str, int, int], float] = {}
    min_samples = 24

    for c in range(n_vars):
        valid = mask[:, :, c]
        global_bias[c] = float((x[:, :, c][valid] - era5[:, :, c][valid]).mean()) if valid.any() else 0.0
        for station_idx, station in enumerate(station_names):
            station_valid = valid & station_arr[:, None].eq(station_idx)
            if station_valid.any():
                station_bias[(str(station), c)] = float((x[:, :, c][station_valid] - era5[:, :, c][station_valid]).mean())
            for month in range(1, 13):
                month_valid = station_valid & months.eq(month)
                if int(month_valid.sum()) >= min_samples:
                    station_month_bias[(str(station), month, c)] = float((x[:, :, c][month_valid] - era5[:, :, c][month_valid]).mean())

    dense = torch.zeros((len(station_names), 13, n_vars), dtype=torch.float32)
    for station_idx, station in enumerate(station_names):
        for month in range(1, 13):
            for c in range(n_vars):
                bias = station_month_bias.get((str(station), month, c))
                if bias is None:
                    bias = station_bias.get((str(station), c), global_bias.get(c, 0.0))
                dense[station_idx, month, c] = float(bias)

    return {
        "global_bias": global_bias,
        "station_bias": station_bias,
        "station_month_bias": station_month_bias,
        "station_to_idx": {str(station): i for i, station in enumerate(station_names)},
        "dense_bias": dense,
    }


def station_month_correct(
    era5: torch.Tensor,
    months: torch.Tensor,
    stations: list[str],
    calibration: dict[str, Any],
) -> torch.Tensor:
    dense = calibration["dense_bias"].to(device=era5.device, dtype=era5.dtype)
    station_to_idx = calibration["station_to_idx"]
    station_idx = torch.tensor([station_to_idx[station] for station in stations], dtype=torch.long, device=era5.device)
    month_idx = months.to(device=era5.device, dtype=torch.long).clamp(1, 12)
    bias = dense[station_idx[:, None], month_idx, :]
    return era5 + bias


def evaluate_config(
    config: dict[str, Any],
    config_path: Path,
    mode: str,
    calibration: dict[str, Any] | None,
    slope: torch.Tensor | None,
    intercept: torch.Tensor | None,
) -> dict[str, float | str]:
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
        if mode == "station_month_bias":
            calibrated = station_month_correct(batch["era5"], batch["month"], list(batch["station_id"]), calibration)
        else:
            if slope is None or intercept is None:
                raise ValueError(f"Missing global calibration tensors for {mode}")
            calibrated = batch["era5"] * slope.view(1, 1, -1) + intercept.view(1, 1, -1)
        pred = torch.where(model_obs.bool(), x_obs, calibrated)
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


def write_latex_table(summary: pd.DataFrame, pattern_summary: pd.DataFrame, out_tex: Path) -> None:
    best = float(summary["mae_mean"].min())
    pattern_wide = pattern_summary.pivot(index="mode", columns="pattern", values="mae_mean")
    lines = [
        "\\begin{table*}[t]",
        "  \\centering",
        "  \\caption{ERA5 direct calibration ablation. All calibration parameters are estimated from training windows only and evaluated on the same 27 block-missing Round 1 configurations. The station-month variant estimates station-, month-, and variable-specific mean AWS-minus-ERA5 bias, with fallback to station-variable and then global variable bias when training samples are insufficient.}",
        "  \\label{tab:era5-direct-calibration}",
        "  \\begin{tabular}{lccccc}",
        "    \\toprule",
        "    Calibration & Overall MAE & RMSE & Short MAE & Medium MAE & Long MAE \\\\",
        "    \\midrule",
    ]
    for row in summary.itertuples(index=False):
        label = MODE_LABELS.get(row.mode, row.mode)
        mae = f"{row.mae_mean:.3f} $\\pm$ {row.mae_std:.3f}"
        rmse = f"{row.rmse_mean:.3f} $\\pm$ {row.rmse_std:.3f}"
        if abs(float(row.mae_mean) - best) < 1e-12:
            mae = f"\\textbf{{{mae}}}"
            rmse = f"\\textbf{{{rmse}}}"
        patterns = pattern_wide.loc[row.mode]
        lines.append(
            f"    {label} & {mae} & {rmse} & "
            f"{patterns['short']:.3f} & {patterns['medium']:.3f} & {patterns['long']:.3f} \\\\"
        )
    lines.extend(["    \\bottomrule", "  \\end{tabular}", "\\end{table*}", ""])
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--revision-dir", type=Path, default=Path("experiments/results/revision"))
    parser.add_argument("--out-tex", type=Path, default=OUT_TEX)
    args = parser.parse_args()

    configs = sorted(args.config_dir.glob("era5_direct_*.yaml"))
    if not configs:
        raise FileNotFoundError(f"No ERA5-direct configs found in {args.config_dir}")

    rows: list[dict[str, float | str]] = []
    for config_path in configs:
        config = load_config(config_path)
        train_x, train_era5, train_mask, train_months, train_stations = collect_train(config)
        fitted: dict[str, tuple[dict[str, Any] | None, torch.Tensor | None, torch.Tensor | None]] = {}
        for mode in MODES:
            if mode == "station_month_bias":
                fitted[mode] = (fit_station_month_bias(train_x, train_era5, train_mask, train_months, train_stations), None, None)
            else:
                slope, intercept = fit_global_calibration(train_x, train_era5, train_mask, mode)
                fitted[mode] = (None, slope, intercept)
        for mode in MODES:
            print(f"{config_path.name}: {mode}", flush=True)
            calibration, slope, intercept = fitted[mode]
            rows.append(evaluate_config(config, config_path, mode, calibration, slope, intercept))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(rows)
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
        .agg(
            mae_mean=("mae_mean", "mean"),
            rmse_mean=("rmse_mean", "mean"),
            n=("mae_mean", "count"),
        )
    )
    runs.to_csv(args.out_dir / "era5_direct_calibration_runs.csv", index=False)
    summary.to_csv(args.out_dir / "era5_direct_calibration_summary.csv", index=False)
    pattern_summary.to_csv(args.out_dir / "era5_direct_calibration_pattern_summary.csv", index=False)
    with open(args.out_dir / "era5_direct_calibration_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(orient="records"), f, indent=2)
    args.revision_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.revision_dir / "era5_calibration_comparison.csv", index=False)
    write_latex_table(summary, pattern_summary, args.out_tex)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
