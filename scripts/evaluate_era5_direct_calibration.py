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
MODES = ("none", "mean_bias", "linear")


def load_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def collect_train(config: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loader = make_loader(config, "train", force_num_workers=0)
    xs, es, ms = [], [], []
    for batch in loader:
        xs.append(batch["x"])
        es.append(batch["era5"])
        ms.append(batch["obs_mask"])
    return torch.cat(xs), torch.cat(es), torch.cat(ms).bool()


def fit_calibration(x: torch.Tensor, era5: torch.Tensor, mask: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
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
            raise ValueError(f"Unknown calibration mode: {mode}")
    return slope, intercept


def evaluate_config(config: dict[str, Any], config_path: Path, mode: str) -> dict[str, float | str]:
    train_x, train_era5, train_mask = collect_train(config)
    slope, intercept = fit_calibration(train_x, train_era5, train_mask, mode)

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    configs = sorted(args.config_dir.glob("era5_direct_*.yaml"))
    if not configs:
        raise FileNotFoundError(f"No ERA5-direct configs found in {args.config_dir}")

    rows: list[dict[str, float | str]] = []
    for config_path in configs:
        config = load_config(config_path)
        for mode in MODES:
            print(f"{config_path.name}: {mode}", flush=True)
            rows.append(evaluate_config(config, config_path, mode))

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
    runs.to_csv(args.out_dir / "era5_direct_calibration_runs.csv", index=False)
    summary.to_csv(args.out_dir / "era5_direct_calibration_summary.csv", index=False)
    with open(args.out_dir / "era5_direct_calibration_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(orient="records"), f, indent=2)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
