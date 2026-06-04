#!/usr/bin/env python3
"""Evaluate MCAR-trained ERA5 models on block-missing test masks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.impute_dataset import ImputationWindowDataset
from src.metrics import masked_mae_rmse
from src.train_impute import artificial_mask_batch, build_model, station_ids_for_split
from src.utils.block_missing import apply_mask


def load_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_loader(config: dict[str, Any], split: str, batch_size: int, num_workers: int) -> DataLoader:
    data_cfg = config["data"]
    dataset = ImputationWindowDataset(
        data_cfg.get("manifest_csv", "data/antaws_impute_manifest.csv"),
        station_groups=data_cfg.get(f"{split}_station_groups", ["main"]),
        window_splits=[split],
        station_ids=station_ids_for_split(data_cfg, split),
        window_subset_csv=data_cfg.get(f"{split}_window_subset_csv", data_cfg.get("window_subset_csv")),
        station_meta_csv=data_cfg.get("station_meta_csv"),
        residual_feature_csv=data_cfg.get("residual_feature_csv"),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


@torch.no_grad()
def evaluate_block(
    config_path: Path,
    checkpoint_path: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict[str, Any]:
    config = load_config(config_path)
    config["device"] = str(device)
    missing_cfg = dict(config.get("missing", {}))
    target_pattern = str(missing_cfg.get("target_pattern", "medium"))
    missing_cfg["pattern"] = target_pattern
    missing_cfg.setdefault("variable_wise", True)

    model = build_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    loader = make_loader(config, "test", batch_size=batch_size, num_workers=num_workers)
    seed = int(config.get("seed", 42))
    preds, targets, masks = [], [], []
    for step, batch in enumerate(loader):
        x = batch["x"].to(device)
        obs_mask = batch["obs_mask"].to(device)
        era5 = batch["era5"].to(device)
        time_enc = batch["time_enc"].to(device)
        artificial = artificial_mask_batch(obs_mask, missing_cfg, seed + 9_000_000_000 + step * 100_000).to(device)
        model_missing = torch.clamp((1.0 - obs_mask) + artificial, 0.0, 1.0)
        x_obs = apply_mask(x, model_missing)
        if getattr(model, "uses_station_features", False):
            if getattr(model, "uses_residual_features", False):
                pred = model(
                    x_obs,
                    model_missing,
                    era5,
                    time_enc,
                    batch["station_features"].to(device),
                    batch["residual_features"].to(device),
                )
            else:
                pred = model(x_obs, model_missing, era5, time_enc, batch["station_features"].to(device))
        else:
            pred = model(x_obs, model_missing, era5, time_enc)
        preds.append(pred.cpu())
        targets.append(x.cpu())
        masks.append(artificial.cpu())

    metrics = masked_mae_rmse(torch.cat(preds), torch.cat(targets), torch.cat(masks))
    return {
        "run_name": config.get("run_name", config_path.stem),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "train_pattern": config.get("missing", {}).get("pattern"),
        "test_pattern": target_pattern,
        "rate": missing_cfg.get("rate"),
        "seed": config.get("seed"),
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=Path("experiments/configs/round2_gated"))
    parser.add_argument("--config-glob", default="ecbit_no_blockmask_*.yaml")
    parser.add_argument("--metrics-dir", type=Path, default=Path("experiments/results/metrics/round2_gated"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results/analysis/mcar_on_block"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    rows = []
    device = torch.device(args.device)
    for config_path in sorted(args.config_dir.glob(args.config_glob)):
        config = load_config(config_path)
        run_name = config.get("run_name", config_path.stem)
        checkpoint = args.metrics_dir / run_name / "best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint for {run_name}: {checkpoint}")
        print(f"Evaluating {run_name} on {config['missing'].get('target_pattern')} block masks", flush=True)
        rows.append(evaluate_block(config_path, checkpoint, args.batch_size, args.num_workers, device))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(rows)
    summary = (
        runs.groupby(["train_pattern", "test_pattern"])["mae_mean"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .sort_values(["test_pattern", "train_pattern"])
    )
    runs_csv = args.output_dir / "mcar_on_block_runs.csv"
    summary_csv = args.output_dir / "mcar_on_block_summary.csv"
    runs.to_csv(runs_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    with open(args.output_dir / "mcar_on_block_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(orient="records"), f, indent=2)
    print(f"Wrote {runs_csv}")
    print(f"Wrote {summary_csv}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
