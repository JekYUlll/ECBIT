#!/usr/bin/env python3
"""Evaluate ERA5 variable importance and temporal-resolution robustness."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.impute_dataset import ImputationWindowDataset
from src.metrics import VARIABLES, masked_mae_rmse
from src.train_impute import artificial_mask_batch, build_model, station_ids_for_split
from src.utils.block_missing import apply_mask


def load_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_loader(config: dict[str, Any], split: str, batch_size: int, num_workers: int) -> DataLoader:
    data_cfg = config["data"]
    groups = data_cfg.get(f"{split}_station_groups", ["heldout"] if split == "test" else ["main"])
    dataset = ImputationWindowDataset(
        data_cfg.get("manifest_csv", "data/antaws_impute_manifest.csv"),
        station_groups=groups,
        window_splits=[split],
        station_ids=station_ids_for_split(data_cfg, split),
        window_subset_csv=data_cfg.get(f"{split}_window_subset_csv", data_cfg.get("window_subset_csv")),
        station_meta_csv=data_cfg.get("station_meta_csv"),
        residual_feature_csv=data_cfg.get("residual_feature_csv"),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def era5_train_mean(config: dict[str, Any], batch_size: int, num_workers: int) -> torch.Tensor:
    loader = make_loader(config, "train", batch_size=batch_size, num_workers=num_workers)
    total = None
    count = 0
    for batch in loader:
        era5 = batch["era5"]
        batch_sum = era5.sum(dim=(0, 1))
        total = batch_sum if total is None else total + batch_sum
        count += era5.shape[0] * era5.shape[1]
    if total is None or count == 0:
        raise RuntimeError("Could not compute ERA5 train mean")
    return total / count


def mask_era5_variable(era5: torch.Tensor, var_idx: int, train_mean: torch.Tensor) -> torch.Tensor:
    out = era5.clone()
    out[:, :, var_idx] = train_mean[var_idx].to(device=era5.device, dtype=era5.dtype)
    return out


def temporal_downsample_era5(era5: torch.Tensor, factor: int) -> torch.Tensor:
    """Downsample along time by `factor` and linearly interpolate back."""
    if factor <= 1:
        return era5
    bsz, seq_len, n_vars = era5.shape
    coarse = era5[:, ::factor, :]
    if coarse.shape[1] == 1:
        return coarse.expand(bsz, seq_len, n_vars)
    coarse_ct = coarse.transpose(1, 2)
    restored = F.interpolate(coarse_ct, size=seq_len, mode="linear", align_corners=True)
    return restored.transpose(1, 2)


def make_perturbations(era5: torch.Tensor, train_mean: torch.Tensor) -> dict[str, torch.Tensor]:
    perturbed = {"baseline": era5}
    for i, name in enumerate(VARIABLES[: era5.shape[-1]]):
        perturbed[f"mask_{name}"] = mask_era5_variable(era5, i, train_mean)
    for factor in (2, 4, 8):
        # The preprocessed ECBIT tensors are already aligned to 3-hourly AntAWS
        # windows, so these correspond to 6h, 12h, and 24h ERA5 inputs.
        perturbed[f"downsample_{factor * 3}h"] = temporal_downsample_era5(era5, factor)
    return perturbed


@torch.no_grad()
def evaluate_config(
    config_path: Path,
    checkpoint_path: Path,
    train_mean: torch.Tensor,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    config = load_config(config_path)
    config["device"] = str(device)
    model = build_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    loader = make_loader(config, "test", batch_size=batch_size, num_workers=num_workers)
    missing_cfg = config.get("missing", {})
    seed = int(config.get("seed", 42))
    preds: dict[str, list[torch.Tensor]] = defaultdict(list)
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []

    for step, batch in enumerate(loader):
        x = batch["x"].to(device)
        obs_mask = batch["obs_mask"].to(device)
        era5 = batch["era5"].to(device)
        time_enc = batch["time_enc"].to(device)
        artificial = artificial_mask_batch(obs_mask, missing_cfg, seed + 9_000_000_000 + step * 100_000).to(device)
        model_missing = torch.clamp((1.0 - obs_mask) + artificial, 0.0, 1.0)
        x_obs = apply_mask(x, model_missing)

        for perturbation, era5_variant in make_perturbations(era5, train_mean.to(device)).items():
            if getattr(model, "uses_station_features", False):
                if getattr(model, "uses_residual_features", False):
                    pred = model(
                        x_obs,
                        model_missing,
                        era5_variant,
                        time_enc,
                        batch["station_features"].to(device),
                        batch["residual_features"].to(device),
                    )
                else:
                    pred = model(x_obs, model_missing, era5_variant, time_enc, batch["station_features"].to(device))
            else:
                pred = model(x_obs, model_missing, era5_variant, time_enc)
            preds[perturbation].append(pred.cpu())
        targets.append(x.cpu())
        masks.append(artificial.cpu())

    target = torch.cat(targets)
    mask = torch.cat(masks)
    rows = []
    for perturbation, parts in sorted(preds.items()):
        metrics = masked_mae_rmse(torch.cat(parts), target, mask)
        row = {
            "run_name": config.get("run_name", config_path.stem),
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
            "pattern": missing_cfg.get("pattern"),
            "rate": missing_cfg.get("rate"),
            "seed": config.get("seed"),
            "perturbation": perturbation,
            **metrics,
        }
        rows.append(row)
    return rows


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    summary = rows.groupby("perturbation")["mae_mean"].agg(["count", "mean", "std"]).reset_index()
    baseline = float(summary.loc[summary["perturbation"].eq("baseline"), "mean"].iloc[0])
    summary["delta_mae"] = summary["mean"] - baseline
    return summary.sort_values("delta_mae", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=Path("experiments/configs/round2_gated"))
    parser.add_argument("--metrics-dir", type=Path, default=Path("experiments/results/metrics/round2_gated"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results/analysis/era5_robustness"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-runs", type=int, help="Optional smoke-test limit")
    args = parser.parse_args()

    config_paths = sorted(args.config_dir.glob("ecbit_full_*.yaml"))
    if args.max_runs:
        config_paths = config_paths[: args.max_runs]
    if not config_paths:
        raise RuntimeError(f"No ecbit_full configs found under {args.config_dir}")

    first_config = load_config(config_paths[0])
    train_mean = era5_train_mean(first_config, batch_size=args.batch_size, num_workers=args.num_workers)
    device = torch.device(args.device)
    all_rows = []
    for config_path in config_paths:
        config = load_config(config_path)
        run_name = config.get("run_name", config_path.stem)
        checkpoint = args.metrics_dir / run_name / "best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint for {run_name}: {checkpoint}")
        print(f"Evaluating {run_name}", flush=True)
        all_rows.extend(
            evaluate_config(
                config_path=config_path,
                checkpoint_path=checkpoint,
                train_mean=train_mean,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(all_rows)
    summary = summarize(runs)
    runs_csv = args.output_dir / "era5_robustness_runs.csv"
    summary_csv = args.output_dir / "era5_robustness_summary.csv"
    runs.to_csv(runs_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    with open(args.output_dir / "era5_robustness_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(orient="records"), f, indent=2)
    print(f"Wrote {runs_csv}")
    print(f"Wrote {summary_csv}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
