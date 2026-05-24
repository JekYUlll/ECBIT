#!/usr/bin/env python3
"""Audit whether decoded outputs perturb observed input positions before copy-back."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate_impute import load_config, make_loader
from src.train_impute import artificial_mask_batch, build_model
from src.utils.block_missing import apply_mask


CONFIG_GLOB = "experiments/configs/round2_gated/ecbit_full_*.yaml"
OUT_DIR = Path("experiments/results/analysis/observed_consistency")


def checkpoint_for_config(config: dict[str, Any]) -> Path:
    return Path(config.get("output_dir", "experiments/results/metrics")) / "best.pt"


@torch.no_grad()
def missing_checkpoint_row(config_path: Path, config: dict[str, Any], checkpoint: Path) -> dict[str, float | str | int]:
    missing_cfg = config.get("missing", {})
    return {
        "config": str(config_path),
        "run_name": str(config.get("run_name", config_path.stem)),
        "variant": str(config.get("model", {}).get("variant", "")),
        "pattern": str(missing_cfg.get("target_pattern", missing_cfg.get("pattern", ""))),
        "rate": float(missing_cfg.get("rate", float("nan"))),
        "seed": int(config.get("seed", -1)),
        "checkpoint": str(checkpoint),
        "status": "missing_checkpoint",
        "visible_positions": 0,
        "pre_copy_mae": float("nan"),
        "pre_copy_rmse": float("nan"),
        "post_copy_mae": float("nan"),
    }


@torch.no_grad()
def audit_config(config_path: Path, device: torch.device, allow_missing: bool = False) -> dict[str, float | str | int]:
    config = load_config(config_path)
    model = build_model(config).to(device)
    checkpoint = checkpoint_for_config(config)
    if not checkpoint.exists():
        if allow_missing:
            return missing_checkpoint_row(config_path, config, checkpoint)
        raise FileNotFoundError(f"Checkpoint not found for {config_path}: {checkpoint}")
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    loader = make_loader(config, "test", force_num_workers=0)
    missing_cfg = config.get("missing", {})
    seed = int(config.get("seed", 42))
    pre_abs_sum = 0.0
    pre_sq_sum = 0.0
    post_abs_sum = 0.0
    count = 0.0
    for step, batch in enumerate(loader):
        x = batch["x"].to(device)
        obs_mask = batch["obs_mask"].to(device)
        era5 = batch["era5"].to(device)
        time_enc = batch["time_enc"].to(device)
        artificial = artificial_mask_batch(obs_mask, missing_cfg, seed + 9_000_000_000 + step * 100_000).to(device)
        model_missing = torch.clamp((1.0 - obs_mask) + artificial, 0.0, 1.0)
        visible = (1.0 - model_missing).bool()
        x_obs = apply_mask(x, model_missing)
        if config["model"]["name"] == "ecbit":
            pred = model(x_obs, model_missing, era5, time_enc)
        else:
            pred = model(x_obs, model_missing, time_enc)
        copied = torch.where(visible, x, pred)
        diff = pred[visible] - x[visible]
        post_diff = copied[visible] - x[visible]
        pre_abs_sum += float(diff.abs().sum().item())
        pre_sq_sum += float((diff**2).sum().item())
        post_abs_sum += float(post_diff.abs().sum().item())
        count += float(diff.numel())

    return {
        "config": str(config_path),
        "run_name": str(config.get("run_name", config_path.stem)),
        "variant": str(config.get("model", {}).get("variant", "")),
        "pattern": str(config.get("missing", {}).get("pattern", "")),
        "rate": float(config.get("missing", {}).get("rate", float("nan"))),
        "seed": int(seed),
        "checkpoint": str(checkpoint),
        "status": "ok",
        "visible_positions": int(count),
        "pre_copy_mae": pre_abs_sum / max(count, 1.0),
        "pre_copy_rmse": (pre_sq_sum / max(count, 1.0)) ** 0.5,
        "post_copy_mae": post_abs_sum / max(count, 1.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-glob", default=CONFIG_GLOB)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--allow-missing", action="store_true", help="Record missing checkpoints instead of failing.")
    args = parser.parse_args()

    configs = sorted(Path().glob(args.config_glob))
    if not configs:
        raise FileNotFoundError(f"No configs matched {args.config_glob}")
    device = torch.device(args.device)
    rows = [audit_config(path, device, allow_missing=args.allow_missing) for path in configs]
    runs = pd.DataFrame(rows)
    ok_runs = runs[runs["status"] == "ok"]
    missing_counts = runs[runs["status"] != "ok"].groupby("variant").size()
    if ok_runs.empty:
        summary = (
            missing_counts.rename("missing_checkpoints")
            .reset_index()
            .assign(pre_copy_mae=float("nan"), pre_copy_rmse=float("nan"), post_copy_mae=float("nan"), runs=0)
            [["variant", "pre_copy_mae", "pre_copy_rmse", "post_copy_mae", "runs", "missing_checkpoints"]]
        )
    else:
        summary = (
            ok_runs.groupby("variant", as_index=False)
            .agg(
                pre_copy_mae=("pre_copy_mae", "mean"),
                pre_copy_rmse=("pre_copy_rmse", "mean"),
                post_copy_mae=("post_copy_mae", "mean"),
                runs=("run_name", "count"),
            )
            .sort_values("variant")
        )
        summary["missing_checkpoints"] = summary["variant"].map(missing_counts).fillna(0).astype(int)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.out_dir / "observed_consistency_runs.csv", index=False)
    summary.to_csv(args.out_dir / "observed_consistency_summary.csv", index=False)
    with open(args.out_dir / "observed_consistency_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(orient="records"), f, indent=2)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
