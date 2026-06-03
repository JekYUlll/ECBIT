#!/usr/bin/env python3
"""Evaluate trained neural models or stateless baselines on ECBIT windows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines.era5_direct import ERA5DirectImputer
from src.baselines.linear_interp import linear_interpolate
from src.baselines.locf import locf_impute
from src.baselines.brits_wrapper import BRITSImputer
from src.baselines.saits_wrapper import SAITSImputer
from src.data.impute_dataset import ImputationWindowDataset
from src.metrics import masked_mae_rmse
from src.train_impute import artificial_mask_batch, build_model, station_ids_for_split
from src.utils.block_missing import apply_mask


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_loader(config: dict[str, Any], split: str, force_num_workers: int | None = None) -> DataLoader:
    data_cfg = config["data"]
    groups_key = f"{split}_station_groups"
    groups = data_cfg.get(groups_key, ["heldout"] if split == "test" else ["main"])
    ds = ImputationWindowDataset(
        data_cfg.get("manifest_csv", "data/antaws_impute_manifest.csv"),
        station_groups=groups,
        window_splits=[split],
        station_ids=station_ids_for_split(data_cfg, split),
        window_subset_csv=data_cfg.get(f"{split}_window_subset_csv", data_cfg.get("window_subset_csv")),
        station_meta_csv=data_cfg.get("station_meta_csv"),
    )
    num_workers = int(config.get("eval", {}).get("num_workers", 2))
    if force_num_workers is not None:
        num_workers = force_num_workers
    return DataLoader(
        ds,
        batch_size=int(config.get("eval", {}).get("batch_size", 64)),
        shuffle=False,
        num_workers=num_workers,
    )


@torch.no_grad()
def evaluate_neural(config: dict[str, Any], checkpoint: Path, split: str) -> dict[str, float]:
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(config).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    # Stateless and PyPOTS baselines repeatedly materialize full train/test
    # arrays. Keep loading single-process to avoid file-descriptor exhaustion
    # when multiple remote workers run concurrently.
    loader = make_loader(config, split, force_num_workers=0)
    missing_cfg = config.get("missing", {})
    seed = int(config.get("seed", 42))
    preds, targets, masks = [], [], []
    for step, batch in enumerate(loader):
        x = batch["x"].to(device)
        obs_mask = batch["obs_mask"].to(device)
        era5 = batch["era5"].to(device)
        time_enc = batch["time_enc"].to(device)
        artificial = artificial_mask_batch(obs_mask, missing_cfg, seed + step * 100_000).to(device)
        model_missing = torch.clamp((1.0 - obs_mask) + artificial, 0.0, 1.0)
        x_obs = apply_mask(x, model_missing)
        if getattr(model, "uses_station_features", False):
            pred = model(x_obs, model_missing, era5, time_enc, batch["station_features"].to(device))
        elif config["model"]["name"] in {"ecbit", "itransformer_era5"}:
            pred = model(x_obs, model_missing, era5, time_enc)
        else:
            pred = model(x_obs, model_missing, time_enc)
        preds.append(pred.cpu())
        targets.append(x.cpu())
        masks.append(artificial.cpu())
    return masked_mae_rmse(torch.cat(preds), torch.cat(targets), torch.cat(masks))


def evaluate_stateless(config: dict[str, Any], split: str) -> dict[str, float]:
    # Stateless baselines materialize full batches on CPU and are commonly run
    # in parallel. Single-process loading avoids file descriptor exhaustion.
    loader = make_loader(config, split, force_num_workers=0)
    missing_cfg = config.get("missing", {})
    seed = int(config.get("seed", 42))
    name = config["model"]["name"]
    era5_imputer = ERA5DirectImputer() if name == "era5_direct" else None
    pypots_imputer = None
    if name in {"saits", "brits", "saits_era5_concat"}:
        model_cls = SAITSImputer if name == "saits" else BRITSImputer
        if name == "saits_era5_concat":
            model_cls = SAITSImputer
        model_cfg = config.get("model", {})
        n_features = int(model_cfg.get("n_features", model_cfg.get("n_vars", 5)))
        pypots_imputer = model_cls(
            n_steps=int(model_cfg.get("seq_len", 168)),
            n_features=n_features,
            **model_cfg.get("model_kwargs", {}),
        )
    if era5_imputer is not None or pypots_imputer is not None:
        train_loader = make_loader(config, "train", force_num_workers=0)
        xs, es, ms = [], [], []
        for batch in train_loader:
            xs.append(batch["x"])
            es.append(batch["era5"])
            ms.append(batch["obs_mask"])
        train_x = torch.cat(xs)
        train_mask = torch.cat(ms)
        if era5_imputer is not None:
            era5_imputer.fit(train_x, torch.cat(es), train_mask)
        if pypots_imputer is not None:
            if name == "saits_era5_concat":
                train_era5 = torch.cat(es)
                train_x_aug = torch.cat([train_x, train_era5], dim=-1)
                train_mask_aug = torch.cat([train_mask.float(), torch.ones_like(train_mask, dtype=torch.float32)], dim=-1)
                pypots_imputer.fit(train_x_aug.numpy(), train_mask_aug.numpy())
            else:
                pypots_imputer.fit(train_x.numpy(), train_mask.numpy())

    preds, targets, masks = [], [], []
    with torch.no_grad():
        for step, batch in enumerate(loader):
            x = batch["x"]
            obs_mask = batch["obs_mask"]
            artificial = artificial_mask_batch(obs_mask, missing_cfg, seed + step * 100_000)
            model_obs = torch.clamp(obs_mask - artificial, 0.0, 1.0)
            x_obs = apply_mask(x, 1.0 - model_obs)
            if name == "linear_interp":
                pred = linear_interpolate(x_obs, model_obs)
            elif name == "locf":
                pred = locf_impute(x_obs, model_obs)
            elif name == "era5_direct":
                assert era5_imputer is not None
                pred = era5_imputer.impute(x_obs, batch["era5"], model_obs)
            elif name in {"saits", "brits", "saits_era5_concat"}:
                assert pypots_imputer is not None
                if name == "saits_era5_concat":
                    x_aug = torch.cat([x_obs, batch["era5"]], dim=-1)
                    obs_aug = torch.cat([model_obs, torch.ones_like(model_obs)], dim=-1)
                    pred = torch.from_numpy(pypots_imputer.impute(x_aug.numpy(), obs_aug.numpy()))[:, :, : x.shape[-1]]
                else:
                    pred = torch.from_numpy(pypots_imputer.impute(x_obs.numpy(), model_obs.numpy()))
            else:
                raise ValueError(f"Unsupported stateless baseline: {name}")
            preds.append(pred.cpu())
            targets.append(x.cpu())
            masks.append(artificial.cpu())
    return masked_mae_rmse(torch.cat(preds), torch.cat(targets), torch.cat(masks))


def evaluate(config: dict[str, Any], split: str, checkpoint: Path | None = None) -> dict[str, float]:
    name = config["model"]["name"]
    if name in {"linear_interp", "locf", "era5_direct", "saits", "brits", "saits_era5_concat"}:
        return evaluate_stateless(config, split)
    if checkpoint is None:
        checkpoint = Path(config.get("checkpoint", "experiments/results/metrics/best.pt"))
    return evaluate_neural(config, checkpoint, split)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(load_config(args.config), args.split, args.checkpoint)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
