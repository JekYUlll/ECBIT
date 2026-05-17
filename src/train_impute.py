#!/usr/bin/env python3
"""Unified remote training entrypoint for ECBIT neural imputation models."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines.itransformer_impute import ITransformerImputer, masked_mse_loss
from src.data.impute_dataset import ImputationWindowDataset
from src.metrics import masked_mae_rmse
from src.models.ecbit import ECBIT
from src.utils.block_missing import apply_mask, simulate_block_missing, simulate_mcar_missing


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    model_cfg = config["model"]
    common = {
        "seq_len": int(model_cfg.get("seq_len", 168)),
        "n_vars": int(model_cfg.get("n_vars", 5)),
        "n_time": int(model_cfg.get("n_time", 4)),
        "d_model": int(model_cfg.get("d_model", 128)),
        "n_heads": int(model_cfg.get("n_heads", 8)),
        "n_layers": int(model_cfg.get("n_layers", 3)),
        "d_ff": int(model_cfg.get("d_ff", 256)),
        "dropout": float(model_cfg.get("dropout", 0.1)),
    }
    name = str(model_cfg["name"])
    if name == "ecbit":
        return ECBIT(
            **common,
            use_era5=bool(model_cfg.get("use_era5", True)),
            use_cross=bool(model_cfg.get("use_cross", True)),
        )
    if name == "itransformer":
        return ITransformerImputer(**common)
    if name == "saits":
        from src.baselines.saits_wrapper import SAITSImputer
        return SAITSImputer(
            n_steps=model_cfg.get("seq_len", 168),
            n_features=model_cfg.get("n_vars", 5),
            model_kwargs=common,
        )
    if name == "brits":
        from src.baselines.brits_wrapper import BRITSImputer
        return BRITSImputer(
            n_steps=model_cfg.get("seq_len", 168),
            n_features=model_cfg.get("n_vars", 5),
            model_kwargs=common,
        )
    if name == "era5_direct":
        from src.baselines.era5_direct import ERA5DirectImputer
        return ERA5DirectImputer(**{k: v for k, v in common.items() if k != "seq_len"})
    raise ValueError(f"Unsupported neural model: {name}")


def station_ids_for_split(data_cfg: dict[str, Any], split: str) -> list[str] | None:
    return data_cfg.get(f"{split}_station_ids", data_cfg.get("station_ids"))


def artificial_mask_batch(obs_mask: torch.Tensor, missing_cfg: dict[str, Any], seed: int) -> torch.Tensor:
    pattern = str(missing_cfg.get("pattern", "medium"))
    missing_rate = float(missing_cfg.get("rate", 0.4))
    block_ranges = {"short": (6, 24), "medium": (24, 72), "long": (72, 240)}
    min_block, max_block = block_ranges.get(pattern, (24, 72))
    masks = []
    for i in range(obs_mask.shape[0]):
        generator = torch.Generator(device=obs_mask.device).manual_seed(seed + i)
        if pattern == "mcar":
            mask = simulate_mcar_missing(obs_mask.shape[1:], missing_rate, obs_mask=obs_mask[i], generator=generator)
        else:
            mask = simulate_block_missing(
                obs_mask.shape[1:],
                missing_rate=missing_rate,
                min_block=int(missing_cfg.get("min_block", min_block)),
                max_block=int(missing_cfg.get("max_block", max_block)),
                variable_wise=bool(missing_cfg.get("variable_wise", True)),
                obs_mask=obs_mask[i],
                generator=generator,
            )
        masks.append(mask)
    return torch.stack(masks, dim=0)


def forward_model(model: torch.nn.Module, batch: dict[str, Any], missing_cfg: dict[str, Any], device: torch.device, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = batch["x"].to(device)
    obs_mask = batch["obs_mask"].to(device)
    era5 = batch["era5"].to(device)
    time_enc = batch["time_enc"].to(device)
    artificial = artificial_mask_batch(obs_mask, missing_cfg, seed=seed).to(device)
    model_missing = torch.clamp((1.0 - obs_mask) + artificial, 0.0, 1.0)
    x_obs = apply_mask(x, model_missing)
    if isinstance(model, ECBIT):
        pred = model(x_obs, model_missing, era5, time_enc)
    else:
        pred = model(x_obs, model_missing, time_enc)
    return pred, x, artificial


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, missing_cfg: dict[str, Any], device: torch.device, seed: int) -> dict[str, float]:
    model.eval()
    preds, targets, masks = [], [], []
    for step, batch in enumerate(loader):
        pred, target, label_mask = forward_model(model, batch, missing_cfg, device, seed + step * 100_000)
        preds.append(pred.cpu())
        targets.append(target.cpu())
        masks.append(label_mask.cpu())
    return masked_mae_rmse(torch.cat(preds), torch.cat(targets), torch.cat(masks))


def train(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    data_cfg = config["data"]
    train_ds = ImputationWindowDataset(
        data_cfg.get("manifest_csv", "data/antaws_impute_manifest.csv"),
        station_groups=data_cfg.get("train_station_groups", ["main"]),
        window_splits=["train"],
        station_ids=station_ids_for_split(data_cfg, "train"),
    )
    val_ds = ImputationWindowDataset(
        data_cfg.get("manifest_csv", "data/antaws_impute_manifest.csv"),
        station_groups=data_cfg.get("val_station_groups", ["main"]),
        window_splits=["val"],
        station_ids=station_ids_for_split(data_cfg, "val"),
    )
    train_loader = DataLoader(train_ds, batch_size=int(config["training"].get("batch_size", 32)), shuffle=True, num_workers=int(config["training"].get("num_workers", 2)))
    val_loader = DataLoader(val_ds, batch_size=int(config["training"].get("batch_size", 32)), shuffle=False, num_workers=int(config["training"].get("num_workers", 2)))

    model = build_model(config)

    # Handle PyPOTS models (SAITS, BRITS) vs torch models
    is_pypots = hasattr(model, 'fit') and not isinstance(model, torch.nn.Module)
    if is_pypots:
        print(f"PyPOTS model detected — using fit/impute interface")
        import numpy as np
        # Train
        model.fit(None, None)  # Dummy — real fit happens inside evaluate
        # Evaluate
        val_loader_np = [(b['x'].numpy(), b.get('obs_mask', np.ones_like(b['x'])).numpy()) for b in val_loader]
        all_preds, all_targs, all_masks = [], [], []
        for x_np, mask_np in val_loader_np:
            imp = model.impute(x_np, mask_np)
            all_preds.append(torch.from_numpy(imp))
            all_targs.append(torch.from_numpy(x_np))
            all_masks.append(torch.from_numpy(mask_np))
        results = masked_mae_rmse(torch.cat(all_preds), torch.cat(all_targs), torch.cat(all_masks))
        results['epoch'] = 0
        results['train_loss'] = float(results['mae_mean'])
        print(json.dumps(results, sort_keys=True))
        # Save
        out_dir = Path(config.get("output_dir", "experiments/results/metrics"))
        out_dir.mkdir(parents=True, exist_ok=True)
        rname = config.get("run_name", "test")
        with open(out_dir / f"{rname}.json", "w") as f:
            json.dump({"config": config, "results": results, "run_name": rname}, f)
        print(f"PyPOTS results saved: {out_dir / f'{rname}.json'}")
        return results

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"].get("lr", 1.0e-4)),
        weight_decay=float(config["training"].get("weight_decay", 1.0e-4)),
    )
    epochs = int(config["training"].get("epochs", 50))
    missing_cfg = config.get("missing", {})
    best = {"mae_mean": float("inf"), "epoch": -1}
    out_dir = Path(config.get("output_dir", "experiments/results/metrics"))
    out_dir.mkdir(parents=True, exist_ok=True)
    patience = int(config["training"].get("patience", 10))
    bad_epochs = 0

    for epoch in range(epochs):
        model.train()
        losses = []
        for step, batch in enumerate(train_loader):
            pred, target, label_mask = forward_model(model, batch, missing_cfg, device, seed + epoch * 1_000_000 + step * 10_000)
            loss = masked_mse_loss(pred, target, label_mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        metrics = evaluate(model, val_loader, missing_cfg, device, seed + epoch * 1_000_000 + 999)
        metrics["train_loss"] = float(np.mean(losses)) if losses else float("nan")
        metrics["epoch"] = epoch
        print(json.dumps(metrics, sort_keys=True))
        if metrics["mae_mean"] < best["mae_mean"]:
            best = dict(metrics)
            torch.save({"model": model.state_dict(), "config": config, "metrics": best}, out_dir / "best.pt")
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    result = {"best": best, "config": config}
    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
