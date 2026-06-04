#!/usr/bin/env python3
"""Evaluate thermodynamic q consistency for trained imputation checkpoints."""

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
from src.physical import specific_humidity_gkg, unnormalize
from src.train_impute import artificial_mask_batch, build_model
from src.utils.block_missing import apply_mask


DEFAULT_CONFIG_GLOB = "experiments/configs/exploration_pcr/*.yaml"
DEFAULT_OUT_CSV = Path("experiments/results/exploration_pcr/pcr_physical_consistency.csv")


def load_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate_one(config_path: Path, split: str, device: torch.device, max_batches: int | None) -> dict[str, Any]:
    config = load_config(config_path)
    checkpoint = Path(config["output_dir"]) / "best.pt"
    base = {
        "run_name": config.get("run_name", config_path.stem),
        "variant": config.get("model", {}).get("variant", ""),
        "pattern": config.get("missing", {}).get("target_pattern", config.get("missing", {}).get("pattern", "")),
        "rate": float(config.get("missing", {}).get("rate", float("nan"))),
        "seed": int(config.get("seed", -1)),
        "status": "ok",
    }
    if not checkpoint.exists():
        return {**base, "status": "missing_checkpoint"}

    cfg = dict(config)
    cfg["device"] = str(device)
    model = build_model(cfg).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    loader = make_loader(cfg, split, force_num_workers=0)
    missing_cfg = cfg.get("missing", {})
    seed = int(cfg.get("seed", 42))

    norm_sum = 0.0
    raw_sum = 0.0
    n_total = 0
    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        x = batch["x"].to(device)
        obs_mask = batch["obs_mask"].to(device)
        era5 = batch["era5"].to(device)
        time_enc = batch["time_enc"].to(device)
        artificial = artificial_mask_batch(obs_mask, missing_cfg, seed + step * 100_000).to(device)
        model_missing = torch.clamp((1.0 - obs_mask) + artificial, 0.0, 1.0)
        x_obs = apply_mask(x, model_missing)
        if getattr(model, "uses_station_features", False):
            pred = model(x_obs, model_missing, era5, time_enc, batch["station_features"].to(device))
        elif cfg["model"]["name"] in {"ecbit", "itransformer_era5"}:
            pred = model(x_obs, model_missing, era5, time_enc)
        else:
            pred = model(x_obs, model_missing, time_enc)

        raw = unnormalize(pred, batch["norm_mean"].to(device), batch["norm_std"].to(device))
        q_expected = specific_humidity_gkg(raw[:, :, 0], raw[:, :, 1], raw[:, :, 3])
        q_pred = raw[:, :, 4]
        raw_resid = (q_pred - q_expected).abs()

        q_std = batch["norm_std"].to(device)
        if q_std.ndim == 2:
            q_std = q_std[:, 4].unsqueeze(1)
        else:
            q_std = q_std[4]
        norm_resid = raw_resid / torch.clamp(q_std, min=1.0e-6)

        thermo_mask = artificial[:, :, [0, 1, 3, 4]].sum(dim=-1) > 0
        if thermo_mask.any():
            norm_sum += float(norm_resid[thermo_mask].sum().item())
            raw_sum += float(raw_resid[thermo_mask].sum().item())
            n_total += int(thermo_mask.sum().item())

    if n_total == 0:
        return {**base, "status": "empty_mask"}
    return {
        **base,
        "n": n_total,
        "q_consistency_mae_norm": norm_sum / n_total,
        "q_consistency_mae_gkg": raw_sum / n_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-glob", default=DEFAULT_CONFIG_GLOB)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args()

    rows = [evaluate_one(path, args.split, torch.device(args.device), args.max_batches) for path in sorted(Path().glob(args.config_glob))]
    out = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv} ({len(out)} rows)")
    ok = out[out["status"].eq("ok")]
    if not ok.empty:
        print(ok.groupby("variant")[["q_consistency_mae_norm", "q_consistency_mae_gkg"]].mean().round(6))


if __name__ == "__main__":
    main()
