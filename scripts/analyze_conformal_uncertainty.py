#!/usr/bin/env python3
"""Split-conformal residual intervals for trained ECBIT-style imputers."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate_impute import make_loader
from src.metrics import VARIABLES
from src.train_impute import artificial_mask_batch, build_model
from src.utils.block_missing import apply_mask


DEFAULT_CONFIG_GLOB = "experiments/configs/fair_era5_baselines/itransformer_era5_*_r40_s*.yaml"
DEFAULT_OUT_DIR = Path("experiments/results/exploration_uncertainty")


def load_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def conformal_quantile(values: torch.Tensor, alpha: float) -> float:
    """Finite-sample split-conformal quantile for absolute residuals."""
    flat = values.detach().flatten()
    flat = flat[torch.isfinite(flat)]
    n = int(flat.numel())
    if n == 0:
        return float("nan")
    rank = min(math.ceil((n + 1) * (1.0 - alpha)), n)
    return float(torch.sort(flat).values[rank - 1].item())


@torch.no_grad()
def collect_abs_errors(
    config: dict[str, Any],
    checkpoint: Path,
    split: str,
    device: torch.device,
    max_batches: int | None = None,
) -> list[torch.Tensor]:
    cfg = dict(config)
    cfg["device"] = str(device)
    model = build_model(cfg).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    loader = make_loader(cfg, split, force_num_workers=0)
    missing_cfg = cfg.get("missing", {})
    seed = int(cfg.get("seed", 42))
    errors = [[] for _ in VARIABLES]
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
        abs_err = (pred - x).abs().detach().cpu()
        mask = artificial.detach().cpu().bool()
        for idx in range(min(abs_err.shape[-1], len(VARIABLES))):
            values = abs_err[:, :, idx][mask[:, :, idx]]
            if values.numel():
                errors[idx].append(values)

    out = []
    for parts in errors:
        out.append(torch.cat(parts) if parts else torch.empty(0))
    return out


def parse_pattern_rate_seed(config: dict[str, Any], path: Path) -> dict[str, Any]:
    missing = config.get("missing", {})
    return {
        "run_name": config.get("run_name", path.stem),
        "pattern": missing.get("target_pattern", missing.get("pattern", "")),
        "rate": float(missing.get("rate", float("nan"))),
        "seed": int(config.get("seed", -1)),
        "model": config.get("model", {}).get("name", ""),
    }


def analyze(
    config_paths: list[Path],
    out_dir: Path,
    alpha: float,
    device: torch.device,
    max_batches: int | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in config_paths:
        config = load_config(path)
        checkpoint = Path(config["output_dir"]) / "best.pt"
        if not checkpoint.exists():
            rows.append({**parse_pattern_rate_seed(config, path), "variable": "all", "status": "missing_checkpoint"})
            continue
        val_errors = collect_abs_errors(config, checkpoint, "val", device, max_batches=max_batches)
        test_errors = collect_abs_errors(config, checkpoint, "test", device, max_batches=max_batches)
        base = parse_pattern_rate_seed(config, path)
        pooled_val = torch.cat([x for x in val_errors if x.numel()]) if any(x.numel() for x in val_errors) else torch.empty(0)
        pooled_test = torch.cat([x for x in test_errors if x.numel()]) if any(x.numel() for x in test_errors) else torch.empty(0)
        for name, val, test in [*zip(VARIABLES, val_errors, test_errors), ("all", pooled_val, pooled_test)]:
            qhat = conformal_quantile(val, alpha)
            if test.numel() and math.isfinite(qhat):
                coverage = float((test <= qhat).float().mean().item())
                test_mae = float(test.mean().item())
                n_test = int(test.numel())
            else:
                coverage = float("nan")
                test_mae = float("nan")
                n_test = 0
            rows.append(
                {
                    **base,
                    "variable": name,
                    "status": "ok",
                    "alpha": alpha,
                    "target_coverage": 1.0 - alpha,
                    "qhat": qhat,
                    "interval_width": 2.0 * qhat if math.isfinite(qhat) else float("nan"),
                    "test_coverage": coverage,
                    "test_mae": test_mae,
                    "n_val": int(val.numel()),
                    "n_test": n_test,
                }
            )
    return pd.DataFrame(rows)


def write_report(rows: pd.DataFrame, out_dir: Path) -> None:
    ok = rows[rows["status"].eq("ok")].copy()
    lines = ["# Conformal Uncertainty Report", ""]
    lines.append(f"Evaluated rows: {len(ok)}; skipped rows: {len(rows) - len(ok)}.")
    if not ok.empty:
        summary = (
            ok.groupby("variable", as_index=False)
            .agg(test_coverage=("test_coverage", "mean"), interval_width=("interval_width", "mean"), test_mae=("test_mae", "mean"))
            .sort_values("variable")
        )
        lines.extend(["", "## Mean Coverage by Variable", ""])
        lines.append(markdown_table(summary, floatfmt=".4f"))
    missing = rows[rows["status"].ne("ok")]
    if not missing.empty:
        lines.extend(["", "## Skipped Configs", ""])
        lines.append(markdown_table(missing[["run_name", "status"]]))
    (out_dir / "conformal_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(df: pd.DataFrame, floatfmt: str = "") -> str:
    """Small dependency-free Markdown table writer."""
    if df.empty:
        return ""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if isinstance(value, float) and floatfmt:
                vals.append(format(value, floatfmt))
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-glob", default=DEFAULT_CONFIG_GLOB)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-configs", type=int)
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args()

    paths = sorted(Path().glob(args.config_glob))
    if args.max_configs is not None:
        paths = paths[: args.max_configs]
    rows = analyze(paths, args.out_dir, args.alpha, torch.device(args.device), args.max_batches)
    out_csv = args.out_dir / "conformal_intervals.csv"
    rows.to_csv(out_csv, index=False)
    write_report(rows, args.out_dir)
    print(f"wrote {out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
