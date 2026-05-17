"""Evaluation metrics for imputation experiments."""

from __future__ import annotations

import torch


VARIABLES = ("T", "RH", "wspd", "P", "q")


def masked_mae_rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    """Compute mean and per-variable MAE/RMSE over `mask == 1` positions."""
    if pred.shape != target.shape or pred.shape != mask.shape or pred.ndim != 3:
        raise ValueError(f"pred, target, mask must share shape (N,T,C), got {pred.shape}, {target.shape}, {mask.shape}")
    err = pred - target
    out: dict[str, float] = {}
    mae_values = []
    rmse_values = []
    for i, name in enumerate(VARIABLES[: pred.shape[-1]]):
        m = mask[:, :, i].bool()
        if m.any():
            mae = err[:, :, i][m].abs().mean()
            rmse = torch.sqrt((err[:, :, i][m] ** 2).mean())
            out[f"mae_{name}"] = float(mae.item())
            out[f"rmse_{name}"] = float(rmse.item())
            mae_values.append(mae)
            rmse_values.append(rmse)
        else:
            out[f"mae_{name}"] = float("nan")
            out[f"rmse_{name}"] = float("nan")
    out["mae_mean"] = float(torch.stack(mae_values).mean().item()) if mae_values else float("nan")
    out["rmse_mean"] = float(torch.stack(rmse_values).mean().item()) if rmse_values else float("nan")
    return out
