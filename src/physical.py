"""Physical consistency helpers for Antarctic AWS imputation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def specific_humidity_gkg(t_c: torch.Tensor, rh_pct: torch.Tensor, p_hpa: torch.Tensor) -> torch.Tensor:
    """Compute specific humidity in g/kg from temperature, RH, and pressure."""
    rh = torch.clamp(rh_pct, 0.0, 100.0)
    es = 6.112 * torch.exp(17.67 * t_c / (t_c + 243.5))
    e = rh / 100.0 * es
    denom = torch.clamp(p_hpa - 0.378 * e, min=1.0e-4)
    q_kgkg = 0.622 * e / denom
    return q_kgkg * 1000.0


def unnormalize(x_norm: torch.Tensor, norm_mean: torch.Tensor, norm_std: torch.Tensor) -> torch.Tensor:
    """Restore normalized `(B,T,C)` values to physical units."""
    if x_norm.ndim != 3:
        raise ValueError(f"x_norm must be (B,T,C), got {tuple(x_norm.shape)}")
    if norm_mean.ndim == 1:
        norm_mean = norm_mean.unsqueeze(0)
    if norm_std.ndim == 1:
        norm_std = norm_std.unsqueeze(0)
    if norm_mean.shape != norm_std.shape or norm_mean.shape[-1] != x_norm.shape[-1]:
        raise ValueError(
            "norm_mean and norm_std must be (C,) or (B,C) with the same channel count as x_norm"
        )
    return x_norm * norm_std.to(x_norm.device, x_norm.dtype).unsqueeze(1) + norm_mean.to(x_norm.device, x_norm.dtype).unsqueeze(1)


def physical_consistency_loss(
    pred_norm: torch.Tensor,
    label_mask: torch.Tensor,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    *,
    delta: float = 1.0,
) -> torch.Tensor:
    """Penalize mismatch between predicted q and q implied by predicted T/RH/P.

    Variables are ordered as `[T, RH, wspd, P, q]`. The residual is measured in
    normalized specific-humidity units, so the weight is comparable across
    stations. The mask includes timesteps where any thermodynamic channel is an
    artificial target.
    """
    if pred_norm.shape != label_mask.shape or pred_norm.ndim != 3 or pred_norm.shape[-1] < 5:
        raise ValueError("pred_norm and label_mask must share shape (B,T,C) with at least five channels")

    raw = unnormalize(pred_norm, norm_mean, norm_std)
    q_expected = specific_humidity_gkg(raw[:, :, 0], raw[:, :, 1], raw[:, :, 3])

    if norm_mean.ndim == 1:
        mean_q = norm_mean[4].to(pred_norm.device, pred_norm.dtype)
        std_q = norm_std[4].to(pred_norm.device, pred_norm.dtype)
    else:
        mean_q = norm_mean[:, 4].to(pred_norm.device, pred_norm.dtype).unsqueeze(1)
        std_q = norm_std[:, 4].to(pred_norm.device, pred_norm.dtype).unsqueeze(1)
    q_expected_norm = (q_expected - mean_q) / torch.clamp(std_q, min=1.0e-6)

    thermo_mask = label_mask[:, :, [0, 1, 3, 4]].sum(dim=-1) > 0
    if not thermo_mask.any():
        return pred_norm.sum() * 0.0
    residual = F.smooth_l1_loss(pred_norm[:, :, 4], q_expected_norm, beta=float(delta), reduction="none")
    return residual[thermo_mask].mean()
