"""Linear interpolation baseline for normalized time-series windows."""

from __future__ import annotations

import torch


def _interp_1d(values: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    idx = torch.arange(values.numel(), device=values.device, dtype=values.dtype)
    obs_idx = idx[observed]
    obs_val = values[observed]
    if obs_idx.numel() == 0:
        return torch.zeros_like(values)
    if obs_idx.numel() == 1:
        return torch.full_like(values, obs_val[0])

    right_pos = torch.searchsorted(obs_idx, idx, right=False)
    right_pos = torch.clamp(right_pos, 0, obs_idx.numel() - 1)
    left_pos = torch.clamp(right_pos - 1, 0, obs_idx.numel() - 1)

    left_idx = obs_idx[left_pos]
    right_idx = obs_idx[right_pos]
    left_val = obs_val[left_pos]
    right_val = obs_val[right_pos]
    denom = torch.clamp(right_idx - left_idx, min=1.0)
    weight = (idx - left_idx) / denom
    out = left_val + weight * (right_val - left_val)
    out[idx <= obs_idx[0]] = obs_val[0]
    out[idx >= obs_idx[-1]] = obs_val[-1]
    return out


def linear_interpolate(x: torch.Tensor, obs_mask: torch.Tensor) -> torch.Tensor:
    """Fill missing values by per-sample, per-variable linear interpolation.

    Args:
        x: `(B, T, C)` tensor. Missing positions may contain any fill value.
        obs_mask: `(B, T, C)` binary tensor where 1 means observed.

    Returns:
        Imputed tensor preserving observed values exactly.
    """
    if x.shape != obs_mask.shape or x.ndim != 3:
        raise ValueError(f"x and obs_mask must both be (B, T, C), got {tuple(x.shape)} and {tuple(obs_mask.shape)}")
    out = x.clone()
    for b in range(x.shape[0]):
        for c in range(x.shape[2]):
            observed = obs_mask[b, :, c].bool()
            out[b, :, c] = _interp_1d(x[b, :, c], observed)
            out[b, observed, c] = x[b, observed, c]
    return out


class LinearInterpolationImputer:
    """Stateless class wrapper matching the baseline interface."""

    def __call__(self, x: torch.Tensor, obs_mask: torch.Tensor) -> torch.Tensor:
        return linear_interpolate(x, obs_mask)
