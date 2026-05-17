"""ERA5 direct substitution baseline with train-set bias correction."""

from __future__ import annotations

import numpy as np
import torch


class ERA5DirectImputer:
    """Fill missing AWS positions with bias-corrected ERA5 values."""

    def __init__(self) -> None:
        self.bias: torch.Tensor | None = None

    def fit(self, x: torch.Tensor | np.ndarray, era5: torch.Tensor | np.ndarray, obs_mask: torch.Tensor | np.ndarray) -> "ERA5DirectImputer":
        x_t = torch.as_tensor(x, dtype=torch.float32)
        e_t = torch.as_tensor(era5, dtype=torch.float32)
        m_t = torch.as_tensor(obs_mask, dtype=torch.bool)
        if x_t.shape != e_t.shape or x_t.shape != m_t.shape or x_t.ndim != 3:
            raise ValueError(f"x, era5, obs_mask must share shape (N, T, C), got {x_t.shape}, {e_t.shape}, {m_t.shape}")

        n_vars = x_t.shape[-1]
        bias = torch.zeros(n_vars, dtype=torch.float32)
        for c in range(n_vars):
            valid = m_t[:, :, c]
            if valid.any():
                bias[c] = (x_t[:, :, c][valid] - e_t[:, :, c][valid]).mean()
        self.bias = bias
        return self

    def impute(
        self,
        x: torch.Tensor | np.ndarray,
        era5: torch.Tensor | np.ndarray,
        obs_mask: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        if self.bias is None:
            raise RuntimeError("ERA5DirectImputer must be fit before impute")
        x_t = torch.as_tensor(x, dtype=torch.float32)
        e_t = torch.as_tensor(era5, dtype=torch.float32)
        m_t = torch.as_tensor(obs_mask, dtype=torch.bool)
        if x_t.shape != e_t.shape or x_t.shape != m_t.shape or x_t.ndim != 3:
            raise ValueError(f"x, era5, obs_mask must share shape (N, T, C), got {x_t.shape}, {e_t.shape}, {m_t.shape}")
        corrected = e_t + self.bias.to(device=e_t.device, dtype=e_t.dtype).view(1, 1, -1)
        return torch.where(m_t, x_t, corrected)


def era5_direct_impute(
    x: torch.Tensor | np.ndarray,
    era5: torch.Tensor | np.ndarray,
    obs_mask: torch.Tensor | np.ndarray,
    bias: torch.Tensor | np.ndarray,
) -> torch.Tensor:
    """Functional direct substitution helper when bias is already known."""
    x_t = torch.as_tensor(x, dtype=torch.float32)
    e_t = torch.as_tensor(era5, dtype=torch.float32)
    m_t = torch.as_tensor(obs_mask, dtype=torch.bool)
    b_t = torch.as_tensor(bias, dtype=torch.float32, device=e_t.device).view(1, 1, -1)
    if x_t.shape != e_t.shape or x_t.shape != m_t.shape or x_t.ndim != 3:
        raise ValueError(f"x, era5, obs_mask must share shape (N, T, C), got {x_t.shape}, {e_t.shape}, {m_t.shape}")
    if b_t.shape[-1] != x_t.shape[-1]:
        raise ValueError(f"bias has {b_t.shape[-1]} variables, expected {x_t.shape[-1]}")
    return torch.where(m_t, x_t, e_t + b_t)
