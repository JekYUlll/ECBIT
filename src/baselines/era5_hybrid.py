"""Operational ERA5 hybrid baselines."""

from __future__ import annotations

import torch


def endpoint_residual_anchor(
    x_obs: torch.Tensor,
    calibrated_era5: torch.Tensor,
    obs_mask: torch.Tensor,
) -> torch.Tensor:
    """Anchor calibrated ERA5 to visible AWS endpoints.

    The method preserves visible AWS values. For each contiguous unobserved
    segment, it adds a linearly interpolated AWS-minus-ERA5 residual estimated
    from the nearest visible endpoints. If only one endpoint is available, the
    single endpoint residual is carried across the segment. If no endpoint is
    available, the method falls back to calibrated ERA5.
    """
    x_t = torch.as_tensor(x_obs, dtype=torch.float32)
    e_t = torch.as_tensor(calibrated_era5, dtype=torch.float32)
    m_t = torch.as_tensor(obs_mask, dtype=torch.bool)
    if x_t.shape != e_t.shape or x_t.shape != m_t.shape or x_t.ndim != 3:
        raise ValueError(
            "x_obs, calibrated_era5, and obs_mask must share shape "
            f"(N,T,C), got {x_t.shape}, {e_t.shape}, {m_t.shape}"
        )

    out = e_t.clone()
    n_windows, seq_len, n_vars = x_t.shape
    for n in range(n_windows):
        for c in range(n_vars):
            observed = m_t[n, :, c]
            if observed.all():
                out[n, :, c] = x_t[n, :, c]
                continue
            if not observed.any():
                continue

            t = 0
            while t < seq_len:
                if observed[t]:
                    t += 1
                    continue
                start = t
                while t < seq_len and not observed[t]:
                    t += 1
                end = t

                left = start - 1 if start > 0 and observed[start - 1] else None
                right = end if end < seq_len and observed[end] else None
                if left is None and right is None:
                    continue
                positions = torch.arange(start, end, device=e_t.device, dtype=e_t.dtype)
                if left is not None:
                    left_res = x_t[n, left, c] - e_t[n, left, c]
                if right is not None:
                    right_res = x_t[n, right, c] - e_t[n, right, c]

                if left is not None and right is not None:
                    denom = float(right - left)
                    alpha = (positions - float(left)) / denom
                    residual = (1.0 - alpha) * left_res + alpha * right_res
                elif left is not None:
                    residual = torch.full_like(positions, float(left_res))
                else:
                    residual = torch.full_like(positions, float(right_res))
                out[n, start:end, c] = e_t[n, start:end, c] + residual

    return torch.where(m_t, x_t, out)
