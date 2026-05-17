"""Last-observation-carried-forward baseline."""

from __future__ import annotations

import torch


def _locf_1d(values: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    obs_idx = torch.nonzero(observed, as_tuple=False).flatten()
    if obs_idx.numel() == 0:
        return torch.zeros_like(values)
    out = values.clone()
    first = int(obs_idx[0].item())
    out[:first] = values[first]
    last = values[first]
    for i in range(first, values.numel()):
        if bool(observed[i].item()):
            last = values[i]
        else:
            out[i] = last
    return out


def locf_impute(x: torch.Tensor, obs_mask: torch.Tensor) -> torch.Tensor:
    """Fill missing values with the last observed value per variable.

    Leading missing values are filled with the first available observation.
    Observed values are preserved exactly.
    """
    if x.shape != obs_mask.shape or x.ndim != 3:
        raise ValueError(f"x and obs_mask must both be (B, T, C), got {tuple(x.shape)} and {tuple(obs_mask.shape)}")
    out = x.clone()
    for b in range(x.shape[0]):
        for c in range(x.shape[2]):
            observed = obs_mask[b, :, c].bool()
            out[b, :, c] = _locf_1d(x[b, :, c], observed)
            out[b, observed, c] = x[b, observed, c]
    return out


class LOCFImputer:
    """Stateless class wrapper matching the baseline interface."""

    def __call__(self, x: torch.Tensor, obs_mask: torch.Tensor) -> torch.Tensor:
        return locf_impute(x, obs_mask)
