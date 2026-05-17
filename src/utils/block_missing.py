"""Block-missing simulation utilities for ECBIT imputation experiments."""

from __future__ import annotations

import torch


def _randint(generator: torch.Generator | None, low: int, high: int, device: torch.device) -> int:
    return int(torch.randint(low, high, (1,), generator=generator, device=device).item())


def _rand(generator: torch.Generator | None, device: torch.device) -> float:
    return float(torch.rand((), generator=generator, device=device).item())


def simulate_block_missing(
    shape: tuple[int, int] | torch.Size,
    missing_rate: float = 0.3,
    min_block: int = 6,
    max_block: int = 72,
    variable_wise: bool = True,
    obs_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Create a binary artificial-missing mask with contiguous blocks.

    Args:
        shape: `(T, C)` sequence shape.
        missing_rate: Approximate target fraction of artificial missing values.
        min_block: Minimum contiguous block length.
        max_block: Maximum contiguous block length.
        variable_wise: If true, sample independent variable blocks. If false,
            sampled blocks cover all variables at once.
        obs_mask: Optional `(T, C)` binary tensor where 1 means real observation
            exists. Artificial labels are restricted to `obs_mask == 1`.
        generator: Optional torch RNG for deterministic tests.

    Returns:
        Float tensor `(T, C)` where 1 means artificial missing label.
    """
    if len(shape) != 2:
        raise ValueError(f"shape must be (T, C), got {tuple(shape)}")
    if not 0.0 <= missing_rate <= 1.0:
        raise ValueError("missing_rate must be in [0, 1]")
    if min_block < 1 or max_block < min_block:
        raise ValueError("require 1 <= min_block <= max_block")

    if obs_mask is not None:
        if tuple(obs_mask.shape) != tuple(shape):
            raise ValueError(f"obs_mask shape {tuple(obs_mask.shape)} != {tuple(shape)}")
        device = obs_mask.device
        allowed = obs_mask.bool()
    else:
        device = torch.device("cpu")
        allowed = torch.ones(tuple(shape), dtype=torch.bool, device=device)

    t_steps, n_vars = int(shape[0]), int(shape[1])
    mask = torch.zeros((t_steps, n_vars), dtype=torch.float32, device=device)
    target = int(round(float(allowed.sum().item()) * missing_rate))
    if target <= 0 or int(allowed.sum().item()) == 0:
        return mask

    attempts = 0
    max_attempts = max(100, target * 20)
    while int(mask.sum().item()) < target and attempts < max_attempts:
        attempts += 1
        start = _randint(generator, 0, t_steps, device)
        block_len = _randint(generator, min_block, max_block + 1, device)
        end = min(start + block_len, t_steps)
        if variable_wise:
            var = _randint(generator, 0, n_vars, device)
            mask[start:end, var] = torch.where(
                allowed[start:end, var],
                torch.ones_like(mask[start:end, var]),
                mask[start:end, var],
            )
        else:
            candidate = allowed[start:end]
            mask[start:end] = torch.where(candidate, torch.ones_like(mask[start:end]), mask[start:end])

    return mask * allowed.float()


def simulate_mcar_missing(
    shape: tuple[int, int] | torch.Size,
    missing_rate: float = 0.3,
    obs_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Create point-wise MCAR artificial-missing labels for ablations."""
    if len(shape) != 2:
        raise ValueError(f"shape must be (T, C), got {tuple(shape)}")
    if not 0.0 <= missing_rate <= 1.0:
        raise ValueError("missing_rate must be in [0, 1]")

    if obs_mask is not None:
        if tuple(obs_mask.shape) != tuple(shape):
            raise ValueError(f"obs_mask shape {tuple(obs_mask.shape)} != {tuple(shape)}")
        device = obs_mask.device
        allowed = obs_mask.bool()
    else:
        device = torch.device("cpu")
        allowed = torch.ones(tuple(shape), dtype=torch.bool, device=device)
    mask = (torch.rand(tuple(shape), generator=generator, device=device) < missing_rate) & allowed
    return mask.float()


def apply_mask(x: torch.Tensor, mask: torch.Tensor, fill_value: float = 0.0) -> torch.Tensor:
    """Return `x` with positions where `mask == 1` replaced by `fill_value`."""
    if x.shape != mask.shape:
        raise ValueError(f"x shape {tuple(x.shape)} != mask shape {tuple(mask.shape)}")
    out = x.clone()
    out[mask.bool()] = fill_value
    return out


def artificial_label_mask(artificial_mask: torch.Tensor, obs_mask: torch.Tensor) -> torch.Tensor:
    """Loss mask for artificially hidden positions that were truly observed."""
    if artificial_mask.shape != obs_mask.shape:
        raise ValueError(
            f"artificial_mask shape {tuple(artificial_mask.shape)} != obs_mask shape {tuple(obs_mask.shape)}"
        )
    return artificial_mask.bool().logical_and(obs_mask.bool()).float()


def missing_variables(mask: torch.Tensor) -> torch.Tensor:
    """Return variable-level gate `(C,)`, 1 where any timestep is missing."""
    if mask.ndim != 2:
        raise ValueError(f"mask must be (T, C), got {tuple(mask.shape)}")
    return (mask.bool().sum(dim=0) > 0).float()
