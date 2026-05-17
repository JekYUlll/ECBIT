from __future__ import annotations

import torch

from src.utils.block_missing import (
    apply_mask,
    artificial_label_mask,
    missing_variables,
    simulate_block_missing,
    simulate_mcar_missing,
)


def block_lengths(mask_1d: torch.Tensor) -> list[int]:
    values = mask_1d.to(torch.int8).cpu().tolist()
    lengths: list[int] = []
    run = 0
    for value in values:
        if value:
            run += 1
        elif run:
            lengths.append(run)
            run = 0
    if run:
        lengths.append(run)
    return lengths


def test_block_missing_respects_observed_mask() -> None:
    obs_mask = torch.ones(64, 3)
    obs_mask[10:30, 1] = 0.0
    generator = torch.Generator().manual_seed(7)

    mask = simulate_block_missing(
        obs_mask.shape,
        missing_rate=0.4,
        min_block=4,
        max_block=8,
        obs_mask=obs_mask,
        generator=generator,
    )

    assert mask.shape == obs_mask.shape
    assert torch.all(mask[obs_mask == 0] == 0)
    assert mask.sum() > 0
    assert set(torch.unique(mask).tolist()).issubset({0.0, 1.0})


def test_block_missing_creates_contiguous_variable_blocks() -> None:
    generator = torch.Generator().manual_seed(11)
    mask = simulate_block_missing((80, 1), missing_rate=0.35, min_block=5, max_block=9, generator=generator)

    lengths = block_lengths(mask[:, 0])
    assert lengths
    assert max(lengths) >= 5
    assert mask.mean() > 0.1


def test_global_block_missing_masks_multiple_variables_together() -> None:
    generator = torch.Generator().manual_seed(3)
    mask = simulate_block_missing(
        (48, 4),
        missing_rate=0.2,
        min_block=3,
        max_block=5,
        variable_wise=False,
        generator=generator,
    )

    rows_with_missing = mask.sum(dim=1) > 0
    assert rows_with_missing.any()
    assert torch.all(mask[rows_with_missing].sum(dim=1) == 4)


def test_mcar_and_apply_mask_helpers() -> None:
    x = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    obs_mask = torch.ones_like(x)
    obs_mask[0, 0] = 0
    generator = torch.Generator().manual_seed(5)
    mask = simulate_mcar_missing(x.shape, missing_rate=0.5, obs_mask=obs_mask, generator=generator)

    assert mask[0, 0] == 0
    y = apply_mask(x, mask, fill_value=-1.0)
    assert torch.all(y[mask.bool()] == -1.0)
    assert torch.all(y[~mask.bool()] == x[~mask.bool()])


def test_label_mask_and_variable_gate() -> None:
    artificial = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 1, 1]], dtype=torch.float32)
    observed = torch.tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=torch.float32)

    labels = artificial_label_mask(artificial, observed)
    assert labels.tolist() == [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]]
    assert missing_variables(labels).tolist() == [1.0, 1.0, 1.0]
