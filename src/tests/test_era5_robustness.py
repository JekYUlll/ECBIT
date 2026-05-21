from __future__ import annotations

import torch

from scripts.evaluate_era5_robustness import mask_era5_variable, temporal_downsample_era5


def test_mask_era5_variable_replaces_only_one_channel() -> None:
    era5 = torch.arange(2 * 6 * 3, dtype=torch.float32).reshape(2, 6, 3)
    train_mean = torch.tensor([10.0, 20.0, 30.0])

    masked = mask_era5_variable(era5, 1, train_mean)

    assert torch.allclose(masked[:, :, 0], era5[:, :, 0])
    assert torch.all(masked[:, :, 1] == 20.0)
    assert torch.allclose(masked[:, :, 2], era5[:, :, 2])


def test_temporal_downsample_era5_restores_shape() -> None:
    era5 = torch.randn(4, 12, 5)

    restored = temporal_downsample_era5(era5, factor=3)

    assert restored.shape == era5.shape
    assert torch.isfinite(restored).all()
    assert torch.allclose(temporal_downsample_era5(era5, factor=1), era5)
