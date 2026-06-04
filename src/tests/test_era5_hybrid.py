from __future__ import annotations

import torch

from src.baselines.era5_hybrid import endpoint_residual_anchor


def test_endpoint_residual_anchor_interpolates_endpoint_residuals() -> None:
    calibrated = torch.tensor([[[10.0], [10.0], [10.0], [10.0], [10.0]]])
    x_obs = torch.tensor([[[12.0], [0.0], [0.0], [0.0], [18.0]]])
    obs = torch.tensor([[[1.0], [0.0], [0.0], [0.0], [1.0]]])

    y = endpoint_residual_anchor(x_obs, calibrated, obs)

    assert torch.allclose(y.flatten(), torch.tensor([12.0, 13.5, 15.0, 16.5, 18.0]))


def test_endpoint_residual_anchor_uses_single_endpoint_at_edges() -> None:
    calibrated = torch.tensor([[[10.0], [10.0], [10.0], [10.0]]])
    x_obs = torch.tensor([[[0.0], [0.0], [14.0], [0.0]]])
    obs = torch.tensor([[[0.0], [0.0], [1.0], [0.0]]])

    y = endpoint_residual_anchor(x_obs, calibrated, obs)

    assert torch.allclose(y.flatten(), torch.tensor([14.0, 14.0, 14.0, 14.0]))


def test_endpoint_residual_anchor_falls_back_without_endpoints() -> None:
    calibrated = torch.tensor([[[1.0], [2.0], [3.0]]])
    x_obs = torch.zeros_like(calibrated)
    obs = torch.zeros_like(calibrated)

    y = endpoint_residual_anchor(x_obs, calibrated, obs)

    assert torch.allclose(y, calibrated)


def test_endpoint_residual_anchor_validates_shapes() -> None:
    try:
        endpoint_residual_anchor(torch.zeros(1, 2, 1), torch.zeros(1, 2, 1), torch.zeros(1, 2, 2))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
