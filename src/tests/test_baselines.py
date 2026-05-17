from __future__ import annotations

import torch

from src.baselines.linear_interp import linear_interpolate
from src.baselines.locf import locf_impute


def test_linear_interpolation_preserves_observed_values() -> None:
    x = torch.tensor([[[0.0], [1.0], [0.0], [0.0], [4.0]]])
    obs = torch.tensor([[[1.0], [1.0], [0.0], [0.0], [1.0]]])

    y = linear_interpolate(x, obs)

    assert y.shape == x.shape
    assert torch.allclose(y[obs.bool()], x[obs.bool()])
    assert torch.allclose(y.flatten(), torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]))


def test_linear_interpolation_handles_edges_and_empty_series() -> None:
    x = torch.tensor(
        [
            [[0.0, 0.0], [2.0, 0.0], [0.0, 0.0], [8.0, 0.0]],
        ]
    )
    obs = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
        ]
    )

    y = linear_interpolate(x, obs)

    assert torch.allclose(y[0, :, 0], torch.tensor([2.0, 2.0, 5.0, 8.0]))
    assert torch.allclose(y[0, :, 1], torch.zeros(4))


def test_locf_expected_behavior() -> None:
    x = torch.tensor([[[0.0], [2.0], [0.0], [0.0], [9.0], [0.0]]])
    obs = torch.tensor([[[0.0], [1.0], [0.0], [0.0], [1.0], [0.0]]])

    y = locf_impute(x, obs)

    assert torch.allclose(y.flatten(), torch.tensor([2.0, 2.0, 2.0, 2.0, 9.0, 9.0]))
    assert torch.allclose(y[obs.bool()], x[obs.bool()])


def test_baselines_validate_shapes() -> None:
    x = torch.zeros(2, 3, 4)
    obs = torch.zeros(2, 3, 5)
    for fn in [linear_interpolate, locf_impute]:
        try:
            fn(x, obs)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
