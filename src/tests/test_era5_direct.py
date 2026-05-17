from __future__ import annotations

import torch

from src.baselines.era5_direct import ERA5DirectImputer, era5_direct_impute


def test_era5_direct_fit_estimates_per_variable_bias() -> None:
    x = torch.tensor([[[3.0, 10.0], [5.0, 20.0]], [[7.0, 30.0], [9.0, 40.0]]])
    era5 = torch.tensor([[[1.0, 7.0], [2.0, 16.0]], [[4.0, 25.0], [6.0, 34.0]]])
    obs = torch.ones_like(x)

    imputer = ERA5DirectImputer().fit(x, era5, obs)

    assert torch.allclose(imputer.bias, torch.tensor([2.75, 4.5]))


def test_era5_direct_impute_preserves_observed_and_fills_missing() -> None:
    x = torch.tensor([[[1.0, 0.0], [2.0, 4.0], [0.0, 0.0]]])
    era5 = torch.tensor([[[10.0, 20.0], [10.0, 20.0], [10.0, 20.0]]])
    obs = torch.tensor([[[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]])

    y = era5_direct_impute(x, era5, obs, bias=torch.tensor([-1.0, 2.0]))

    assert torch.allclose(y[obs.bool()], x[obs.bool()])
    assert y[0, 0, 1] == 22.0
    assert y[0, 2, 0] == 9.0
    assert y[0, 2, 1] == 22.0


def test_era5_direct_requires_fit() -> None:
    imputer = ERA5DirectImputer()
    try:
        imputer.impute(torch.zeros(1, 2, 1), torch.zeros(1, 2, 1), torch.ones(1, 2, 1))
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError")
