from __future__ import annotations

import pytest
import torch

from src.data.impute_dataset import STATION_FEATURE_DIM
from src.models.ecbit_sabc import ECBITSABC, StationAdaptiveBiasCorrection


def test_station_adaptive_bias_correction_zero_initialized() -> None:
    layer = StationAdaptiveBiasCorrection(
        d_model=16,
        n_vars=3,
        n_station_features=STATION_FEATURE_DIM,
        hidden_dim=8,
        dropout=0.0,
    )
    z_era5 = torch.randn(2, 3, 16)
    station_features = torch.randn(2, STATION_FEATURE_DIM)

    y = layer(z_era5, station_features)

    assert torch.allclose(y, z_era5, atol=1.0e-7)


def test_station_adaptive_bias_correction_validates_station_feature_shape() -> None:
    layer = StationAdaptiveBiasCorrection(
        d_model=16,
        n_vars=3,
        n_station_features=STATION_FEATURE_DIM,
        hidden_dim=8,
        dropout=0.0,
    )
    with pytest.raises(ValueError, match="station_features"):
        layer(torch.randn(2, 3, 16), torch.randn(2, STATION_FEATURE_DIM + 1))


def test_ecbit_sabc_forward_shape() -> None:
    model = ECBITSABC(
        seq_len=24,
        n_vars=5,
        n_time=4,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
        fusion_type="gated",
        n_station_features=STATION_FEATURE_DIM,
        sabc_hidden_dim=16,
    )
    x = torch.randn(4, 24, 5)
    missing = torch.zeros(4, 24, 5)
    missing[:, 3:8, 2] = 1.0
    era5 = torch.randn(4, 24, 5)
    time_enc = torch.randn(4, 24, 4)
    station_features = torch.randn(4, STATION_FEATURE_DIM)

    y = model(x, missing, era5, time_enc, station_features)

    assert y.shape == (4, 24, 5)
    assert torch.isfinite(y).all()
