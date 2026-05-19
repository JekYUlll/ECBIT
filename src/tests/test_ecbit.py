from __future__ import annotations

import torch

from src.models.ecbit import ConditionalCrossAttention, ECBIT, GatedFeatureInjection, VariateTokenEncoder


def test_variate_token_encoder_shape() -> None:
    encoder = VariateTokenEncoder(
        seq_len=12,
        n_vars=5,
        per_step_dim=6,
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
        dropout=0.0,
    )
    x = torch.randn(3, 5, 12, 6)
    y = encoder(x)
    assert y.shape == (3, 5, 16)


def test_conditional_cross_attention_zero_gate_independent_of_era5() -> None:
    layer = ConditionalCrossAttention(d_model=16, n_heads=4, dropout=0.0)
    layer.eval()
    z_obs = torch.randn(2, 5, 16)
    z_era5_a = torch.randn(2, 5, 16)
    z_era5_b = torch.randn(2, 5, 16) * 10.0
    missing_vars = torch.zeros(2, 5)

    y_a = layer(z_obs, z_era5_a, missing_vars)
    y_b = layer(z_obs, z_era5_b, missing_vars)
    assert torch.allclose(y_a, y_b, atol=1e-6)


def test_gated_feature_injection_zero_missing_independent_of_era5() -> None:
    layer = GatedFeatureInjection(d_model=16, dropout=0.0)
    layer.eval()
    z_obs = torch.randn(2, 5, 16)
    z_era5_a = torch.randn(2, 5, 16)
    z_era5_b = torch.randn(2, 5, 16) * 10.0
    missing_vars = torch.zeros(2, 5)

    y_a = layer(z_obs, z_era5_a, missing_vars)
    y_b = layer(z_obs, z_era5_b, missing_vars)
    assert torch.allclose(y_a, y_b, atol=1e-6)


def test_ecbit_forward_shape() -> None:
    model = ECBIT(
        seq_len=24,
        n_vars=5,
        n_time=4,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
        fusion_type="gated",
    )
    x = torch.randn(4, 24, 5)
    missing = torch.zeros(4, 24, 5)
    missing[:, 3:8, 2] = 1.0
    era5 = torch.randn(4, 24, 5)
    time_enc = torch.randn(4, 24, 4)

    y = model(x, missing, era5, time_enc)
    assert y.shape == (4, 24, 5)
    assert torch.isfinite(y).all()


def test_ecbit_observable_variable_output_independent_of_era5() -> None:
    torch.manual_seed(13)
    model = ECBIT(
        seq_len=16,
        n_vars=3,
        n_time=4,
        d_model=24,
        n_heads=4,
        n_layers=1,
        d_ff=48,
        dropout=0.0,
        fusion_type="gated",
    )
    model.eval()
    x = torch.randn(2, 16, 3)
    missing = torch.zeros(2, 16, 3)
    missing[:, 4:10, 1] = 1.0
    era5_a = torch.randn(2, 16, 3)
    era5_b = torch.randn(2, 16, 3) * 20.0
    time_enc = torch.randn(2, 16, 4)

    y_a = model(x, missing, era5_a, time_enc)
    y_b = model(x, missing, era5_b, time_enc)

    assert torch.allclose(y_a[:, :, 0], y_b[:, :, 0], atol=1e-5)
    assert torch.allclose(y_a[:, :, 2], y_b[:, :, 2], atol=1e-5)


def test_ecbit_no_era5_ablation_shape() -> None:
    model = ECBIT(
        seq_len=10,
        n_vars=2,
        n_time=4,
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
        dropout=0.0,
        use_era5=False,
    )
    y = model(torch.randn(1, 10, 2), torch.zeros(1, 10, 2), None, torch.randn(1, 10, 4))
    assert y.shape == (1, 10, 2)
