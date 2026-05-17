from __future__ import annotations

import torch

from src.baselines.itransformer_impute import ITransformerImputer, masked_mse_loss


def test_itransformer_imputer_forward_shape() -> None:
    model = ITransformerImputer(seq_len=24, n_vars=5, n_time=4, d_model=32, n_heads=4, n_layers=1, d_ff=64, dropout=0.0)
    x = torch.randn(3, 24, 5)
    missing = torch.zeros(3, 24, 5)
    missing[:, 4:9, 2] = 1
    time_enc = torch.randn(3, 24, 4)

    y = model(x, missing, time_enc)

    assert y.shape == (3, 24, 5)
    assert torch.isfinite(y).all()


def test_masked_mse_loss_uses_only_label_positions() -> None:
    pred = torch.tensor([[[1.0, 10.0], [3.0, 30.0]]])
    target = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])
    label_mask = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])

    loss = masked_mse_loss(pred, target, label_mask)

    assert torch.allclose(loss, torch.tensor((1.0 + 4.0) / 2.0))


def test_masked_mse_loss_handles_empty_label_mask() -> None:
    pred = torch.ones(2, 3, 4)
    target = torch.zeros(2, 3, 4)
    label_mask = torch.zeros(2, 3, 4)
    assert masked_mse_loss(pred, target, label_mask).item() == 0.0
