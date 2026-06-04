from __future__ import annotations

import torch

from src.physical import physical_consistency_loss, specific_humidity_gkg, unnormalize


def test_specific_humidity_shape_and_positive_values() -> None:
    t = torch.tensor([[0.0, -10.0]])
    rh = torch.tensor([[50.0, 80.0]])
    p = torch.tensor([[1000.0, 850.0]])
    q = specific_humidity_gkg(t, rh, p)
    assert q.shape == t.shape
    assert torch.all(q > 0)


def test_unnormalize_accepts_batch_stats() -> None:
    x = torch.zeros(2, 3, 5)
    mean = torch.arange(10, dtype=torch.float32).reshape(2, 5)
    std = torch.ones_like(mean) * 2
    raw = unnormalize(x, mean, std)
    assert raw.shape == x.shape
    assert torch.allclose(raw[:, 0, :], mean)


def test_physical_consistency_loss_is_zero_without_thermo_targets() -> None:
    pred = torch.zeros(2, 4, 5, requires_grad=True)
    mask = torch.zeros_like(pred)
    mean = torch.zeros(5)
    std = torch.ones(5)
    loss = physical_consistency_loss(pred, mask, mean, std)
    assert loss.item() == 0.0


def test_physical_consistency_loss_returns_finite_value() -> None:
    pred = torch.zeros(2, 4, 5, requires_grad=True)
    mask = torch.zeros_like(pred)
    mask[:, :, 4] = 1.0
    mean = torch.tensor([-20.0, 70.0, 5.0, 900.0, 1.0])
    std = torch.tensor([10.0, 20.0, 5.0, 80.0, 0.5])
    loss = physical_consistency_loss(pred, mask, mean, std)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
