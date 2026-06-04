from __future__ import annotations

import torch
import pytest

from scripts.analyze_conformal_uncertainty import conformal_quantile


def test_conformal_quantile_uses_finite_sample_rank() -> None:
    values = torch.tensor([0.1, 0.2, 0.3, 0.4])
    assert conformal_quantile(values, alpha=0.2) == pytest.approx(0.4)


def test_conformal_quantile_ignores_nonfinite_values() -> None:
    values = torch.tensor([0.1, float("nan"), 0.2])
    assert conformal_quantile(values, alpha=0.5) == pytest.approx(0.2)
