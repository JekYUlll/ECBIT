from __future__ import annotations

import numpy as np
import pytest

from src.baselines.brits_wrapper import BRITSImputer
from src.baselines.pypots_common import to_pypots_dataset
from src.baselines.saits_wrapper import SAITSImputer


def test_to_pypots_dataset_restores_nan_missing_values() -> None:
    x = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
    obs = np.ones_like(x, dtype=np.float32)
    obs[0, 1, 0] = 0

    dataset = to_pypots_dataset(x, obs)

    assert set(dataset) == {"X"}
    assert dataset["X"].shape == (2, 3, 2)
    assert np.isnan(dataset["X"][0, 1, 0])
    assert dataset["X"][0, 0, 0] == x[0, 0, 0]


def test_to_pypots_dataset_validates_shape() -> None:
    with pytest.raises(ValueError):
        to_pypots_dataset(np.zeros((3, 2), dtype=np.float32))
    with pytest.raises(ValueError):
        to_pypots_dataset(np.zeros((1, 2, 3), dtype=np.float32), np.zeros((1, 2, 4), dtype=np.float32))


def test_wrappers_are_lazy_when_pypots_missing() -> None:
    saits = SAITSImputer(n_steps=168, n_features=5, n_layers=1)
    brits = BRITSImputer(n_steps=168, n_features=5)

    assert saits.model_name == "SAITS"
    assert brits.model_name == "BRITS"
    assert saits.model is None
    assert brits.model is None
