"""BRITS baseline wrapper via PyPOTS."""

from __future__ import annotations

from typing import Any

from src.baselines.pypots_common import PyPOTSWrapper


class BRITSImputer(PyPOTSWrapper):
    """Lazy PyPOTS BRITS wrapper for ECBIT window tensors."""

    def __init__(self, n_steps: int, n_features: int, **model_kwargs: Any) -> None:
        defaults = {
            "rnn_hidden_size": 128,
            "batch_size": 32,
            "epochs": 50,
            "patience": 10,
            "num_workers": 0,
            "verbose": False,
        }
        defaults.update(model_kwargs)
        super().__init__("BRITS", n_steps=n_steps, n_features=n_features, model_kwargs=defaults)
