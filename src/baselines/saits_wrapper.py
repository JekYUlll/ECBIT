"""SAITS baseline wrapper via PyPOTS."""

from __future__ import annotations

from typing import Any

from src.baselines.pypots_common import PyPOTSWrapper


class SAITSImputer(PyPOTSWrapper):
    """Lazy PyPOTS SAITS wrapper for ECBIT window tensors."""

    def __init__(self, n_steps: int, n_features: int, **model_kwargs: Any) -> None:
        defaults = {
            "n_layers": 2,
            "d_model": 128,
            "n_heads": 4,
            "d_k": 32,
            "d_v": 32,
            "d_ffn": 256,
            "dropout": 0.1,
            "attn_dropout": 0.1,
            "batch_size": 32,
            "epochs": 50,
            "patience": 10,
            "num_workers": 0,
            "verbose": False,
        }
        defaults.update(model_kwargs)
        super().__init__("SAITS", n_steps=n_steps, n_features=n_features, model_kwargs=defaults)
