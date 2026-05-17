"""BRITS baseline wrapper via PyPOTS."""

from __future__ import annotations

from typing import Any

from src.baselines.pypots_common import PyPOTSWrapper


class BRITSImputer(PyPOTSWrapper):
    """Lazy PyPOTS BRITS wrapper for ECBIT window tensors."""

    def __init__(self, n_steps: int, n_features: int, **model_kwargs: Any) -> None:
        super().__init__("BRITS", n_steps=n_steps, n_features=n_features, model_kwargs=model_kwargs)
