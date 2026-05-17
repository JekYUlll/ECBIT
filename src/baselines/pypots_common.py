"""Shared helpers for PyPOTS imputation baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def to_pypots_dataset(x: np.ndarray, obs_mask: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Convert normalized ECBIT windows to PyPOTS `{X: ...}` format.

    PyPOTS uses NaNs to denote missing values. ECBIT stores filled model inputs
    plus a binary observation mask, so this helper restores NaNs before handing
    data to SAITS/BRITS.
    """
    if x.ndim != 3:
        raise ValueError(f"x must be (N, T, C), got {x.shape}")
    arr = x.astype(np.float32, copy=True)
    if obs_mask is not None:
        if obs_mask.shape != x.shape:
            raise ValueError(f"obs_mask shape {obs_mask.shape} != x shape {x.shape}")
        arr[obs_mask <= 0] = np.nan
    return {"X": arr}


def _import_pypots_model(model_name: str) -> type[Any]:
    try:
        module = __import__("pypots.imputation", fromlist=[model_name])
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyPOTS is required for SAITS/BRITS baselines. Install it in the "
            "remote training environment before submitting those jobs."
        ) from exc
    return getattr(module, model_name)


@dataclass
class PyPOTSWrapper:
    """Thin lazy wrapper around a PyPOTS imputation model."""

    model_name: str
    n_steps: int
    n_features: int
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    model: Any | None = None

    def build(self) -> Any:
        cls = _import_pypots_model(self.model_name)
        kwargs = {"n_steps": self.n_steps, "n_features": self.n_features, **self.model_kwargs}
        self.model = cls(**kwargs)
        return self.model

    def fit(self, x: np.ndarray, obs_mask: np.ndarray | None = None, **fit_kwargs: Any) -> "PyPOTSWrapper":
        model = self.model if self.model is not None else self.build()
        model.fit(to_pypots_dataset(x, obs_mask), **fit_kwargs)
        return self

    def impute(self, x: np.ndarray, obs_mask: np.ndarray | None = None) -> np.ndarray:
        model = self.model if self.model is not None else self.build()
        result = model.impute(to_pypots_dataset(x, obs_mask))
        if isinstance(result, dict):
            result = result.get("imputation", result.get("X", result))
        return np.asarray(result, dtype=np.float32)
