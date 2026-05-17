from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.data.impute_dataset import ImputationWindowDataset
from src.metrics import masked_mae_rmse
from src.train_impute import artificial_mask_batch, build_model


def make_tiny_dataset(tmp_path):
    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    path = data_dir / "tiny_impute.npz"
    n, t, c = 4, 8, 3
    np.savez_compressed(
        path,
        X=np.ones((n, t, c), dtype=np.float32),
        obs_mask=np.ones((n, t, c), dtype=np.float32),
        E_3h=np.ones((n, t, c), dtype=np.float32) * 2,
        T_enc=np.zeros((n, t, 4), dtype=np.float32),
        window_split=np.asarray(["train", "train", "val", "test"]),
    )
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "station": "Tiny",
                "station_id": "tiny",
                "station_group": "main",
                "path": str(path),
            }
        ]
    ).to_csv(manifest, index=False)
    return manifest


def test_imputation_window_dataset_filters_split(tmp_path) -> None:
    manifest = make_tiny_dataset(tmp_path)
    ds = ImputationWindowDataset(manifest, station_groups=["main"], window_splits=["train"])

    assert len(ds) == 2
    item = ds[0]
    assert item["x"].shape == (8, 3)
    assert item["era5"].shape == (8, 3)
    assert item["time_enc"].shape == (8, 4)


def test_artificial_mask_batch_respects_obs_mask() -> None:
    obs = torch.ones(2, 24, 3)
    obs[:, 5:10, 1] = 0
    mask = artificial_mask_batch(obs, {"pattern": "short", "rate": 0.5, "min_block": 3, "max_block": 5}, seed=9)

    assert mask.shape == obs.shape
    assert torch.all(mask[obs == 0] == 0)
    assert mask.sum() > 0


def test_metrics_masked_mae_rmse() -> None:
    pred = torch.tensor([[[2.0, 10.0], [4.0, 20.0]]])
    target = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    mask = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    metrics = masked_mae_rmse(pred, target, mask)

    assert metrics["mae_T"] == 2.0
    assert round(metrics["rmse_T"], 6) == round(((1.0 + 9.0) / 2) ** 0.5, 6)


def test_build_model_from_config() -> None:
    model = build_model(
        {
            "model": {
                "name": "ecbit",
                "seq_len": 8,
                "n_vars": 3,
                "n_time": 4,
                "d_model": 16,
                "n_heads": 4,
                "n_layers": 1,
                "d_ff": 32,
                "dropout": 0.0,
            }
        }
    )
    assert model.seq_len == 8
