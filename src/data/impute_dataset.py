"""Dataset utilities for ECBIT imputation windows."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ImputationWindowDataset(Dataset):
    """Load preprocessed station NPZ windows filtered by station group and split."""

    def __init__(
        self,
        manifest_csv: str | Path = "data/antaws_impute_manifest.csv",
        station_groups: Iterable[str] = ("main",),
        window_splits: Iterable[str] = ("train",),
        station_ids: Iterable[str] | None = None,
    ) -> None:
        self.manifest_csv = Path(manifest_csv)
        manifest = pd.read_csv(self.manifest_csv)
        groups = set(station_groups)
        splits = set(window_splits)
        if station_ids is not None:
            station_ids = set(station_ids)
            manifest = manifest[manifest["station_id"].isin(station_ids)]
        manifest = manifest[manifest["station_group"].isin(groups)].copy()
        if manifest.empty:
            raise ValueError("No stations match dataset filters")

        arrays = []
        index_rows = []
        offset = 0
        for _, row in manifest.iterrows():
            path = Path(row["path"])
            data = np.load(path, allow_pickle=True)
            window_split = data["window_split"].astype(str)
            keep = np.isin(window_split, list(splits))
            if not keep.any():
                continue
            if "timestamps" in data.files:
                month = data["timestamps"][keep].astype("datetime64[M]").astype(int) % 12 + 1
            else:
                month = np.ones(data["X"][keep].shape[:2], dtype=np.int64)
            item = {
                "X": data["X"][keep].astype(np.float32),
                "obs_mask": data["obs_mask"][keep].astype(np.float32),
                "E_3h": data["E_3h"][keep].astype(np.float32),
                "T_enc": data["T_enc"][keep].astype(np.float32),
                "month": month,
                "station_id": str(row["station_id"]),
                "station_group": str(row["station_group"]),
                "window_split": window_split[keep],
            }
            arrays.append(item)
            n = item["X"].shape[0]
            index_rows.extend((len(arrays) - 1, i) for i in range(n))
            offset += n
        if not arrays:
            raise ValueError("No windows match dataset filters")
        self.arrays = arrays
        self.index_rows = index_rows
        self.n_windows = offset

    def __len__(self) -> int:
        return len(self.index_rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        array_idx, local_idx = self.index_rows[idx]
        item = self.arrays[array_idx]
        return {
            "x": torch.from_numpy(item["X"][local_idx]),
            "obs_mask": torch.from_numpy(item["obs_mask"][local_idx]),
            "era5": torch.from_numpy(item["E_3h"][local_idx]),
            "time_enc": torch.from_numpy(item["T_enc"][local_idx]),
            "month": torch.from_numpy(item["month"][local_idx].astype(np.int64)),
            "station_id": item["station_id"],
            "station_group": item["station_group"],
            "window_split": str(item["window_split"][local_idx]),
        }
