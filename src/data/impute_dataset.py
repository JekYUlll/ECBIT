"""Dataset utilities for ECBIT imputation windows."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


STATION_FEATURE_COLUMNS = (
    "lat_scaled",
    "lon_sin",
    "lon_cos",
    "elev_z",
    "record_years_z",
    "temperature_completeness",
    "pressure_completeness",
    "wind_speed_completeness",
    "relative_humidity_completeness",
)
STATION_FEATURE_DIM = len(STATION_FEATURE_COLUMNS)

RESIDUAL_FEATURE_COLUMNS = (
    "station_bias",
    "station_std",
    "station_mae",
    "station_log_count",
    "station_month_bias",
    "station_month_std",
    "station_month_mae",
    "station_month_log_count",
)
RESIDUAL_FEATURE_DIM = len(RESIDUAL_FEATURE_COLUMNS)


def _zscore(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    std = float(values.std(ddof=0))
    if not np.isfinite(std) or std < 1.0e-6:
        return values * 0.0
    return (values - float(values.mean())) / std


def _load_station_features(station_meta_csv: Path | None) -> dict[str, np.ndarray]:
    if station_meta_csv is None:
        station_meta_csv = Path("data/station_meta_ecbit.csv")
    if not station_meta_csv.exists():
        return {}

    meta = pd.read_csv(station_meta_csv).copy()
    required = {
        "station_id",
        "lat",
        "lon",
        "elev_m",
        "record_years",
        "temperature_completeness",
        "pressure_completeness",
        "wind_speed_completeness",
        "relative_humidity_completeness",
    }
    missing = required.difference(meta.columns)
    if missing:
        raise ValueError(f"station metadata CSV is missing columns: {sorted(missing)}")

    if "split" in meta.columns:
        selected = meta["split"].isin(["main", "heldout"])
        if selected.any():
            meta = meta[selected].copy()

    lon_rad = np.deg2rad(meta["lon"].astype(float))
    features = pd.DataFrame(
        {
            "station_id": meta["station_id"].astype(str),
            "lat_scaled": meta["lat"].astype(float) / 90.0,
            "lon_sin": np.sin(lon_rad),
            "lon_cos": np.cos(lon_rad),
            "elev_z": _zscore(meta["elev_m"]),
            "record_years_z": _zscore(meta["record_years"]),
            "temperature_completeness": meta["temperature_completeness"].astype(float),
            "pressure_completeness": meta["pressure_completeness"].astype(float),
            "wind_speed_completeness": meta["wind_speed_completeness"].astype(float),
            "relative_humidity_completeness": meta["relative_humidity_completeness"].astype(float),
        }
    )
    features = features.fillna(0.0)
    return {
        str(row["station_id"]): row[list(STATION_FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
        for _, row in features.iterrows()
    }


def _load_residual_features(residual_feature_csv: Path | None) -> dict[str, np.ndarray]:
    if residual_feature_csv is None:
        return {}
    if not residual_feature_csv.exists():
        raise FileNotFoundError(f"residual feature CSV not found: {residual_feature_csv}")

    table = pd.read_csv(residual_feature_csv).copy()
    required = {"station_id", "month", "variable_index", *RESIDUAL_FEATURE_COLUMNS}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"residual feature CSV is missing columns: {sorted(missing)}")

    features: dict[str, np.ndarray] = {}
    for station_id, group in table.groupby("station_id"):
        n_vars = int(group["variable_index"].max()) + 1
        grid = np.zeros((12, n_vars, RESIDUAL_FEATURE_DIM), dtype=np.float32)
        for _, row in group.iterrows():
            month_idx = int(row["month"]) - 1
            var_idx = int(row["variable_index"])
            if 0 <= month_idx < 12 and 0 <= var_idx < n_vars:
                grid[month_idx, var_idx] = row[list(RESIDUAL_FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
        features[str(station_id)] = grid
    return features


class ImputationWindowDataset(Dataset):
    """Load preprocessed station NPZ windows filtered by station group and split."""

    def __init__(
        self,
        manifest_csv: str | Path = "data/antaws_impute_manifest.csv",
        station_groups: Iterable[str] = ("main",),
        window_splits: Iterable[str] = ("train",),
        station_ids: Iterable[str] | None = None,
        window_subset_csv: str | Path | None = None,
        station_meta_csv: str | Path | None = None,
        residual_feature_csv: str | Path | None = None,
    ) -> None:
        self.manifest_csv = Path(manifest_csv)
        manifest = pd.read_csv(self.manifest_csv)
        station_features = _load_station_features(Path(station_meta_csv) if station_meta_csv is not None else None)
        residual_features = _load_residual_features(
            Path(residual_feature_csv) if residual_feature_csv is not None else None
        )
        groups = set(station_groups)
        splits = set(window_splits)
        subset_by_station: dict[str, set[int]] | None = None
        if window_subset_csv is not None:
            subset = pd.read_csv(window_subset_csv)
            required = {"station_id", "window_local_index"}
            missing = required.difference(subset.columns)
            if missing:
                raise ValueError(f"window subset CSV is missing columns: {sorted(missing)}")
            subset_by_station = {
                str(station): set(group["window_local_index"].astype(int).tolist())
                for station, group in subset.groupby("station_id")
            }
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
            if subset_by_station is not None:
                station_subset = subset_by_station.get(str(row["station_id"]), set())
                keep &= np.isin(np.arange(len(window_split)), list(station_subset))
            if not keep.any():
                continue
            original_indices = np.flatnonzero(keep).astype(np.int64)
            if "timestamps" in data.files:
                month = data["timestamps"][keep].astype("datetime64[M]").astype(int) % 12 + 1
            else:
                month = np.ones(data["X"][keep].shape[:2], dtype=np.int64)
            n_vars = int(data["X"].shape[-1])
            norm_mean = data["normalization_mean"].astype(np.float32) if "normalization_mean" in data.files else np.zeros(n_vars, dtype=np.float32)
            norm_std = data["normalization_std"].astype(np.float32) if "normalization_std" in data.files else np.ones(n_vars, dtype=np.float32)
            item = {
                "X": data["X"][keep].astype(np.float32),
                "obs_mask": data["obs_mask"][keep].astype(np.float32),
                "E_3h": data["E_3h"][keep].astype(np.float32),
                "T_enc": data["T_enc"][keep].astype(np.float32),
                "normalization_mean": norm_mean,
                "normalization_std": norm_std,
                "month": month,
                "station_features": station_features.get(
                    str(row["station_id"]),
                    np.zeros(STATION_FEATURE_DIM, dtype=np.float32),
                ),
                "residual_features_by_month": residual_features.get(
                    str(row["station_id"]),
                    np.zeros((12, n_vars, RESIDUAL_FEATURE_DIM), dtype=np.float32),
                ),
                "station_id": str(row["station_id"]),
                "station_group": str(row["station_group"]),
                "window_split": window_split[keep],
                "window_local_index": original_indices,
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
        month = item["month"][local_idx].astype(np.int64)
        month_index = np.clip(month, 1, 12) - 1
        residual_features = item["residual_features_by_month"][month_index].mean(axis=0)
        return {
            "x": torch.from_numpy(item["X"][local_idx]),
            "obs_mask": torch.from_numpy(item["obs_mask"][local_idx]),
            "era5": torch.from_numpy(item["E_3h"][local_idx]),
            "time_enc": torch.from_numpy(item["T_enc"][local_idx]),
            "norm_mean": torch.from_numpy(item["normalization_mean"]),
            "norm_std": torch.from_numpy(item["normalization_std"]),
            "month": torch.from_numpy(month),
            "station_features": torch.from_numpy(item["station_features"]),
            "residual_features": torch.from_numpy(residual_features.astype(np.float32)),
            "station_id": item["station_id"],
            "station_group": item["station_group"],
            "window_split": str(item["window_split"][local_idx]),
            "window_local_index": int(item["window_local_index"][local_idx]),
        }
