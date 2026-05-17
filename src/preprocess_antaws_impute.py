#!/usr/bin/env python3
"""Preprocess selected AntAWS stations for sparse block-imputation experiments.

The original plan assumed fully observed training windows. The selected AntAWS
stations are too sparse for that at 168 steps, so this pipeline preserves sparse
windows and writes explicit observation masks. Artificial block masks should be
sampled later only from positions where `obs_mask == 1`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


META_CSV = Path("data/station_meta_ecbit.csv")
ERA5_DIR = Path("data/era5_3h")
OUTPUT_DIR = Path("data/antaws/processed")
MANIFEST_CSV = Path("data/antaws_impute_manifest.csv")
SCALER_NPZ = Path("data/antaws_impute_scaler.npz")

ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "latin1")
TIME_COLS = {
    "year": "Year",
    "month": "Month",
    "day": "Day",
    "hour": "Three-hourly observation time(UTC)",
}
VALUE_COLS = {
    "T": "Temperature(℃)",
    "RH": "Relative Humidity(%)",
    "wspd": "Wind Speed(m/s)",
    "P": "Pressure(hPa)",
}
VARIABLES = np.asarray(["T", "RH", "wspd", "P", "q"])
TIME_FEATURES = np.asarray(["sin_doy", "cos_doy", "sin_hour", "cos_hour"])


def read_csv_with_fallback(path: Path, **kwargs) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError(f"Could not read {path} with encodings {ENCODINGS}") from last_error


def make_timestamps(df: pd.DataFrame) -> pd.DatetimeIndex:
    for col in TIME_COLS.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")
    ts = pd.to_datetime(
        {
            "year": df[TIME_COLS["year"]].astype("Int64"),
            "month": df[TIME_COLS["month"]].astype("Int64"),
            "day": df[TIME_COLS["day"]].astype("Int64"),
            "hour": df[TIME_COLS["hour"]].astype("Int64").clip(0, 23),
        },
        errors="coerce",
    )
    if ts.isna().any():
        raise ValueError("AntAWS station file contains invalid timestamps")
    return pd.DatetimeIndex(ts)


def specific_humidity_gkg(t_c: np.ndarray, rh_pct: np.ndarray, p_hpa: np.ndarray) -> np.ndarray:
    """Derive specific humidity in g/kg from temperature, RH, and pressure."""
    es = 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))
    e = np.clip(rh_pct, 0.0, 100.0) / 100.0 * es
    q_kgkg = 0.622 * e / (p_hpa - 0.378 * e)
    return q_kgkg * 1000.0


def load_antaws_station(csv_path: Path) -> tuple[pd.DatetimeIndex, np.ndarray]:
    usecols = list(TIME_COLS.values()) + list(VALUE_COLS.values())
    df = read_csv_with_fallback(csv_path, usecols=usecols)
    timestamps = make_timestamps(df)

    values = np.full((len(df), len(VARIABLES)), np.nan, dtype=np.float32)
    for i, var in enumerate(VARIABLES[:4]):
        values[:, i] = pd.to_numeric(df[VALUE_COLS[str(var)]], errors="coerce").to_numpy(np.float32)
    values[:, 4] = specific_humidity_gkg(values[:, 0], values[:, 1], values[:, 3]).astype(np.float32)
    return timestamps, values


def load_era5_station(era5_dir: Path, station_id: str) -> tuple[np.ndarray, np.ndarray]:
    path = era5_dir / f"{station_id}_era5_3h.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing ERA5 3h file: {path}")
    data = np.load(path, allow_pickle=True)
    return data["timestamps"].astype("datetime64[ns]"), data["era5_met"].astype(np.float32)


def time_encoding(timestamps: pd.DatetimeIndex) -> np.ndarray:
    doy = timestamps.dayofyear.to_numpy(dtype=np.float32)
    hour = timestamps.hour.to_numpy(dtype=np.float32)
    return np.stack(
        [
            np.sin(2.0 * np.pi * (doy - 1.0) / 366.0),
            np.cos(2.0 * np.pi * (doy - 1.0) / 366.0),
            np.sin(2.0 * np.pi * hour / 24.0),
            np.cos(2.0 * np.pi * hour / 24.0),
        ],
        axis=1,
    ).astype(np.float32)


def compute_scaler(meta: pd.DataFrame, train_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Fit global variable stats on main-station temporal train segments only."""
    chunks = []
    for _, row in meta[meta["split"].eq("main")].iterrows():
        _, values = load_antaws_station(Path(row["file"]))
        n_train = int(len(values) * train_fraction)
        chunks.append(values[:n_train])
    all_train = np.concatenate(chunks, axis=0)
    mean = np.nanmean(all_train, axis=0).astype(np.float32)
    std = np.nanstd(all_train, axis=0).astype(np.float32)
    std = np.where(np.isfinite(std) & (std > 1.0e-6), std, 1.0).astype(np.float32)
    if not np.isfinite(mean).all():
        raise ValueError(f"Non-finite scaler mean: {mean}")
    return mean, std


def station_windows(
    values: np.ndarray,
    era5: np.ndarray,
    t_enc: np.ndarray,
    timestamps: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    seq_len: int,
    stride: int,
    train_fraction: float,
    val_fraction: float,
    min_obs_fraction: float,
    min_obs_steps_per_var: int,
    min_targetable_vars: int,
) -> dict[str, np.ndarray]:
    starts = np.arange(0, len(values) - seq_len + 1, stride, dtype=np.int64)
    kept: list[int] = []
    obs_mask = np.isfinite(values)
    for start in starts:
        stop = start + seq_len
        win_mask = obs_mask[start:stop]
        per_var_obs = win_mask.sum(axis=0)
        targetable_vars = int((per_var_obs >= min_obs_steps_per_var).sum())
        if float(win_mask.mean()) >= min_obs_fraction and targetable_vars >= min_targetable_vars:
            kept.append(int(start))

    if not kept:
        raise ValueError("No windows retained; relax preprocessing thresholds")

    n = len(kept)
    n_train = int(n * train_fraction)
    n_val = int(n * val_fraction)
    n_train = min(max(n_train, 1), max(n - 2, 1))
    n_val = min(max(n_val, 1), max(n - n_train - 1, 1))
    splits = np.asarray(["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val))

    x = np.empty((n, seq_len, len(VARIABLES)), dtype=np.float32)
    m = np.empty_like(x)
    e = np.empty_like(x)
    te = np.empty((n, seq_len, len(TIME_FEATURES)), dtype=np.float32)
    window_times = np.empty((n, seq_len), dtype="datetime64[ns]")
    start_times = np.empty(n, dtype="datetime64[ns]")

    for i, start in enumerate(kept):
        stop = start + seq_len
        raw = values[start:stop]
        raw_mask = np.isfinite(raw).astype(np.float32)
        norm = (raw - mean) / std
        norm = np.where(np.isfinite(norm), norm, 0.0)

        era_norm = (era5[start:stop] - mean) / std
        if not np.isfinite(era_norm).all():
            raise ValueError("Non-finite normalized ERA5 window")

        x[i] = norm.astype(np.float32)
        m[i] = raw_mask
        e[i] = era_norm.astype(np.float32)
        te[i] = t_enc[start:stop]
        window_times[i] = timestamps[start:stop]
        start_times[i] = timestamps[start]

    return {
        "X": x,
        "obs_mask": m,
        "E_3h": e,
        "T_enc": te,
        "timestamps": window_times,
        "window_start": start_times,
        "window_split": splits,
        "window_start_index": np.asarray(kept, dtype=np.int64),
    }


def process_station(
    row: pd.Series,
    era5_dir: Path,
    output_dir: Path,
    mean: np.ndarray,
    std: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, object]:
    station = str(row["station"])
    station_id = str(row["station_id"])
    timestamps, values = load_antaws_station(Path(row["file"]))
    era5_times, era5 = load_era5_station(era5_dir, station_id)
    aws_times = timestamps.values.astype("datetime64[ns]")
    if len(era5) != len(values) or not np.array_equal(aws_times, era5_times):
        raise ValueError(f"{station_id}: ERA5 timestamps do not match AntAWS")

    windows = station_windows(
        values=values,
        era5=era5,
        t_enc=time_encoding(timestamps),
        timestamps=aws_times,
        mean=mean,
        std=std,
        seq_len=args.seq_len,
        stride=args.stride,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        min_obs_fraction=args.min_obs_fraction,
        min_obs_steps_per_var=args.min_obs_steps_per_var,
        min_targetable_vars=args.min_targetable_vars,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{station_id}_impute.npz"
    np.savez_compressed(
        out_path,
        **windows,
        station=station,
        station_id=station_id,
        station_group=str(row["split"]),
        variables=VARIABLES,
        time_features=TIME_FEATURES,
        normalization_mean=mean,
        normalization_std=std,
        seq_len=args.seq_len,
        stride=args.stride,
        min_obs_fraction=args.min_obs_fraction,
        min_obs_steps_per_var=args.min_obs_steps_per_var,
        min_targetable_vars=args.min_targetable_vars,
    )

    split_counts = pd.Series(windows["window_split"]).value_counts().to_dict()
    obs = windows["obs_mask"]
    return {
        "station": station,
        "station_id": station_id,
        "station_group": str(row["split"]),
        "path": str(out_path),
        "n_windows": int(obs.shape[0]),
        "train_windows": int(split_counts.get("train", 0)),
        "val_windows": int(split_counts.get("val", 0)),
        "test_windows": int(split_counts.get("test", 0)),
        "obs_fraction": float(obs.mean()),
        **{f"obs_fraction_{var}": float(obs[:, :, i].mean()) for i, var in enumerate(VARIABLES)},
    }


def preprocess(args: argparse.Namespace) -> pd.DataFrame:
    meta = pd.read_csv(args.meta_csv)
    selected = meta[meta["split"].isin(["main", "heldout"])].copy()
    if selected.empty:
        raise ValueError(f"No selected stations in {args.meta_csv}")
    if args.stride is None:
        args.stride = max(args.seq_len // 4, 1)

    mean, std = compute_scaler(selected, args.train_fraction)
    args.scaler_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.scaler_npz, mean=mean, std=std, variables=VARIABLES)

    rows = []
    for _, row in selected.iterrows():
        summary = process_station(row, args.era5_dir, args.output_dir, mean, std, args)
        rows.append(summary)
        print(
            f"{summary['station_id']}: windows={summary['n_windows']} "
            f"obs={summary['obs_fraction']:.3f} "
            f"train/val/test={summary['train_windows']}/"
            f"{summary['val_windows']}/{summary['test_windows']}"
        )

    manifest = pd.DataFrame(rows).sort_values(["station_group", "station_id"])
    args.manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.manifest_csv, index=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-csv", type=Path, default=META_CSV)
    parser.add_argument("--era5-dir", type=Path, default=ERA5_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--manifest-csv", type=Path, default=MANIFEST_CSV)
    parser.add_argument("--scaler-npz", type=Path, default=SCALER_NPZ)
    parser.add_argument("--seq-len", type=int, default=168)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--min-obs-fraction", type=float, default=0.25)
    parser.add_argument("--min-obs-steps-per-var", type=int, default=12)
    parser.add_argument("--min-targetable-vars", type=int, default=3)
    args = parser.parse_args()

    manifest = preprocess(args)
    print(f"Wrote {args.manifest_csv} ({len(manifest)} stations)")
    print(
        manifest.groupby("station_group")
        .agg(stations=("station_id", "count"), windows=("n_windows", "sum"), obs_fraction=("obs_fraction", "mean"))
        .to_string()
    )


if __name__ == "__main__":
    main()
