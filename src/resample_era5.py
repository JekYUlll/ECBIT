#!/usr/bin/env python3
"""Download/align ERA5 point time series for selected AntAWS stations.

The output cadence is AntAWS 3-hourly timestamps. Despite the filename, this
script handles both stages needed by ECBIT:
1. ensure a station-level ERA5 NetCDF time-series file exists;
2. interpolate ERA5 to AntAWS 3h timestamps and save a compact `.npz`.
"""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


ANTAWS_RAW_DIR = Path("/home/horeb/_code/microclimate_demo/data/AntAWS/3_hourly")
META_CSV = Path("data/station_meta_ecbit.csv")
ERA5_RAW_DIR = Path("data/era5")
OUTPUT_DIR = Path("data/era5_3h")

ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "latin1")
ERA5_VARS = [
    "2m_temperature",
    "surface_pressure",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]
TIME_COLS = {
    "year": "Year",
    "month": "Month",
    "day": "Day",
    "hour": "Three-hourly observation time(UTC)",
}


def read_csv_with_fallback(path: Path, **kwargs) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError(f"Could not read {path} with encodings {ENCODINGS}") from last_error


def safe_station_id(name: str) -> str:
    sid = re.sub(r"[^A-Za-z0-9]+", "_", name.strip()).strip("_").lower()
    return sid or "station"


def has_required_era5_vars(nc_path: Path) -> bool:
    try:
        ds = xr.open_dataset(nc_path)
        names = set(ds.data_vars)
        ds.close()
    except Exception:
        return False
    basic = {"t2m", "sp", "u10", "v10"} <= names
    basic |= {
        "2m_temperature",
        "surface_pressure",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
    } <= names
    humidity = bool({"d2m", "2m_dewpoint_temperature", "q", "specific_humidity"} & names)
    return basic and humidity


def normalize_cds_download(nc_path: Path) -> None:
    """CDS sometimes returns a zip at the requested .nc path."""
    try:
        is_zip = zipfile.is_zipfile(nc_path)
    except FileNotFoundError:
        return
    if not is_zip:
        return

    extract_dir = nc_path.with_suffix(".extract")
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(nc_path) as zf:
        zf.extractall(extract_dir)
    candidates = sorted(extract_dir.rglob("*.nc"))
    if not candidates:
        raise FileNotFoundError(f"CDS zip contained no NetCDF file: {nc_path}")
    tmp_path = nc_path.with_suffix(".tmp.nc")
    candidates[0].replace(tmp_path)
    tmp_path.replace(nc_path)
    for p in sorted(extract_dir.rglob("*"), reverse=True):
        if p.is_file():
            p.unlink()
        elif p.is_dir():
            p.rmdir()
    extract_dir.rmdir()


def antaws_timestamps(csv_path: Path) -> pd.DatetimeIndex:
    df = read_csv_with_fallback(csv_path, usecols=list(TIME_COLS.values()))
    for col in TIME_COLS.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()
    ts = pd.to_datetime(
        {
            "year": df[TIME_COLS["year"]].astype(int),
            "month": df[TIME_COLS["month"]].astype(int),
            "day": df[TIME_COLS["day"]].astype(int),
            "hour": df[TIME_COLS["hour"]].astype(int).clip(0, 23),
        },
        errors="coerce",
    ).dropna()
    if ts.empty:
        raise ValueError(f"No valid AntAWS timestamps in {csv_path}")
    return pd.DatetimeIndex(ts).sort_values()


def download_era5_timeseries(
    station_id: str,
    lat: float,
    lon: float,
    start: str,
    end: str,
    era5_raw_dir: Path,
) -> Path:
    import cdsapi

    station_dir = era5_raw_dir / station_id
    station_dir.mkdir(parents=True, exist_ok=True)
    nc_path = station_dir / "timeseries.nc"
    if nc_path.exists() and nc_path.stat().st_size > 0 and has_required_era5_vars(nc_path):
        return nc_path
    if nc_path.exists():
        nc_path.unlink()

    client = cdsapi.Client()
    start_date = pd.to_datetime(start).strftime("%Y-%m-%d")
    end_date = pd.to_datetime(end).strftime("%Y-%m-%d")
    client.retrieve(
        "reanalysis-era5-single-levels-timeseries",
        {
            "variable": ERA5_VARS,
            "location": {"latitude": float(lat), "longitude": float(lon)},
            "date": [f"{start_date}/{end_date}"],
            "data_format": "netcdf",
        },
        str(nc_path),
    )
    normalize_cds_download(nc_path)
    if not nc_path.exists() or nc_path.stat().st_size == 0:
        raise FileNotFoundError(f"ERA5 download did not create {nc_path}")
    return nc_path


def open_era5(nc_path: Path) -> xr.Dataset:
    ds = xr.open_dataset(nc_path)
    if "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    if "expver" in ds.dims:
        ds = ds.isel(expver=0, drop=True)
    if "latitude" in ds.dims or "lat" in ds.dims:
        lat_dim = "latitude" if "latitude" in ds.dims else "lat"
        lon_dim = "longitude" if "longitude" in ds.dims else "lon"
        ds = ds.mean(dim=[lat_dim, lon_dim])
    return ds.sortby("time")


def era5_to_met(ds: xr.Dataset, timestamps: pd.DatetimeIndex) -> np.ndarray:
    target = timestamps.values.astype("datetime64[ns]")
    ds_interp = ds.interp(time=target, method="linear")

    t_name = "t2m" if "t2m" in ds_interp else "2m_temperature"
    p_name = "sp" if "sp" in ds_interp else "surface_pressure"
    u_name = "u10" if "u10" in ds_interp else "10m_u_component_of_wind"
    v_name = "v10" if "v10" in ds_interp else "10m_v_component_of_wind"

    t_c = ds_interp[t_name].values - 273.15
    p_hpa = ds_interp[p_name].values / 100.0
    u10 = ds_interp[u_name].values
    v10 = ds_interp[v_name].values
    wspd = np.sqrt(u10**2 + v10**2)

    if "d2m" in ds_interp:
        td_c = ds_interp["d2m"].values - 273.15
    elif "2m_dewpoint_temperature" in ds_interp:
        td_c = ds_interp["2m_dewpoint_temperature"].values - 273.15
    else:
        td_c = None

    es = 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))
    if td_c is None:
        q_source = "q" if "q" in ds_interp else "specific_humidity"
        q_kgkg = ds_interp[q_source].values
        e = q_kgkg * p_hpa / (0.622 + 0.378 * q_kgkg)
        q_gkg = q_kgkg * 1000.0
    else:
        e = 6.112 * np.exp(17.67 * td_c / (td_c + 243.5))
        q_kgkg = 0.622 * e / (p_hpa - 0.378 * e)
        q_gkg = q_kgkg * 1000.0
    rh = np.clip((e / es) * 100.0, 0, 100)

    era5_met = np.stack([t_c, rh, wspd, p_hpa, q_gkg], axis=1).astype(np.float32)
    if not np.isfinite(era5_met).all():
        raise ValueError(f"Non-finite ERA5 values after interpolation: {np.isnan(era5_met).sum()} NaNs")
    return era5_met


def process_station(row: pd.Series, raw_dir: Path, era5_raw_dir: Path, output_dir: Path, dry_run: bool) -> Path | None:
    station = str(row["station"])
    sid = str(row.get("station_id") or safe_station_id(station))
    csv_path = Path(row["file"]) if "file" in row and pd.notna(row["file"]) else raw_dir / f"{station}_3h.csv"
    out_path = output_dir / f"{sid}_era5_3h.npz"

    timestamps = antaws_timestamps(csv_path)
    if dry_run:
        print(f"[DRY] {station} ({sid}): {timestamps[0]} -> {timestamps[-1]}, n={len(timestamps)}")
        return None
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"{station}: exists, skipping {out_path}")
        return out_path

    nc_path = download_era5_timeseries(
        sid,
        float(row["lat"]),
        float(row["lon"]),
        str(timestamps[0]),
        str(timestamps[-1]),
        era5_raw_dir,
    )
    ds = open_era5(nc_path)
    try:
        era5_met = era5_to_met(ds, timestamps)
    finally:
        ds.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        era5_met=era5_met,
        timestamps=timestamps.values.astype("datetime64[ns]"),
        station=station,
        station_id=sid,
        lat=float(row["lat"]),
        lon=float(row["lon"]),
        source_file=str(nc_path),
        variables=np.asarray(["T", "RH", "wspd", "P", "q"]),
    )
    print(f"{station}: saved {out_path} shape={era5_met.shape}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-csv", type=Path, default=META_CSV)
    parser.add_argument("--raw-dir", type=Path, default=ANTAWS_RAW_DIR)
    parser.add_argument("--era5-raw-dir", type=Path, default=ERA5_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--stations", type=str, default=None, help="Comma-separated station names or station_ids")
    parser.add_argument("--splits", type=str, default="main,heldout")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    meta = pd.read_csv(args.meta_csv)
    wanted_splits = {s.strip() for s in args.splits.split(",") if s.strip()}
    meta = meta[meta["split"].isin(wanted_splits)].copy()
    if args.stations:
        wanted = {s.strip() for s in args.stations.split(",") if s.strip()}
        meta = meta[meta["station"].isin(wanted) | meta["station_id"].isin(wanted)]
    if meta.empty:
        raise ValueError("No stations selected for ERA5 processing")

    print(f"Processing {len(meta)} stations")
    ok = 0
    for _, row in meta.iterrows():
        try:
            result = process_station(row, args.raw_dir, args.era5_raw_dir, args.output_dir, args.dry_run)
            ok += int(result is not None or args.dry_run)
        except Exception as exc:
            print(f"ERROR {row['station']}: {exc}")
    print(f"Done: {ok}/{len(meta)} station(s)")


if __name__ == "__main__":
    main()
