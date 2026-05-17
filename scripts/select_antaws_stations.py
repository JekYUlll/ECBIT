#!/usr/bin/env python3
"""Select AntAWS stations for ECBIT experiments.

The selector scans AntAWS 3-hourly CSV files, computes record length and
variable completeness, then emits a station metadata CSV with a main/held-out
split. It is intentionally read-only with respect to raw data.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_RAW_DIR = Path("/home/horeb/_code/microclimate_demo/data/AntAWS/3_hourly")
DEFAULT_OUTPUT = Path("data/station_meta_ecbit.csv")

ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "latin1")
META_FILE = "Main characteristics of automatic weather station.csv"

REQUIRED_COLUMNS = {
    "year": "Year",
    "month": "Month",
    "day": "Day",
    "hour": "Three-hourly observation time(UTC)",
    "temperature": "Temperature(℃)",
    "pressure": "Pressure(hPa)",
    "wind_speed": "Wind Speed(m/s)",
    "relative_humidity": "Relative Humidity(%)",
}


def read_csv_with_fallback(path: Path, **kwargs) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError(f"Could not read {path} with encodings {ENCODINGS}") from last_error


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def station_id(name: str) -> str:
    sid = re.sub(r"[^A-Za-z0-9]+", "_", name.strip()).strip("_").lower()
    return sid or "station"


def load_station_metadata(raw_dir: Path) -> pd.DataFrame:
    meta_path = raw_dir / META_FILE
    meta = read_csv_with_fallback(meta_path)
    meta = meta.loc[:, ~meta.columns.str.startswith("Unnamed")]
    meta = meta.rename(
        columns={
            "Station": "station",
            "Lat(℃)": "lat",
            "Lon(℃)": "lon",
            "Elevation(m)": "elev_m",
            "Institution": "institution",
        }
    )
    meta["station_key"] = meta["station"].map(normalize_name)
    return meta[["station_key", "station", "lat", "lon", "elev_m", "institution"]]


def station_name_from_file(path: Path) -> str:
    return path.stem.removesuffix("_3h")


def valid_datetime_frame(df: pd.DataFrame) -> pd.DataFrame:
    cols = REQUIRED_COLUMNS
    out = df.copy()
    for col in [cols["year"], cols["month"], cols["day"], cols["hour"]]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=[cols["year"], cols["month"], cols["day"], cols["hour"]])
    out[cols["hour"]] = out[cols["hour"]].astype(int).clip(0, 23)
    out["timestamp"] = pd.to_datetime(
        {
            "year": out[cols["year"]].astype(int),
            "month": out[cols["month"]].astype(int),
            "day": out[cols["day"]].astype(int),
            "hour": out[cols["hour"]].astype(int),
        },
        errors="coerce",
    )
    return out.dropna(subset=["timestamp"]).sort_values("timestamp")


def summarize_station(path: Path) -> dict[str, object] | None:
    station = station_name_from_file(path)
    try:
        df = read_csv_with_fallback(path)
    except Exception as exc:
        return {"station": station, "file": str(path), "read_error": str(exc)}

    missing = [col for col in REQUIRED_COLUMNS.values() if col not in df.columns]
    if missing:
        return {
            "station": station,
            "file": str(path),
            "read_error": f"missing columns: {missing}",
        }

    df = valid_datetime_frame(df)
    if df.empty:
        return {"station": station, "file": str(path), "read_error": "no valid timestamps"}

    value_cols = {
        "temperature": REQUIRED_COLUMNS["temperature"],
        "pressure": REQUIRED_COLUMNS["pressure"],
        "wind_speed": REQUIRED_COLUMNS["wind_speed"],
        "relative_humidity": REQUIRED_COLUMNS["relative_humidity"],
    }
    for col in value_cols.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")

    start = df["timestamp"].iloc[0]
    end = df["timestamp"].iloc[-1]
    expected_steps = int(((end - start).total_seconds() // (3 * 3600)) + 1)
    observed_steps = len(df)
    record_years = expected_steps * 3 / (24 * 365.25)
    temporal_coverage = observed_steps / expected_steps if expected_steps else np.nan

    stats: dict[str, object] = {
        "station": station,
        "station_id": station_id(station),
        "station_key": normalize_name(station),
        "file": str(path),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "record_years": record_years,
        "expected_steps": expected_steps,
        "observed_steps": observed_steps,
        "temporal_coverage": temporal_coverage,
        "read_error": "",
    }

    completeness_values = []
    for name, col in value_cols.items():
        completeness = float(df[col].notna().sum() / expected_steps)
        stats[f"{name}_completeness"] = completeness
        completeness_values.append(completeness)
    stats["core_mean_completeness"] = float(np.mean(completeness_values))
    stats["core_min_completeness"] = float(np.min(completeness_values))
    return stats


def choose_heldout(selected: pd.DataFrame, n_heldout: int) -> set[str]:
    """Greedy farthest-point held-out split in normalized geo space."""
    if len(selected) <= n_heldout:
        return set(selected["station"])

    features = selected[["lat", "lon", "elev_m"]].astype(float).to_numpy()
    col_std = np.nanstd(features, axis=0)
    col_std[col_std == 0] = 1.0
    x = (features - np.nanmean(features, axis=0)) / col_std
    x = np.nan_to_num(x)

    center = np.mean(x, axis=0, keepdims=True)
    first = int(np.argmax(np.linalg.norm(x - center, axis=1)))
    chosen = [first]
    while len(chosen) < n_heldout:
        dist_to_chosen = np.min(
            np.linalg.norm(x[:, None, :] - x[chosen][None, :, :], axis=2),
            axis=1,
        )
        dist_to_chosen[chosen] = -np.inf
        chosen.append(int(np.argmax(dist_to_chosen)))
    return set(selected.iloc[chosen]["station"])


def build_selection(
    raw_dir: Path,
    output: Path,
    min_years: float,
    min_t_completeness: float,
    min_core_mean: float,
    target_total: int,
    n_heldout: int,
) -> pd.DataFrame:
    meta = load_station_metadata(raw_dir)
    rows = []
    for path in sorted(raw_dir.glob("*_3h.csv")):
        rows.append(summarize_station(path))

    df = pd.DataFrame([row for row in rows if row is not None])
    df = df.merge(meta, on="station_key", how="left", suffixes=("", "_meta"))
    df["station"] = df["station_meta"].fillna(df["station"])
    df = df.drop(columns=[c for c in ["station_meta"] if c in df.columns])

    numeric_cols = [
        "record_years",
        "temperature_completeness",
        "pressure_completeness",
        "wind_speed_completeness",
        "relative_humidity_completeness",
        "core_mean_completeness",
        "core_min_completeness",
        "temporal_coverage",
        "lat",
        "lon",
        "elev_m",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    eligible = (
        df["read_error"].fillna("").eq("")
        & (df["record_years"] >= min_years)
        & (df["temperature_completeness"] >= min_t_completeness)
        & (df["core_mean_completeness"] >= min_core_mean)
        & df["lat"].notna()
        & df["lon"].notna()
    )
    df["eligible"] = eligible
    df["selection_score"] = (
        df["core_mean_completeness"].fillna(0) * 100
        + df["temperature_completeness"].fillna(0) * 25
        + np.minimum(df["record_years"].fillna(0), 20) * 2
        + df["temporal_coverage"].fillna(0) * 10
    )

    selected = (
        df.loc[df["eligible"]]
        .sort_values(
            ["selection_score", "record_years", "core_mean_completeness"],
            ascending=[False, False, False],
        )
        .head(target_total)
        .copy()
    )
    heldout = choose_heldout(selected, n_heldout)
    df["split"] = "excluded"
    df.loc[df["station"].isin(selected["station"]), "split"] = "main"
    df.loc[df["station"].isin(heldout), "split"] = "heldout"

    ordered_cols = [
        "station",
        "station_id",
        "split",
        "eligible",
        "selection_score",
        "lat",
        "lon",
        "elev_m",
        "institution",
        "start",
        "end",
        "record_years",
        "expected_steps",
        "observed_steps",
        "temporal_coverage",
        "temperature_completeness",
        "pressure_completeness",
        "wind_speed_completeness",
        "relative_humidity_completeness",
        "core_mean_completeness",
        "core_min_completeness",
        "file",
        "read_error",
    ]
    split_order = {"main": 0, "heldout": 1, "excluded": 2}
    df["_split_order"] = df["split"].map(split_order).fillna(99)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = df.sort_values(["_split_order", "selection_score"], ascending=[True, False])
    df = df.drop(columns=["_split_order"])
    df[ordered_cols].to_csv(output, index=False)
    return df[ordered_cols]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-years", type=float, default=10.0)
    parser.add_argument("--min-t-completeness", type=float, default=0.70)
    parser.add_argument("--min-core-mean", type=float, default=0.50)
    parser.add_argument("--target-total", type=int, default=40)
    parser.add_argument("--heldout", type=int, default=5)
    args = parser.parse_args()

    df = build_selection(
        raw_dir=args.raw_dir,
        output=args.output,
        min_years=args.min_years,
        min_t_completeness=args.min_t_completeness,
        min_core_mean=args.min_core_mean,
        target_total=args.target_total,
        n_heldout=args.heldout,
    )
    split_counts = df["split"].value_counts().to_dict()
    eligible_count = int(df["eligible"].sum())
    print(f"Wrote {args.output}")
    print(f"Total station files: {len(df)}")
    print(f"Eligible stations: {eligible_count}")
    print(f"Split counts: {split_counts}")
    print("\nSelected stations:")
    selected = df[df["split"].isin(["main", "heldout"])]
    for _, row in selected.sort_values(["split", "station"]).iterrows():
        print(
            f"  {row['split']:7s} {row['station']:<32s} "
            f"years={row['record_years']:.1f} "
            f"T={row['temperature_completeness']:.2f} "
            f"mean={row['core_mean_completeness']:.2f}"
        )


if __name__ == "__main__":
    main()
