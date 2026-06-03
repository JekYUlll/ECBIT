#!/usr/bin/env python3
"""Local feasibility probe for station-adaptive ERA5 bias correction.

This is a stateless analysis. It does not train ECBIT and does not evaluate
block-imputation accuracy. It asks a narrower question: do held-out stations
contain enough train-period ERA5-AWS residual structure to justify a learned
station-adaptive correction layer?
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.preprocess_antaws_impute import (  # noqa: E402
    VARIABLES,
    load_antaws_station,
    load_era5_station,
    time_encoding,
)

OUT_DIR = Path("explorations/2026-06-03-sabc/results")
META_CSV = Path("data/station_meta_ecbit.csv")
SCALER_NPZ = Path("data/antaws_impute_scaler.npz")
ERA5_DIR = Path("data/era5_3h")


@dataclass
class StationSeries:
    station_id: str
    station: str
    group: str
    timestamps: pd.DatetimeIndex
    month: np.ndarray
    aws_norm: np.ndarray
    era5_norm: np.ndarray
    obs_mask: np.ndarray
    time_features: np.ndarray
    train_idx: np.ndarray
    test_idx: np.ndarray


def split_indices(n: int, train_fraction: float = 0.7, val_fraction: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    n_train = int(n * train_fraction)
    n_val = int(n * val_fraction)
    train = np.arange(0, n_train, dtype=np.int64)
    test = np.arange(n_train + n_val, n, dtype=np.int64)
    return train, test


def load_station_series(row: pd.Series, mean: np.ndarray, std: np.ndarray) -> StationSeries:
    station_id = str(row["station_id"])
    timestamps, aws = load_antaws_station(Path(row["file"]))
    era5_times, era5 = load_era5_station(ERA5_DIR, station_id)
    aws_times = timestamps.values.astype("datetime64[ns]")
    if len(aws) != len(era5) or not np.array_equal(aws_times, era5_times):
        raise ValueError(f"{station_id}: ERA5 timestamps do not match AWS timestamps")

    aws_norm = (aws - mean) / std
    era5_norm = (era5 - mean) / std
    obs_mask = np.isfinite(aws_norm)
    train_idx, test_idx = split_indices(len(aws_norm))
    return StationSeries(
        station_id=station_id,
        station=str(row["station"]),
        group=str(row["split"]),
        timestamps=timestamps,
        month=timestamps.month.to_numpy(dtype=np.int16),
        aws_norm=aws_norm.astype(np.float32),
        era5_norm=era5_norm.astype(np.float32),
        obs_mask=obs_mask,
        time_features=time_encoding(timestamps).astype(np.float32),
        train_idx=train_idx,
        test_idx=test_idx,
    )


def residual(series: StationSeries) -> np.ndarray:
    return series.aws_norm - series.era5_norm


def fit_global_mean(main_series: list[StationSeries]) -> np.ndarray:
    chunks = []
    masks = []
    for s in main_series:
        idx = s.train_idx
        chunks.append(residual(s)[idx])
        masks.append(s.obs_mask[idx])
    r = np.concatenate(chunks, axis=0)
    m = np.concatenate(masks, axis=0)
    out = np.zeros(len(VARIABLES), dtype=np.float32)
    for c in range(len(VARIABLES)):
        vals = r[:, c][m[:, c]]
        out[c] = float(np.nanmean(vals)) if len(vals) else 0.0
    return out


def fit_target_mean(series: StationSeries) -> np.ndarray:
    r = residual(series)[series.train_idx]
    m = series.obs_mask[series.train_idx]
    out = np.zeros(len(VARIABLES), dtype=np.float32)
    for c in range(len(VARIABLES)):
        vals = r[:, c][m[:, c]]
        out[c] = float(np.nanmean(vals)) if len(vals) else 0.0
    return out


def fit_target_month(series: StationSeries, fallback: np.ndarray) -> np.ndarray:
    r = residual(series)[series.train_idx]
    m = series.obs_mask[series.train_idx]
    months = series.month[series.train_idx]
    table = np.tile(fallback[None, :], (13, 1)).astype(np.float32)
    for month in range(1, 13):
        month_mask = months == month
        for c in range(len(VARIABLES)):
            vals = r[month_mask, c][m[month_mask, c]]
            if len(vals) >= 24:
                table[month, c] = float(np.nanmean(vals))
    return table


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float = 1.0e-3) -> np.ndarray:
    x_aug = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float32), x], axis=1)
    eye = np.eye(x_aug.shape[1], dtype=np.float32)
    eye[0, 0] = 0.0
    return np.linalg.solve(x_aug.T @ x_aug + alpha * eye, x_aug.T @ y).astype(np.float32)


def fit_state_linear(series: StationSeries) -> list[np.ndarray]:
    coefs: list[np.ndarray] = []
    r_all = residual(series)
    for c in range(len(VARIABLES)):
        idx = series.train_idx
        mask = series.obs_mask[idx, c] & np.isfinite(series.era5_norm[idx, c])
        x = np.column_stack([series.era5_norm[idx, c], series.time_features[idx]])[mask]
        y = r_all[idx, c][mask]
        if len(y) < 128:
            coefs.append(np.zeros(6, dtype=np.float32))
        else:
            coefs.append(ridge_fit(x.astype(np.float32), y.astype(np.float32)))
    return coefs


def predict_state_linear(series: StationSeries, coefs: list[np.ndarray], idx: np.ndarray) -> np.ndarray:
    pred = np.zeros((len(idx), len(VARIABLES)), dtype=np.float32)
    for c, beta in enumerate(coefs):
        x = np.column_stack([series.era5_norm[idx, c], series.time_features[idx]]).astype(np.float32)
        x_aug = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float32), x], axis=1)
        pred[:, c] = x_aug @ beta
    return pred


def evaluate(series: StationSeries, mode: str, correction: np.ndarray | list[np.ndarray]) -> list[dict[str, object]]:
    idx = series.test_idx
    target = series.aws_norm[idx]
    base = series.era5_norm[idx]
    mask = series.obs_mask[idx]
    if mode in {"none", "global_mean", "target_mean"}:
        corr = np.asarray(correction, dtype=np.float32)[None, :]
    elif mode == "target_month":
        table = np.asarray(correction, dtype=np.float32)
        corr = table[series.month[idx]]
    elif mode == "target_state_linear":
        corr = predict_state_linear(series, correction, idx)  # type: ignore[arg-type]
    else:
        raise ValueError(mode)
    pred = base + corr
    err = np.abs(pred - target)
    rows: list[dict[str, object]] = []
    for c, var in enumerate(VARIABLES):
        vals = err[:, c][mask[:, c]]
        rows.append(
            {
                "station_id": series.station_id,
                "station": series.station,
                "mode": mode,
                "variable": str(var),
                "mae": float(np.nanmean(vals)) if len(vals) else np.nan,
                "n_obs": int(len(vals)),
            }
        )
    vals_all = err[mask]
    rows.append(
        {
            "station_id": series.station_id,
            "station": series.station,
            "mode": mode,
            "variable": "mean",
            "mae": float(np.nanmean(vals_all)) if len(vals_all) else np.nan,
            "n_obs": int(len(vals_all)),
        }
    )
    return rows


def improvement_summary(results: pd.DataFrame) -> pd.DataFrame:
    mean_rows = results[results["variable"].eq("mean")].copy()
    wide = mean_rows.pivot(index=["station_id", "station"], columns="mode", values="mae").reset_index()
    baseline = wide["none"]
    for mode in ["global_mean", "target_mean", "target_month", "target_state_linear"]:
        wide[f"{mode}_gain_vs_none"] = baseline - wide[mode]
        wide[f"{mode}_pct_gain_vs_none"] = 100.0 * (baseline - wide[mode]) / baseline
    if "global_mean" in wide:
        for mode in ["target_mean", "target_month", "target_state_linear"]:
            wide[f"{mode}_gain_vs_global"] = wide["global_mean"] - wide[mode]
    return wide


def markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(str(c) for c in cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for col in cols:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                cells.append(format(float(value), floatfmt) if np.isfinite(value) else "nan")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(station_summary: pd.DataFrame, variable_results: pd.DataFrame, out_path: Path) -> None:
    best = station_summary.sort_values("target_state_linear_gain_vs_global", ascending=False)
    lines = [
        "# SABC Feasibility Probe",
        "",
        "This local analysis evaluates whether held-out stations contain station-specific ERA5-AWS residual structure that can be reduced by simple train-period adaptation.",
        "",
        "## Station-Level Summary",
        "",
        markdown_table(station_summary),
        "",
        "## Largest State-Linear Gains Over Global Mean",
        "",
        markdown_table(best[[
            "station_id",
            "global_mean",
            "target_month",
            "target_state_linear",
            "target_state_linear_gain_vs_global",
        ]]),
        "",
        "## Variable-Level Mean MAE by Mode",
        "",
        markdown_table(variable_results.groupby(["mode", "variable"], as_index=False)["mae"].mean()),
        "",
        "## Initial Interpretation",
        "",
    ]
    mean_gain = station_summary["target_state_linear_gain_vs_global"].mean()
    hard_gain = station_summary.loc[
        station_summary["station_id"].isin(["mount_sidley", "zhongshan"]),
        "target_state_linear_gain_vs_global",
    ].max()
    if mean_gain >= 0.02 or hard_gain >= 0.04:
        lines.append("The probe is positive under the Stage 0 gate: station-adaptive residual structure is large enough to justify a neural SABC experiment.")
    elif mean_gain >= 0.01 or hard_gain >= 0.02:
        lines.append("The probe is borderline: adaptive correction helps, but the expected neural SABC gain should be treated as uncertain.")
    else:
        lines.append("The probe is weak: simple station adaptation does not clearly outperform global calibration, so neural SABC should be postponed or redesigned.")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_plot(results: pd.DataFrame, out_path: Path) -> None:
    mean_rows = results[results["variable"].eq("mean")].copy()
    mode_order = ["none", "global_mean", "target_mean", "target_month", "target_state_linear"]
    station_order = list(mean_rows["station_id"].drop_duplicates())
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(station_order))
    width = 0.15
    colors = {
        "none": "#595959",
        "global_mean": "#d89c28",
        "target_mean": "#5aa6d6",
        "target_month": "#4f9b6b",
        "target_state_linear": "#b35b8b",
    }
    for i, mode in enumerate(mode_order):
        vals = (
            mean_rows[mean_rows["mode"].eq(mode)]
            .set_index("station_id")
            .loc[station_order, "mae"]
            .to_numpy()
        )
        ax.bar(x + (i - 2) * width, vals, width=width, label=mode.replace("_", " "), color=colors[mode])
    ax.set_xticks(x)
    ax.set_xticklabels(station_order, rotation=25, ha="right")
    ax.set_ylabel("ERA5-AWS residual MAE (normalized)")
    ax.set_title("Held-out station residual correction probe")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(META_CSV)
    selected = meta[meta["split"].isin(["main", "heldout"])].copy()
    scaler = np.load(SCALER_NPZ, allow_pickle=True)
    mean = scaler["mean"].astype(np.float32)
    std = scaler["std"].astype(np.float32)

    series = [load_station_series(row, mean, std) for _, row in selected.iterrows()]
    main_series = [s for s in series if s.group == "main"]
    heldout_series = [s for s in series if s.group == "heldout"]
    global_mean = fit_global_mean(main_series)

    rows: list[dict[str, object]] = []
    for s in heldout_series:
        target_mean = fit_target_mean(s)
        target_month = fit_target_month(s, target_mean)
        state_linear = fit_state_linear(s)
        rows.extend(evaluate(s, "none", np.zeros(len(VARIABLES), dtype=np.float32)))
        rows.extend(evaluate(s, "global_mean", global_mean))
        rows.extend(evaluate(s, "target_mean", target_mean))
        rows.extend(evaluate(s, "target_month", target_month))
        rows.extend(evaluate(s, "target_state_linear", state_linear))

    results = pd.DataFrame(rows)
    station_summary = improvement_summary(results)
    variable_summary = results.groupby(["mode", "variable"], as_index=False).agg(
        mae=("mae", "mean"),
        n_obs=("n_obs", "sum"),
    )

    results.to_csv(OUT_DIR / "heldout_bias_probe_by_variable.csv", index=False)
    station_summary.to_csv(OUT_DIR / "heldout_bias_probe_station_summary.csv", index=False)
    variable_summary.to_csv(OUT_DIR / "heldout_bias_probe_variable_summary.csv", index=False)
    write_report(station_summary, results, OUT_DIR / "sabc_feasibility_report.md")
    write_plot(results, OUT_DIR / "heldout_bias_probe_modes.png")
    print(station_summary.to_string(index=False))
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
