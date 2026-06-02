#!/usr/bin/env python3
"""Real historical-gap plausibility diagnostics without hidden ground truth."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_era5_direct_calibration import fit_calibration


MANIFEST_CSV = Path("data/antaws_impute_manifest.csv")
OUT_DIR = Path("experiments/results/analysis/real_gap_plausibility")
FIG_DIR = Path("paper/figures/real_gap_cases")

VAR_LABELS = {
    "T": "Temperature ($^\\circ$C)",
    "RH": "Relative humidity (%)",
    "wspd": "Wind speed (m/s)",
    "P": "Pressure (hPa)",
    "q": "Specific humidity (g/kg)",
}


def contiguous_false_spans(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask.astype(bool)):
        if not value and start is None:
            start = idx
        if start is not None and (value or idx == len(mask) - 1):
            end = idx + 1 if (not value and idx == len(mask) - 1) else idx
            if end - start >= min_len:
                spans.append((start, end))
            start = None
    return spans


def reconstruct_station(data: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    starts = data["window_start_index"].astype(int)
    seq_len = int(data["X"].shape[1])
    total = int(starts.max() + seq_len)
    n_vars = int(data["X"].shape[2])
    x = np.full((total, n_vars), np.nan, dtype=np.float32)
    era5 = np.full_like(x, np.nan)
    obs = np.zeros((total, n_vars), dtype=bool)
    timestamps = np.full(total, np.datetime64("NaT"), dtype="datetime64[ns]")
    for win_idx, start in enumerate(starts):
        stop = start + seq_len
        window_obs = data["obs_mask"][win_idx].astype(bool)
        fill = np.isnan(x[start:stop])
        for c in range(n_vars):
            pos = fill[:, c] | window_obs[:, c]
            x[start:stop, c][pos] = data["X"][win_idx, :, c][pos]
            obs[start:stop, c][pos] = window_obs[:, c][pos]
            era5[start:stop, c][pos] = data["E_3h"][win_idx, :, c][pos]
        time_fill = np.isnat(timestamps[start:stop])
        timestamps[start:stop][time_fill] = data["timestamps"][win_idx][time_fill]
    return {"x": x, "era5": era5, "obs": obs, "timestamps": timestamps}


def fit_station_linear(station: dict[str, np.ndarray], train_until: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.from_numpy(station["x"][:train_until][None, :, :])
    e = torch.from_numpy(station["era5"][:train_until][None, :, :])
    m = torch.from_numpy(station["obs"][:train_until][None, :, :])
    valid = torch.isfinite(x) & torch.isfinite(e) & m.bool()
    return fit_calibration(torch.nan_to_num(x), torch.nan_to_num(e), valid, "linear")


def nearest_observed(obs: np.ndarray, idx: int, direction: int, max_search: int) -> int | None:
    cur = idx
    steps = 0
    while 0 <= cur < len(obs) and steps <= max_search:
        if obs[cur]:
            return cur
        cur += direction
        steps += 1
    return None


def audit_station(
    manifest_row: pd.Series,
    min_gap_steps: int,
    max_boundary_search: int,
    max_cases: int,
    fig_dir: Path,
) -> list[dict[str, object]]:
    data = np.load(Path(manifest_row["path"]), allow_pickle=True)
    variables = [str(v) for v in data["variables"]]
    mean = data["normalization_mean"].astype(float)
    std = data["normalization_std"].astype(float)
    station = reconstruct_station(data)
    train_until = int(len(station["x"]) * 0.7)
    slope, intercept = fit_station_linear(station, train_until)
    calibrated = station["era5"] * slope.numpy().reshape(1, -1) + intercept.numpy().reshape(1, -1)

    rows: list[dict[str, object]] = []
    case_candidates: list[dict[str, object]] = []
    for var_idx, var in enumerate(variables):
        obs = station["obs"][:, var_idx]
        for start, end in contiguous_false_spans(obs, min_gap_steps):
            left = nearest_observed(obs, start - 1, -1, max_boundary_search)
            right = nearest_observed(obs, end, 1, max_boundary_search)
            if left is None or right is None:
                continue
            gap_fill = calibrated[start:end, var_idx]
            if not np.isfinite(gap_fill).all():
                continue
            left_obs = station["x"][left, var_idx]
            right_obs = station["x"][right, var_idx]
            if not np.isfinite(left_obs) or not np.isfinite(right_obs):
                continue
            left_jump = abs(gap_fill[0] - left_obs) * std[var_idx]
            right_jump = abs(gap_fill[-1] - right_obs) * std[var_idx]
            gap_slope = (gap_fill[-1] - gap_fill[0]) / max(end - start, 1)
            boundary_slope = (right_obs - left_obs) / max(right - left, 1)
            row = {
                "station": manifest_row["station"],
                "station_id": manifest_row["station_id"],
                "station_group": manifest_row["station_group"],
                "variable": var,
                "variable_label": VAR_LABELS[var],
                "start_time": str(station["timestamps"][start]),
                "end_time": str(station["timestamps"][end - 1]),
                "gap_steps": int(end - start),
                "gap_days": float((end - start) * 3 / 24),
                "left_boundary_gap_steps": int(start - left),
                "right_boundary_gap_steps": int(right - end + 1),
                "left_jump_raw": float(left_jump),
                "right_jump_raw": float(right_jump),
                "mean_boundary_jump_raw": float((left_jump + right_jump) / 2),
                "slope_difference_raw_per_day": float(abs(gap_slope - boundary_slope) * std[var_idx] * 8),
            }
            rows.append(row)
            case_candidates.append({**row, "start": start, "end": end, "left": left, "right": right, "var_idx": var_idx})

    case_candidates = sorted(case_candidates, key=lambda r: (r["gap_steps"], -r["mean_boundary_jump_raw"]), reverse=True)
    for case_idx, case in enumerate(case_candidates[:max_cases]):
        plot_case(station, calibrated, mean, std, variables, case, fig_dir / f"real_gap_case_{manifest_row['station_id']}_{case_idx+1}.png")
    return rows


def audit_station_windows(
    manifest_row: pd.Series,
    min_gap_steps: int,
    max_cases: int,
    fig_dir: Path,
) -> list[dict[str, object]]:
    data = np.load(Path(manifest_row["path"]), allow_pickle=True)
    variables = [str(v) for v in data["variables"]]
    mean = data["normalization_mean"].astype(float)
    std = data["normalization_std"].astype(float)
    train = data["window_split"].astype(str) == "train"
    slope, intercept = fit_calibration(
        torch.from_numpy(data["X"][train]),
        torch.from_numpy(data["E_3h"][train]),
        torch.from_numpy(data["obs_mask"][train]).bool(),
        "linear",
    )
    calibrated = data["E_3h"] * slope.numpy().reshape(1, 1, -1) + intercept.numpy().reshape(1, 1, -1)
    rows: list[dict[str, object]] = []
    cases: list[dict[str, object]] = []
    candidate_windows = np.flatnonzero(data["window_split"].astype(str) == "test")
    for win_idx in candidate_windows:
        obs_win = data["obs_mask"][win_idx].astype(bool)
        x_win = data["X"][win_idx]
        times = data["timestamps"][win_idx]
        for var_idx, var in enumerate(variables):
            for start, end in contiguous_false_spans(obs_win[:, var_idx], min_gap_steps):
                left = start - 1
                right = end
                if left < 0 or right >= obs_win.shape[0] or not obs_win[left, var_idx] or not obs_win[right, var_idx]:
                    continue
                gap_fill = calibrated[win_idx, start:end, var_idx]
                if not np.isfinite(gap_fill).all():
                    continue
                left_obs = x_win[left, var_idx]
                right_obs = x_win[right, var_idx]
                left_jump = abs(gap_fill[0] - left_obs) * std[var_idx]
                right_jump = abs(gap_fill[-1] - right_obs) * std[var_idx]
                gap_slope = (gap_fill[-1] - gap_fill[0]) / max(end - start, 1)
                boundary_slope = (right_obs - left_obs) / max(right - left, 1)
                row = {
                    "station": manifest_row["station"],
                    "station_id": manifest_row["station_id"],
                    "station_group": manifest_row["station_group"],
                    "window_local_index": int(win_idx),
                    "variable": var,
                    "variable_label": VAR_LABELS[var],
                    "start_time": str(times[start]),
                    "end_time": str(times[end - 1]),
                    "gap_steps": int(end - start),
                    "gap_days": float((end - start) * 3 / 24),
                    "left_jump_raw": float(left_jump),
                    "right_jump_raw": float(right_jump),
                    "mean_boundary_jump_raw": float((left_jump + right_jump) / 2),
                    "slope_difference_raw_per_day": float(abs(gap_slope - boundary_slope) * std[var_idx] * 8),
                }
                rows.append(row)
                cases.append({**row, "win_idx": int(win_idx), "start": start, "end": end, "var_idx": var_idx})
    cases = sorted(cases, key=lambda r: (r["gap_steps"], -r["mean_boundary_jump_raw"]), reverse=True)
    for case_idx, case in enumerate(cases[:max_cases]):
        plot_window_case(
            data,
            calibrated,
            mean,
            std,
            variables,
            case,
            fig_dir / f"real_gap_case_{manifest_row['station_id']}_{case_idx+1}.png",
        )
    return rows


def denorm(values: np.ndarray, mean: np.ndarray, std: np.ndarray, var_idx: int) -> np.ndarray:
    return values * std[var_idx] + mean[var_idx]


def plot_case(
    station: dict[str, np.ndarray],
    calibrated: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    variables: list[str],
    case: dict[str, object],
    out_path: Path,
) -> None:
    start = int(case["start"])
    end = int(case["end"])
    left = int(case["left"])
    right = int(case["right"])
    var_idx = int(case["var_idx"])
    lo = max(left - 16, 0)
    hi = min(right + 17, len(station["x"]))
    t = pd.to_datetime(station["timestamps"][lo:hi])
    obs = station["obs"][lo:hi, var_idx]
    x_raw = denorm(station["x"][lo:hi, var_idx], mean, std, var_idx)
    fill_raw = denorm(calibrated[lo:hi, var_idx], mean, std, var_idx)

    fig, ax = plt.subplots(figsize=(7.2, 2.3))
    ax.plot(t, fill_raw, color="#0072B2", linewidth=1.2, label="ERA5 linear-calibrated")
    ax.scatter(t[obs], x_raw[obs], s=10, color="#222222", label="Observed AWS", zorder=3)
    ax.axvspan(pd.to_datetime(station["timestamps"][start]), pd.to_datetime(station["timestamps"][end - 1]), color="#E69F00", alpha=0.18)
    ax.set_ylabel(VAR_LABELS[variables[var_idx]])
    ax.set_xlabel("Time")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=2)
    fig.autofmt_xdate(rotation=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_window_case(
    data: np.lib.npyio.NpzFile,
    calibrated: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    variables: list[str],
    case: dict[str, object],
    out_path: Path,
) -> None:
    win_idx = int(case["win_idx"])
    start = int(case["start"])
    end = int(case["end"])
    var_idx = int(case["var_idx"])
    lo = max(start - 16, 0)
    hi = min(end + 17, data["X"].shape[1])
    t = pd.to_datetime(data["timestamps"][win_idx, lo:hi])
    obs = data["obs_mask"][win_idx, lo:hi, var_idx].astype(bool)
    x_raw = denorm(data["X"][win_idx, lo:hi, var_idx], mean, std, var_idx)
    fill_raw = denorm(calibrated[win_idx, lo:hi, var_idx], mean, std, var_idx)

    fig, ax = plt.subplots(figsize=(7.2, 2.3))
    ax.plot(t, fill_raw, color="#0072B2", linewidth=1.2, label="ERA5 linear-calibrated")
    ax.scatter(t[obs], x_raw[obs], s=10, color="#222222", label="Observed AWS", zorder=3)
    ax.axvspan(pd.to_datetime(data["timestamps"][win_idx, start]), pd.to_datetime(data["timestamps"][win_idx, end - 1]), color="#E69F00", alpha=0.18)
    ax.set_ylabel(VAR_LABELS[variables[var_idx]])
    ax.set_xlabel("Time")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=2)
    fig.autofmt_xdate(rotation=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", type=Path, default=MANIFEST_CSV)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    parser.add_argument("--min-gap-steps", type=int, default=24)
    parser.add_argument("--max-boundary-search", type=int, default=56)
    parser.add_argument("--cases-per-station", type=int, default=1)
    parser.add_argument("--station-group", choices=["main", "heldout", "all"], default="heldout")
    parser.add_argument("--max-stations", type=int, default=None)
    parser.add_argument("--full-timeline", action="store_true")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest_csv)
    if args.station_group != "all":
        manifest = manifest[manifest["station_group"].eq(args.station_group)].copy()
    if args.max_stations is not None:
        manifest = manifest.head(args.max_stations).copy()
    rows: list[dict[str, object]] = []
    for _, row in manifest.iterrows():
        if args.full_timeline:
            rows.extend(audit_station(row, args.min_gap_steps, args.max_boundary_search, args.cases_per_station, args.fig_dir))
        else:
            rows.extend(audit_station_windows(row, args.min_gap_steps, args.cases_per_station, args.fig_dir))
    gaps = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gaps.to_csv(args.out_dir / "real_gap_plausibility_gaps.csv", index=False)
    summary = (
        gaps.groupby(["variable", "variable_label"], as_index=False)
        .agg(
            gaps=("gap_steps", "count"),
            median_gap_days=("gap_days", "median"),
            p90_gap_days=("gap_days", lambda s: float(np.percentile(s, 90))),
            mean_boundary_jump_raw=("mean_boundary_jump_raw", "mean"),
            median_boundary_jump_raw=("mean_boundary_jump_raw", "median"),
            mean_slope_difference_raw_per_day=("slope_difference_raw_per_day", "mean"),
        )
        .sort_values("variable")
    )
    summary.to_csv(args.out_dir / "real_gap_plausibility_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {args.out_dir}")
    print(f"Wrote case figures to {args.fig_dir}")


if __name__ == "__main__":
    main()
