#!/usr/bin/env python3
"""Plot a qualitative long-block imputation case from AntAWS windows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paper_plot_style import apply_paper_style, save_figure, style_axis
from src.baselines.era5_direct import ERA5DirectImputer
from src.baselines.linear_interp import linear_interpolate
from src.utils.block_missing import apply_mask, simulate_block_missing


MANIFEST = Path("data/antaws_impute_manifest.csv")
OUT_PDF = Path("paper/figures/fig_imputation_case.pdf")
OUT_PNG = Path("paper/figures/fig_imputation_case.png")
VARIABLES = ["T", "RH", "wspd", "P", "q"]


def load_station(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def fit_era5_direct(manifest_csv: Path, limit_stations: int | None) -> ERA5DirectImputer:
    manifest = pd.read_csv(manifest_csv)
    manifest = manifest[manifest["station_group"].eq("main")].copy()
    if limit_stations is not None:
        manifest = manifest.head(limit_stations)
    xs, era5s, masks = [], [], []
    for _, row in manifest.iterrows():
        station = load_station(Path(row["path"]))
        keep = station["window_split"].astype(str) == "train"
        xs.append(station["X"][keep])
        era5s.append(station["E_3h"][keep])
        masks.append(station["obs_mask"][keep])
    return ERA5DirectImputer().fit(np.concatenate(xs), np.concatenate(era5s), np.concatenate(masks))


def longest_true_span(mask_1d: np.ndarray) -> tuple[int, int]:
    best = (0, 0)
    start = None
    for idx, value in enumerate(mask_1d.astype(bool)):
        if value and start is None:
            start = idx
        if (not value or idx == len(mask_1d) - 1) and start is not None:
            end = idx + 1 if value and idx == len(mask_1d) - 1 else idx
            if end - start > best[1] - best[0]:
                best = (start, end)
            start = None
    return best


def select_case(
    station: dict[str, np.ndarray],
    var_idx: int,
    missing_rate: float,
    min_block: int,
    max_block: int,
    seed: int,
) -> tuple[int, torch.Tensor]:
    split = station["window_split"].astype(str)
    candidates = np.flatnonzero(split == "test")
    best_idx = int(candidates[0])
    best_mask = None
    best_len = -1
    for rank, idx in enumerate(candidates[:250]):
        obs = torch.from_numpy(station["obs_mask"][idx].astype(np.float32))
        generator = torch.Generator().manual_seed(seed + rank)
        mask = simulate_block_missing(
            obs.shape,
            missing_rate=missing_rate,
            min_block=min_block,
            max_block=max_block,
            variable_wise=True,
            obs_mask=obs,
            generator=generator,
        )
        start, end = longest_true_span(mask[:, var_idx].numpy())
        length = end - start
        if length > best_len:
            best_idx = int(idx)
            best_mask = mask
            best_len = length
        if length >= min(48, max_block):
            break
    if best_mask is None:
        raise RuntimeError("Failed to sample a valid artificial mask")
    return best_idx, best_mask


def denorm(values: np.ndarray, mean: np.ndarray, std: np.ndarray, var_idx: int) -> np.ndarray:
    return values[..., var_idx] * std[var_idx] + mean[var_idx]


def plot_case(args: argparse.Namespace) -> None:
    era5_direct = fit_era5_direct(args.manifest, args.bias_stations)
    station = load_station(args.station_npz)
    var_idx = VARIABLES.index(args.variable)
    idx, artificial = select_case(station, var_idx, args.missing_rate, args.min_block, args.max_block, args.seed)

    x = torch.from_numpy(station["X"][idx : idx + 1].astype(np.float32))
    obs_mask = torch.from_numpy(station["obs_mask"][idx : idx + 1].astype(np.float32))
    era5 = torch.from_numpy(station["E_3h"][idx : idx + 1].astype(np.float32))
    artificial_b = artificial.unsqueeze(0)
    model_missing = torch.clamp((1.0 - obs_mask) + artificial_b, 0.0, 1.0)
    x_obs = apply_mask(x, model_missing)
    observed_after_hide = 1.0 - model_missing

    linear = linear_interpolate(x_obs, observed_after_hide).numpy()[0]
    era5_fill = era5_direct.impute(x_obs, era5, observed_after_hide).numpy()[0]

    mean = station["normalization_mean"].astype(np.float32)
    std = station["normalization_std"].astype(np.float32)
    timestamps = pd.to_datetime(station["timestamps"][idx])
    true = denorm(station["X"][idx], mean, std, var_idx)
    era5_raw = denorm(station["E_3h"][idx], mean, std, var_idx)
    linear_y = denorm(linear, mean, std, var_idx)
    era5_y = denorm(era5_fill, mean, std, var_idx)
    observed_y = np.where(observed_after_hide.numpy()[0, :, var_idx] > 0.5, true, np.nan)
    hidden_y = np.where(artificial[:, var_idx].numpy() > 0.5, true, np.nan)

    apply_paper_style(font_size=9.0)
    fig, ax = plt.subplots(figsize=(7.35, 2.65), constrained_layout=False)
    real_observed = obs_mask.numpy()[0, :, var_idx] > 0.5
    true_observed_line = np.where(real_observed, true, np.nan)
    ax.plot(timestamps, true_observed_line, color="#222222", linewidth=1.2, label="AWS truth")
    ax.plot(timestamps, era5_raw, color="#3182BD", linewidth=1.0, alpha=0.75, label="ERA5")
    ax.plot(timestamps, linear_y, color="#E6550D", linewidth=1.0, linestyle="--", label="Linear fill")
    ax.plot(timestamps, era5_y, color="#31A354", linewidth=1.0, linestyle="-.", label="ERA5 direct fill")
    ax.scatter(timestamps, observed_y, s=8, color="#222222", alpha=0.65, label="Observed", zorder=3)
    ax.scatter(timestamps, hidden_y, s=10, color="#D62728", alpha=0.8, label="Artificially hidden", zorder=4)

    mask = artificial[:, var_idx].numpy().astype(bool)
    start = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        if (not value or i == len(mask) - 1) and start is not None:
            end = i + 1 if value and i == len(mask) - 1 else i
            ax.axvspan(timestamps[start], timestamps[end - 1], color="#D62728", alpha=0.08, linewidth=0)
            start = None

    ax.set_ylabel(args.ylabel)
    ax.set_xlabel("Time")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    style_axis(ax, grid_axis="y")
    ax.legend(
        ncol=3,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        columnspacing=1.4,
        handlelength=2.0,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(top=0.80, bottom=0.20, left=0.08, right=0.99)

    save_figure(fig, args.out_pdf, args.out_png)
    print(f"Selected window index {idx}")
    print(f"Wrote {args.out_pdf}")
    print(f"Wrote {args.out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--station-npz", type=Path, default=Path("data/antaws/processed/aws06_impute.npz"))
    parser.add_argument("--variable", choices=VARIABLES, default="T")
    parser.add_argument("--ylabel", default=r"Temperature ($^\circ$C)")
    parser.add_argument("--missing-rate", type=float, default=0.4)
    parser.add_argument("--min-block", type=int, default=72)
    parser.add_argument("--max-block", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--bias-stations", type=int, default=None)
    parser.add_argument("--out-pdf", type=Path, default=OUT_PDF)
    parser.add_argument("--out-png", type=Path, default=OUT_PNG)
    args = parser.parse_args()
    plot_case(args)


if __name__ == "__main__":
    main()
