#!/usr/bin/env python3
"""Audit benchmark windows and realized artificial block lengths."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train_impute import artificial_mask_batch


MANIFEST_CSV = Path("data/antaws_impute_manifest.csv")
OUT_DIR = Path("experiments/results/analysis/benchmark_protocol")
PATTERNS = ("short", "medium", "long")
RATES = (0.2, 0.4, 0.6)


def contiguous_lengths(mask_1d: np.ndarray) -> list[int]:
    lengths: list[int] = []
    start: int | None = None
    for idx, value in enumerate(mask_1d.astype(bool)):
        if value and start is None:
            start = idx
        if start is not None and ((not value) or idx == len(mask_1d) - 1):
            end = idx + 1 if value and idx == len(mask_1d) - 1 else idx
            if end > start:
                lengths.append(end - start)
            start = None
    return lengths


def trace_block_missing(
    obs_mask: np.ndarray,
    pattern: str,
    missing_rate: float,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    ranges = {"short": (6, 24), "medium": (24, 72), "long": (72, 240)}
    min_block, max_block = ranges[pattern]
    allowed = obs_mask.astype(bool)
    t_steps, n_vars = allowed.shape
    mask = np.zeros((t_steps, n_vars), dtype=np.float32)
    target = int(round(float(allowed.sum()) * missing_rate))
    traces: list[dict[str, int]] = []
    if target <= 0 or int(allowed.sum()) == 0:
        return mask, traces

    generator = torch.Generator(device="cpu").manual_seed(seed)
    attempts = 0
    max_attempts = max(100, target * 20)
    while int(mask.sum()) < target and attempts < max_attempts:
        attempts += 1
        start = int(torch.randint(0, t_steps, (1,), generator=generator).item())
        requested = int(torch.randint(min_block, max_block + 1, (1,), generator=generator).item())
        end = min(start + requested, t_steps)
        var = int(torch.randint(0, n_vars, (1,), generator=generator).item())
        before = int(mask.sum())
        candidate = allowed[start:end, var]
        mask[start:end, var] = np.where(candidate, 1.0, mask[start:end, var])
        added = int(mask.sum()) - before
        traces.append(
            {
                "attempt": attempts,
                "variable_index": var,
                "start": start,
                "requested_steps": requested,
                "clipped_steps": end - start,
                "new_labels_added": added,
            }
        )
    return mask * allowed.astype(np.float32), traces


def load_windows(manifest_csv: Path) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    manifest = pd.read_csv(manifest_csv)
    rows: list[dict[str, object]] = []
    arrays: list[dict[str, object]] = []
    for _, station in manifest.iterrows():
        data = np.load(Path(station["path"]), allow_pickle=True)
        starts = data["window_start"].astype("datetime64[ns]")
        start_indices = data["window_start_index"].astype(int)
        splits = data["window_split"].astype(str)
        obs = data["obs_mask"].astype(np.float32)
        arrays.append(
            {
                "station": station["station"],
                "station_id": station["station_id"],
                "station_group": station["station_group"],
                "obs_mask": obs,
                "split": splits,
            }
        )
        for idx in range(obs.shape[0]):
            rows.append(
                {
                    "station": station["station"],
                    "station_id": station["station_id"],
                    "station_group": station["station_group"],
                    "window_local_index": idx,
                    "window_start": str(starts[idx]),
                    "window_start_index": int(start_indices[idx]),
                    "window_split": splits[idx],
                    "obs_fraction": float(obs[idx].mean()),
                    "targetable_vars_12steps": int((obs[idx].sum(axis=0) >= 12).sum()),
                }
            )
    return pd.DataFrame(rows), arrays


def summarize_windows(index: pd.DataFrame) -> pd.DataFrame:
    return (
        index.groupby(["station_group", "station", "station_id", "window_split"], as_index=False)
        .agg(
            windows=("window_local_index", "count"),
            mean_obs_fraction=("obs_fraction", "mean"),
            min_obs_fraction=("obs_fraction", "min"),
            mean_targetable_vars=("targetable_vars_12steps", "mean"),
        )
        .sort_values(["station_group", "station_id", "window_split"])
    )


def realized_blocks(
    arrays: list[dict[str, object]],
    split: str,
    station_group: str,
    max_windows_per_station: int | None,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in arrays:
        if item["station_group"] != station_group:
            continue
        obs = item["obs_mask"]
        splits = item["split"]
        keep = np.flatnonzero(splits == split)
        if max_windows_per_station is not None:
            keep = keep[:max_windows_per_station]
        for pattern in PATTERNS:
            for rate in RATES:
                missing_cfg = {"pattern": pattern, "rate": rate, "variable_wise": True}
                for local_pos, idx in enumerate(keep):
                    mask_seed = int(seed + int(idx) * 10_000)
                    mask, traces = trace_block_missing(obs[idx], pattern, rate, mask_seed)
                    lengths = []
                    for var_idx in range(mask.shape[1]):
                        lengths.extend(contiguous_lengths(mask[:, var_idx]))
                    if not lengths:
                        continue
                    for trace in traces:
                        rows.append(
                            {
                                "station_id": item["station_id"],
                                "window_local_index": int(idx),
                                "sample_index": int(local_pos),
                                "split": split,
                                "pattern": pattern,
                                "rate": rate,
                                "length_type": "sampled_interval",
                                "realized_length_steps": int(trace["requested_steps"]),
                                "realized_length_hours": int(trace["requested_steps"] * 3),
                                "hidden_fraction": float(mask.sum() / max(obs[idx].sum(), 1.0)),
                            }
                        )
                        rows.append(
                            {
                                "station_id": item["station_id"],
                                "window_local_index": int(idx),
                                "sample_index": int(local_pos),
                                "split": split,
                                "pattern": pattern,
                                "rate": rate,
                                "length_type": "clipped_interval",
                                "realized_length_steps": int(trace["clipped_steps"]),
                                "realized_length_hours": int(trace["clipped_steps"] * 3),
                                "hidden_fraction": float(mask.sum() / max(obs[idx].sum(), 1.0)),
                            }
                        )
                    for length in lengths:
                        rows.append(
                            {
                                "station_id": item["station_id"],
                                "window_local_index": int(idx),
                                "sample_index": int(local_pos),
                                "split": split,
                                "pattern": pattern,
                                "rate": rate,
                                "length_type": "supervised_label_segment",
                                "realized_length_steps": int(length),
                                "realized_length_hours": int(length * 3),
                                "hidden_fraction": float(mask.sum() / max(obs[idx].sum(), 1.0)),
                            }
                        )
    return pd.DataFrame(rows)


def summarize_blocks(blocks: pd.DataFrame) -> pd.DataFrame:
    if blocks.empty:
        return blocks
    return (
        blocks.groupby(["pattern", "rate", "length_type"], as_index=False)
        .agg(
            blocks=("realized_length_steps", "count"),
            mean_steps=("realized_length_steps", "mean"),
            median_steps=("realized_length_steps", "median"),
            p10_steps=("realized_length_steps", lambda s: float(np.percentile(s, 10))),
            p90_steps=("realized_length_steps", lambda s: float(np.percentile(s, 90))),
            max_steps=("realized_length_steps", "max"),
            mean_hours=("realized_length_hours", "mean"),
            max_hours=("realized_length_hours", "max"),
            mean_hidden_fraction=("hidden_fraction", "mean"),
        )
        .sort_values(["pattern", "rate"])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", type=Path, default=MANIFEST_CSV)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--split", default="test")
    parser.add_argument("--station-group", default="main")
    parser.add_argument("--max-windows-per-station", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260524)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index, arrays = load_windows(args.manifest_csv)
    summary = summarize_windows(index)
    blocks = realized_blocks(arrays, args.split, args.station_group, args.max_windows_per_station, args.seed)
    block_summary = summarize_blocks(blocks)

    index.to_csv(args.out_dir / "window_index.csv", index=False)
    summary.to_csv(args.out_dir / "station_window_summary.csv", index=False)
    blocks.to_csv(args.out_dir / "realized_block_lengths.csv", index=False)
    block_summary.to_csv(args.out_dir / "realized_block_length_summary.csv", index=False)

    print(f"Wrote {args.out_dir / 'window_index.csv'} ({len(index)} windows)")
    print(f"Wrote {args.out_dir / 'station_window_summary.csv'}")
    print(f"Wrote {args.out_dir / 'realized_block_lengths.csv'} ({len(blocks)} blocks)")
    print(block_summary.to_string(index=False))


if __name__ == "__main__":
    main()
