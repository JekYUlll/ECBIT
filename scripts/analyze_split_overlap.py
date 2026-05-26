#!/usr/bin/env python3
"""Audit chronological split gaps and construct a non-overlapping test subset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MANIFEST_CSV = Path("data/antaws_impute_manifest.csv")
OUT_DIR = Path("experiments/results/analysis/split_overlap")
OUT_TEX = Path("paper/tables/tab_split_overlap_audit.tex")


def greedy_non_overlap(starts: np.ndarray, indices: np.ndarray, seq_len: int) -> np.ndarray:
    order = np.argsort(starts)
    chosen: list[int] = []
    last_end = -1
    for pos in order:
        start = int(starts[pos])
        if start >= last_end:
            chosen.append(int(indices[pos]))
            last_end = start + seq_len
    return np.asarray(chosen, dtype=np.int64)


def median_stride(starts: np.ndarray) -> float:
    if len(starts) < 2:
        return float("nan")
    return float(np.median(np.diff(np.sort(starts))))


def audit_station(row: pd.Series) -> tuple[dict[str, object], list[dict[str, object]]]:
    data = np.load(row["path"], allow_pickle=True)
    starts = data["window_start_index"].astype(int)
    start_times = data["window_start"].astype("datetime64[ns]") if "window_start" in data.files else None
    splits = data["window_split"].astype(str)
    seq_len = int(data["seq_len"]) if "seq_len" in data.files else int(data["X"].shape[1])
    local_indices = np.arange(len(splits), dtype=np.int64)

    summary: dict[str, object] = {
        "station": row["station"],
        "station_id": row["station_id"],
        "station_group": row["station_group"],
        "seq_len_steps": seq_len,
    }
    for split in ("train", "val", "test"):
        idx = local_indices[splits == split]
        split_starts = starts[idx]
        summary[f"{split}_windows"] = int(len(idx))
        summary[f"{split}_median_stride_steps"] = median_stride(split_starts)
        summary[f"{split}_adjacent_overlap_pairs"] = int((np.diff(np.sort(split_starts)) < seq_len).sum()) if len(idx) > 1 else 0
        summary[f"{split}_min_start"] = int(split_starts.min()) if len(idx) else np.nan
        summary[f"{split}_max_start"] = int(split_starts.max()) if len(idx) else np.nan

    def boundary_gap(left: str, right: str) -> float:
        left_idx = local_indices[splits == left]
        right_idx = local_indices[splits == right]
        if len(left_idx) == 0 or len(right_idx) == 0:
            return float("nan")
        return float(starts[right_idx].min() - (starts[left_idx].max() + seq_len))

    summary["train_val_gap_steps"] = boundary_gap("train", "val")
    summary["train_test_gap_steps"] = boundary_gap("train", "test")
    summary["val_test_gap_steps"] = boundary_gap("val", "test")

    test_idx = local_indices[splits == "test"]
    chosen = greedy_non_overlap(starts[test_idx], test_idx, seq_len) if len(test_idx) else np.asarray([], dtype=np.int64)
    summary["test_nonoverlap_windows"] = int(len(chosen))
    summary["test_nonoverlap_fraction"] = float(len(chosen) / len(test_idx)) if len(test_idx) else float("nan")

    subset_rows: list[dict[str, object]] = []
    for idx in chosen:
        subset_rows.append(
            {
                "station": row["station"],
                "station_id": row["station_id"],
                "station_group": row["station_group"],
                "window_local_index": int(idx),
                "window_start_index": int(starts[idx]),
                "window_start": str(start_times[idx]) if start_times is not None else "",
                "window_split": "test",
            }
        )
    return summary, subset_rows


def write_latex_table(summary: pd.DataFrame, out_tex: Path) -> None:
    total_test = int(summary["test_windows"].sum())
    total_nonoverlap = int(summary["test_nonoverlap_windows"].sum())
    retained = 100.0 * total_nonoverlap / total_test
    train_test_min = int(summary["train_test_gap_steps"].min())
    train_test_median = int(summary["train_test_gap_steps"].median())
    val_test_min = int(summary["val_test_gap_steps"].min())
    val_test_median = int(summary["val_test_gap_steps"].median())
    test_stride = int(summary["test_median_stride_steps"].median())
    train_test_overlap = int((summary["train_test_gap_steps"] < 0).sum())
    val_test_overlap = int((summary["val_test_gap_steps"] < 0).sum())
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Chronological split-overlap audit. Gaps are measured between the end of the earlier split window and the start of the later split window within each station. Negative values indicate context overlap. The non-overlap subset greedily retains test windows separated by at least one 168-step window length.}",
        "  \\label{tab:split-overlap-audit}",
        "  \\footnotesize",
        "  \\setlength{\\tabcolsep}{3pt}",
        "  \\begin{tabular}{@{}p{0.36\\linewidth}p{0.39\\linewidth}p{0.18\\linewidth}@{}}",
        "    \\toprule",
        "    Quantity & Value & Stations \\\\",
        "    \\midrule",
        f"    Test windows & {total_test} & 32 \\\\",
        f"    Non-overlap subset & {total_nonoverlap} ({retained:.1f}\\%) & 32 \\\\",
        f"    Test stride & median {test_stride} steps & 32 \\\\",
        f"    Train--test gap & min {train_test_min}; median {train_test_median} steps & {train_test_overlap} overlap \\\\",
        f"    Val--test gap & min {val_test_min}; median {val_test_median} steps & {val_test_overlap} overlap \\\\",
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ]
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", type=Path, default=MANIFEST_CSV)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--out-tex", type=Path, default=OUT_TEX)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest_csv)
    station_rows: list[dict[str, object]] = []
    subset_rows: list[dict[str, object]] = []
    for row in manifest.itertuples(index=False):
        station_summary, station_subset = audit_station(pd.Series(row._asdict()))
        station_rows.append(station_summary)
        subset_rows.extend(station_subset)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(station_rows)
    subset = pd.DataFrame(subset_rows)
    overview = pd.DataFrame(
        [
            {
                "stations": int(len(summary)),
                "test_windows": int(summary["test_windows"].sum()),
                "nonoverlap_test_windows": int(summary["test_nonoverlap_windows"].sum()),
                "nonoverlap_fraction": float(summary["test_nonoverlap_windows"].sum() / summary["test_windows"].sum()),
                "train_test_overlap_stations": int((summary["train_test_gap_steps"] < 0).sum()),
                "val_test_overlap_stations": int((summary["val_test_gap_steps"] < 0).sum()),
                "min_train_test_gap_steps": int(summary["train_test_gap_steps"].min()),
                "median_train_test_gap_steps": int(summary["train_test_gap_steps"].median()),
                "min_val_test_gap_steps": int(summary["val_test_gap_steps"].min()),
                "median_val_test_gap_steps": int(summary["val_test_gap_steps"].median()),
            }
        ]
    )
    summary.to_csv(args.out_dir / "split_overlap_station_summary.csv", index=False)
    overview.to_csv(args.out_dir / "split_overlap_summary.csv", index=False)
    subset.to_csv(args.out_dir / "non_overlap_test_windows.csv", index=False)
    write_latex_table(summary, args.out_tex)
    print(overview.to_string(index=False))
    print(f"wrote {args.out_dir / 'non_overlap_test_windows.csv'} ({len(subset)} rows)")


if __name__ == "__main__":
    main()
