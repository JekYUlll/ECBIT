#!/usr/bin/env python3
"""Generate evaluation-only configs for the non-overlapping test-window subset."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


DEFAULT_SOURCE_DIR = Path("experiments/configs/fair_era5_baselines")
DEFAULT_OUT_DIR = Path("experiments/configs/revision/non_overlap_eval")
DEFAULT_OUTPUT_ROOT = Path("experiments/results/metrics/non_overlap_eval")
DEFAULT_SUBSET_CSV = Path("experiments/results/analysis/split_overlap/non_overlap_test_windows.csv")


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def convert_config(source_path: Path, out_dir: Path, output_root: Path, subset_csv: Path) -> Path:
    config = load_yaml(source_path)
    run_name = source_path.stem
    source_output_dir = Path(config["output_dir"])
    config["runner"] = "evaluate"
    config["checkpoint"] = str(source_output_dir / "best.pt")
    config["output_dir"] = str(output_root / run_name)
    config["run_name"] = run_name
    config.setdefault("data", {})["test_window_subset_csv"] = str(subset_csv)
    config.setdefault("eval", {})["num_workers"] = 0
    out_path = out_dir / source_path.name
    write_yaml(out_path, config)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--subset-csv", type=Path, default=DEFAULT_SUBSET_CSV)
    parser.add_argument(
        "--pattern",
        default="itransformer_era5_*.yaml",
        help="Glob pattern for source configs with retained checkpoints.",
    )
    args = parser.parse_args()

    source_configs = sorted(args.source_dir.glob(args.pattern))
    if not source_configs:
        raise SystemExit(f"No source configs matched {args.source_dir / args.pattern}")
    written = [convert_config(path, args.out_dir, args.output_root, args.subset_csv) for path in source_configs]
    print(f"wrote {len(written)} configs to {args.out_dir}")


if __name__ == "__main__":
    main()
