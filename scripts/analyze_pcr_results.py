#!/usr/bin/env python3
"""Aggregate physical-consistency regularization exploration results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from scipy import stats


METRICS_DIR = Path("experiments/results/metrics/exploration_pcr")
OUT_DIR = Path("experiments/results/exploration_pcr")


def load_result(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_runs(metrics_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result_path in sorted(metrics_dir.glob("*/result.json")):
        result = load_result(result_path)
        config = result["config"]
        missing = config.get("missing", {})
        model = config.get("model", {})
        metrics = result.get("test", result.get("best", {}))
        row = {
            "run_name": config.get("run_name", result_path.parent.name),
            "variant": model.get("variant", ""),
            "pattern": missing.get("target_pattern", missing.get("pattern", "")),
            "rate": float(missing.get("rate", float("nan"))),
            "seed": int(config.get("seed", -1)),
        }
        row.update({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
        rows.append(row)
    return pd.DataFrame(rows)


def paired_test(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame()
    idx = ["pattern", "rate", "seed"]
    baseline = runs[runs["variant"].eq("gated_baseline")][idx + ["mae_mean"]].rename(columns={"mae_mean": "baseline"})
    pcr = runs[runs["variant"].eq("pcr_w005")][idx + ["mae_mean"]].rename(columns={"mae_mean": "pcr"})
    pairs = baseline.merge(pcr, on=idx, how="inner")
    if pairs.empty:
        return pd.DataFrame()
    pairs["improvement"] = pairs["baseline"] - pairs["pcr"]
    rows = []
    for group_name, group in [("overall", pairs), *[(f"pattern:{p}", g) for p, g in pairs.groupby("pattern")]]:
        diff = group["improvement"].to_numpy(float)
        n = len(diff)
        mean = float(diff.mean())
        if n > 1:
            sd = float(diff.std(ddof=1))
            se = sd / math.sqrt(n)
            ci = stats.t.ppf(0.975, n - 1) * se
            t, p = stats.ttest_1samp(diff, 0.0)
        else:
            ci = float("nan")
            t = p = float("nan")
        rows.append(
            {
                "group": group_name,
                "n": n,
                "mean_improvement": mean,
                "ci95_low": mean - ci if math.isfinite(ci) else float("nan"),
                "ci95_high": mean + ci if math.isfinite(ci) else float("nan"),
                "t": float(t),
                "p": float(p),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df.empty:
        return ""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                vals.append(format(value, floatfmt))
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(runs: pd.DataFrame, tests: pd.DataFrame, out_dir: Path) -> None:
    lines = ["# PCR Gate Report", ""]
    lines.append(f"Completed runs: {len(runs)}")
    if not tests.empty:
        overall = tests[tests["group"].eq("overall")].iloc[0]
        mean = float(overall["mean_improvement"])
        ci_low = float(overall["ci95_low"])
        ci_high = float(overall["ci95_high"])
        passed = mean >= -0.003
        lines.append("")
        lines.append(
            f"Overall mean improvement (gated baseline - PCR): {mean:.4f} normalized MAE "
            f"with 95% CI [{ci_low:.4f}, {ci_high:.4f}]."
        )
        lines.append("")
        lines.append(f"MAE non-degradation gate (`>= -0.003`): {'pass' if passed else 'no pass'}")
        lines.extend(["", "## Paired Tests", "", markdown_table(tests)])
    else:
        lines.append("")
        lines.append("No matched baseline/PCR pairs are available yet.")
    (out_dir / "pcr_gate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, default=METRICS_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs = collect_runs(args.metrics_dir)
    runs.to_csv(args.out_dir / "pcr_runs.csv", index=False)
    tests = paired_test(runs)
    tests.to_csv(args.out_dir / "pcr_paired_tests.csv", index=False)
    write_report(runs, tests, args.out_dir)
    print(f"wrote {len(runs)} runs and {len(tests)} paired-test rows to {args.out_dir}")


if __name__ == "__main__":
    main()
