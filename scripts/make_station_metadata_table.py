#!/usr/bin/env python3
"""Create station-level metadata table for the ECBIT benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


META_CSV = Path("data/station_meta_ecbit.csv")
OUT_CSV = Path("experiments/results/revision/station_metadata_table.csv")
OUT_TEX = Path("paper/tables/tab_station_metadata.tex")
OUT_FULL_TEX = Path("paper/tables/tab_station_metadata_full.tex")
VALUE_COLS = {
    "T": "Temperature(℃)",
    "RH": "Relative Humidity(%)",
    "P": "Pressure(hPa)",
}
ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "latin1")


def pct(value: float) -> str:
    return f"{100.0 * value:.0f}"


def read_csv_with_fallback(path: Path, usecols: list[str]) -> pd.DataFrame:
    for encoding in ENCODINGS:
        try:
            return pd.read_csv(path, usecols=usecols, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, usecols=usecols, encoding="utf-8", encoding_errors="replace")


def q_completeness(path: Path, expected_steps: int) -> float:
    df = read_csv_with_fallback(path, list(VALUE_COLS.values()))
    valid = pd.Series(True, index=df.index)
    for col in VALUE_COLS.values():
        valid &= pd.to_numeric(df[col], errors="coerce").notna()
    return float(valid.sum() / expected_steps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-csv", type=Path, default=META_CSV)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--out-tex", type=Path, default=OUT_TEX)
    parser.add_argument("--out-full-tex", type=Path, default=OUT_FULL_TEX)
    args = parser.parse_args()

    meta = pd.read_csv(args.meta_csv)
    selected = meta[meta["split"].isin(["main", "heldout"])].copy()
    selected["start_year"] = pd.to_datetime(selected["start"]).dt.year
    selected["end_year"] = pd.to_datetime(selected["end"]).dt.year
    selected["specific_humidity_completeness"] = [
        q_completeness(Path(row.file), int(row.expected_steps)) for row in selected.itertuples(index=False)
    ]
    table = selected[
        [
            "station",
            "lat",
            "lon",
            "elev_m",
            "start_year",
            "end_year",
            "temperature_completeness",
            "relative_humidity_completeness",
            "wind_speed_completeness",
            "pressure_completeness",
            "specific_humidity_completeness",
            "split",
        ]
    ].sort_values(["split", "station"])
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_csv, index=False)

    summary = (
        table.groupby("split", as_index=False)
        .agg(
            stations=("station", "count"),
            start_year=("start_year", "min"),
            end_year=("end_year", "max"),
            elev_m_median=("elev_m", "median"),
            T_mean=("temperature_completeness", "mean"),
            RH_mean=("relative_humidity_completeness", "mean"),
            wind_mean=("wind_speed_completeness", "mean"),
            P_mean=("pressure_completeness", "mean"),
            q_mean=("specific_humidity_completeness", "mean"),
        )
        .sort_values("split")
    )
    summary_lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Benchmark station summary. Elevation is the split median; completeness values are mean observed fractions over retained raw 3-hourly station records. Full station metadata are reported in Appendix Table~\\ref{tab:station-metadata-full}.}",
        "  \\label{tab:station-metadata}",
        "  \\begin{tabular}{lrrrrrr}",
        "    \\toprule",
        "    Split & Stations & Years & Median elev. & T & RH & q \\\\",
        "    & & & (m) & (\\%) & (\\%) & (\\%) \\\\",
        "    \\midrule",
    ]
    for row in summary.itertuples(index=False):
        split = "held-out" if row.split == "heldout" else "main"
        summary_lines.append(
            f"    {split} & {int(row.stations)} & {int(row.start_year)}--{int(row.end_year)} & "
            f"{row.elev_m_median:.0f} & {pct(row.T_mean)} & {pct(row.RH_mean)} & {pct(row.q_mean)} \\\\"
        )
    summary_lines.extend(["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""])
    args.out_tex.parent.mkdir(parents=True, exist_ok=True)
    args.out_tex.write_text("\n".join(summary_lines), encoding="utf-8")

    lines = [
        "\\begin{table*}[p]",
        "  \\centering",
        "  \\tiny",
        "  \\caption{Selected station metadata for the AntAWS-derived benchmark. Completeness columns report the fraction of observed 3-hourly values in the retained raw station record before sparse-window construction.}",
        "  \\label{tab:station-metadata-full}",
        "  \\begin{tabular}{lrrrrrrrrrrl}",
        "    \\toprule",
        "    Station & Lat. & Lon. & Elev. & Start & End & T & RH & Wind & P & q & Split \\\\",
        "    & (deg) & (deg) & (m) & & & (\\%) & (\\%) & (\\%) & (\\%) & (\\%) & \\\\",
        "    \\midrule",
    ]
    for row in table.itertuples(index=False):
        station = str(row.station).strip().replace("_", "\\_")
        split = "held-out" if row.split == "heldout" else "main"
        lines.append(
            f"    {station} & {row.lat:.2f} & {row.lon:.2f} & {row.elev_m:.0f} & "
            f"{int(row.start_year)} & {int(row.end_year)} & "
            f"{pct(row.temperature_completeness)} & {pct(row.relative_humidity_completeness)} & "
            f"{pct(row.wind_speed_completeness)} & {pct(row.pressure_completeness)} & "
            f"{pct(row.specific_humidity_completeness)} & {split} \\\\"
        )
    lines.extend(["    \\bottomrule", "  \\end{tabular}", "\\end{table*}", ""])
    args.out_full_tex.parent.mkdir(parents=True, exist_ok=True)
    args.out_full_tex.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_tex}")
    print(f"wrote {args.out_full_tex}")


if __name__ == "__main__":
    main()
