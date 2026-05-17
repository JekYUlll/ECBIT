# ECBIT Findings

## Starting Context

ECBIT starts from the archived ECAFT diagnosis:

- Real ERA5 data has now been validated in the parent ECAFT project.
- Naive ERA5-AWS cross-attention did not outperform single-stream PatchTST.
- ERA5 and AWS share highly correlated T, pressure, and q signals.
- ERA5 RH is biased and weakly correlated with AWS RH at several Antarctic stations.
- The new design should condition on ERA5 only where observations are missing.

## Initial Hypothesis

Block-wise imputation is a better fit than forecasting for ERA5 conditioning:

- Missing variables require external priors.
- Observed variables should remain anchored to AWS observations.
- ERA5 bias can be limited with variable-level missingness gates.
- Real polar AWS outages are block structured rather than MCAR point missing.

## Open Questions

- Which AntAWS stations satisfy completeness and record-length criteria?
- Are aligned ERA5 files available for the selected AntAWS stations, or only for IMAU stations?
- Does ERA5 conditioning help most for long block gaps and high-missingness regimes?
- Which variables benefit from ERA5, and where does RH bias hurt?

## Data Asset Audit (2026-05-18)

- AntAWS raw 3h CSV files exist locally at `/home/horeb/_code/microclimate_demo/data/AntAWS/3_hourly/`.
- AntAWS asset count: 268 files within 3 directory levels, total size about 305 MB.
- Existing processed AntAWS demo splits exist for Cape Hallett, Zhongshan, Taishan, and Kunlun under `/home/horeb/_code/microclimate_demo/data/processed/`.
- Repaired ECAFT ERA5 files exist only for 9 IMAU stations in `/home/horeb/_code/ECAFT/data/processed/`.
- Full AntAWS station-level ERA5 alignment is not yet confirmed; station selection must happen before ERA5 download/alignment scope is fixed.
- Remote GPU server is reachable; `darts` environment has Python 3.12.12 and 6 x RTX 4090.

## AntAWS Station Selection (2026-05-18)

- Implemented `scripts/select_antaws_stations.py`.
- Input: `/home/horeb/_code/microclimate_demo/data/AntAWS/3_hourly/`.
- Output: `data/station_meta_ecbit.csv`.
- Scanned 267 station CSV files.
- Eligibility criteria: record length >= 10 years, temperature completeness >= 70%, core mean completeness >= 50%, core minimum completeness >= 20%, valid lat/lon metadata.
- Eligible stations: 32.
- Selected split: 27 main + 5 held-out + 235 excluded.
- Minimum selected record length: 10.09 years.
- Minimum selected temperature completeness: 0.712.
- Held-out stations selected by geographic farthest-point sampling: Butcher Ridge, Mount Sidley, Nico, Sabrina, Zhongshan.

Important correction: an initial selection with only mean core completeness retained stations with RH almost or entirely missing. The selector now requires each core variable to have at least 20% completeness, keeping enough stations while removing unusable all-missing RH cases.

## ERA5 AntAWS Alignment Smoke Test (2026-05-18)

- Implemented `src/resample_era5.py`.
- Because full AntAWS ERA5 files were not available, the script handles both download and 3h alignment:
  1. download station point ERA5 from `reanalysis-era5-single-levels-timeseries`;
  2. derive `[T, RH, wspd, P, q]`;
  3. interpolate to exact AntAWS 3h timestamps;
  4. save `data/era5_3h/<station_id>_era5_3h.npz`.
- Smoke tested `aws06`: output shape `(35064, 5)`, timestamps shape `(35064,)`, all finite.
- Generated ERA5 NetCDF and aligned NPZ are ignored by Git; they are reproducible data artifacts.
- Next gate: run the same script for all 36 selected stations and verify every selected station has a finite 3h ERA5 file.
- Full selected-station ERA5 alignment completed: all 32 currently selected stations have `data/era5_3h/<station_id>_era5_3h.npz`.
- Validation passed with `scripts/validate_era5_3h.py`: each file has shape `(observed_steps, 5)`, timestamp length matches AntAWS, and `nan_count=0`.
- Validation manifest written to `data/era5_3h_manifest.csv`.

## Real Missing Pattern Analysis (2026-05-18)

- Implemented `scripts/analyze_missing_patterns.py`.
- Output files:
  - `results/missing_analysis/pattern_stats.csv`
  - `results/missing_analysis/pattern_summary.csv`
- The real AntAWS missingness is strongly block-structured, not MCAR-like.
- Wind speed has the highest average missing rate among selected main stations: 0.436.
- RH also has high missingness and extreme long blocks: main-station mean missing rate 0.301, max block 75,792 steps.
- Temperature has many short blocks: main-station mean block length 4.27 steps, but occasional long outages still occur.
- These findings support the ECBIT design choice of block missing simulation with long-block regimes.
