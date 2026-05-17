# ECBIT Progress Log

## Session: 2026-05-18

### Phase 0 · Initialization
- Added ECBIT as a Git submodule inside the archived ECAFT repository.
- Read `PLAN.md` and extracted the initial workflow.
- Created the project directory scaffold.
- Added `AGENTS.md`, `CLAUDE.md`, `README.md`, and initial agent planning files.
- Audited local data assets: AntAWS raw 3h CSVs are available under the existing `microclimate_demo` project; ECAFT repaired ERA5 assets cover 9 IMAU stations only.
- Verified remote GPU connectivity: server reachable, `darts` environment active, 6 x RTX 4090 available.
- Next task: implement station metadata/completeness audit for AntAWS and decide selected station set before ERA5 alignment.

### Phase 1 · Data Engineering
- Implemented `scripts/select_antaws_stations.py` with CSV encoding fallback, metadata merge, completeness/record-length statistics, and deterministic geographic held-out selection.
- Generated `data/station_meta_ecbit.csv`: 267 scanned station files, initially 36 eligible, then refined to 32 eligible after requiring every core variable to have at least 20% completeness.
- Acceptance check passed: selected station count is within 30-45 and held-out stations are clearly labeled.
- Next task: determine whether ERA5 exists for these selected AntAWS stations or whether ERA5 download/alignment must be implemented before 3h resampling.
- Confirmed selected AntAWS stations do not have pre-existing ECBIT ERA5 files; ECAFT repaired ERA5 only covers 9 IMAU stations and cannot satisfy ECBIT.
- Implemented `src/resample_era5.py` to download CDS ERA5 point time series and interpolate to AntAWS 3h timestamps.
- Smoke tested `aws06`; generated finite `data/era5_3h/aws06_era5_3h.npz` with shape `(35064, 5)`.
- Added `.gitignore` rules so generated ERA5 NetCDF/NPZ files stay out of version control.
- Ran full ERA5 alignment for selected stations; CDS time-series requests completed successfully.
- Added `scripts/validate_era5_3h.py` and generated `data/era5_3h_manifest.csv`.
- ERA5 validation passed: 32/32 selected stations, no non-finite values, all shapes match AntAWS observed steps.
- Implemented `scripts/analyze_missing_patterns.py` and generated missing-pattern CSVs under `results/missing_analysis/`.
- Real missingness is strongly block-structured; wind and RH have the largest missing burden, supporting ECBIT's block-imputation framing.
- Quantified complete-window feasibility and found that strict 168-step all-variable complete windows are unusable (35 total windows). Decided to implement sparse-window preprocessing with observation masks and artificial missing labels sampled only from observed positions.
- Implemented `src/preprocess_antaws_impute.py` and generated sparse imputation windows for all 32 selected stations.
- Preprocessing output: 41,388 retained windows at `seq_len=168`, including 34,962 main-station windows and 6,426 held-out-station windows.
- Validation passed under the local `darts` conda environment: shape checks, finite normalized `X/E_3h/T_enc`, binary `obs_mask`, and nonzero train/val/test windows for every station.
- Local environment note: plain `python` is not on PATH and system `python3` lacks NumPy; use `/home/horeb/miniconda3/bin/conda run -n darts python` for local data commands.

## Blocked Issues
| Timestamp | Issue | Status | Action |
|-----------|-------|--------|--------|
| 2026-05-18 | Full AntAWS station-level ERA5 alignment not yet confirmed | resolved | Downloaded and validated ERA5 3h files for all 32 selected AntAWS stations |
| 2026-05-18 | `python` missing from PATH; system `python3` lacks NumPy | resolved | Use `/home/horeb/miniconda3/bin/conda run -n darts python` for local data scripts |
