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

## Preprocessing Design Correction (2026-05-18)

- A strict "all variables complete for a 168-step window" requirement is infeasible on AntAWS.
- With 32 selected stations and stride `seq_len/4`, all-variable complete windows are:
  - seq=24: 15,008 windows
  - seq=56: 1,710 windows
  - seq=84: 519 windows
  - seq=168: 35 windows
- Therefore, ECBIT preprocessing must keep sparse windows with observation masks rather than only `X_full` windows.
- Training/evaluation should simulate artificial block missing only on positions where real ground truth is observed, and compute loss only on those artificially hidden observed positions.
- This is a correction to the initial PLAN wording, but it better matches the paper title: sparse Antarctic station time series.

## AntAWS Imputation Preprocessing (2026-05-18)

- Implemented sparse-window preprocessing in `src/preprocess_antaws_impute.py`.
- Output manifest: `data/antaws_impute_manifest.csv`.
- Generated station NPZ files under `data/antaws/processed/` with fields `X`, `obs_mask`, `E_3h`, `T_enc`, `timestamps`, `window_split`, and normalization metadata.
- Windowing configuration: `seq_len=168`, `stride=42`, `min_obs_fraction=0.25`, `min_obs_steps_per_var=12`, `min_targetable_vars=3`.
- Total retained windows: 41,388.
- Main stations: 27 stations, 34,962 windows, mean observed fraction 0.799.
- Held-out stations: 5 stations, 6,426 windows, mean observed fraction 0.813.
- Temporal split is assigned by chronological order after sparse-window filtering, so every selected station retains nonzero train/val/test windows without shuffling.
- Validation passed: sample NPZ files have `X/E_3h/obs_mask` shape `(N,168,5)`, `T_enc` shape `(N,168,4)`, finite normalized inputs, and binary observation masks.

## Block Missing Simulator (2026-05-18)

- Implemented `src/utils/block_missing.py`.
- `simulate_block_missing()` supports variable-wise blocks and all-variable outage blocks.
- Artificial missing labels are constrained by `obs_mask`, preventing loss from being computed on real missing values.
- Added `simulate_mcar_missing()` for point-missing ablations, `apply_mask()`, `artificial_label_mask()`, and `missing_variables()`.
- Unit tests passed: `pytest -q src/tests/test_block_missing.py` reports 5 passed.

## ECBIT Model Smoke Test (2026-05-18)

- Implemented `src/models/ecbit.py` with variate-token AWS encoder, variate-token ERA5 encoder, conditional cross-attention, and per-variable reconstruction head.
- The cross-attention gate is variable-level: variables without missing positions are independent of ERA5 changes after fusion.
- Added `src/tests/test_ecbit.py`.
- Unit tests plus block-missing tests passed: 10 tests passed.
- Real preprocessed-data forward smoke test passed with `B=4`, `T=168`, `C=5`; output shape is `(4,168,5)` and all values are finite.

## Simple Baselines (2026-05-18)

- Implemented `src/baselines/linear_interp.py`.
- Implemented `src/baselines/locf.py`.
- Both baselines preserve observed values exactly and handle leading/trailing missing segments.
- Test suite passed for block-missing utilities, ECBIT model, and simple baselines: 14 tests passed.

## PyPOTS Baseline Wrappers (2026-05-18)

- Local `darts` environment does not currently include PyPOTS.
- Implemented lazy wrappers for SAITS and BRITS in `src/baselines/saits_wrapper.py` and `src/baselines/brits_wrapper.py`.
- Added `src/baselines/pypots_common.py` to convert ECBIT filled arrays plus `obs_mask` into PyPOTS `{X: ...}` arrays with NaNs restored at missing positions.
- Tests validate conversion and lazy wrapper construction without requiring PyPOTS import.
- Full SAITS/BRITS training acceptance still requires PyPOTS installation in the remote training environment before those jobs are submitted.

## iTransformer Imputation Baseline (2026-05-18)

- Implemented `src/baselines/itransformer_impute.py`.
- The baseline uses the same variate-token observation stream as ECBIT but removes ERA5 conditioning entirely.
- Added `masked_mse_loss()` to compute loss only on artificially hidden observed positions.
- Full local test suite passed: 20 tests passed.

## ERA5 Direct Baseline (2026-05-18)

- Implemented `src/baselines/era5_direct.py`.
- The baseline estimates per-variable train-set AWS-ERA5 bias on observed positions.
- Missing positions are filled with bias-corrected ERA5; observed AWS values are preserved exactly.
- Full local test suite passed: 23 tests passed.

## Training Framework (2026-05-18)

- Implemented `src/data/impute_dataset.py` for manifest-driven station/window filtering.
- Implemented `src/metrics.py` for per-variable and mean MAE/RMSE on artificial label masks.
- Implemented `src/train_impute.py` for remote neural training with YAML configs, artificial block/MCAR masks, early stopping, checkpoint, and result JSON output.
- Implemented `src/evaluate_impute.py` for neural checkpoints and stateless baselines.
- Local tests validate dataset filtering, artificial masks, metrics, and model config construction.
- Full local test suite passed: 27 tests passed.

## Experiment Configs (2026-05-18)

- Implemented `scripts/generate_experiment_configs.py`.
- Generated Round 1 baseline configs: 162 YAML files.
- Generated Round 2 ECBIT ablation configs: 108 YAML files.
- Generated Round 3 held-out generalization configs: 45 YAML files.
- Added `runner` metadata so stateless/PyPOTS baselines can be routed to evaluation-style jobs and neural models to training jobs.
- Corrected held-out configuration semantics: `test_station_ids` filters only the test dataset, while training/validation remain on main stations.
- Full local test suite passed after config generation: 29 tests passed.

## Paper Draft Scaffold (2026-05-18)

- Created IEEE-style LaTeX scaffold under `paper/`.
- Drafted abstract, Introduction, Related Work, Methodology, experiment protocol, and conclusion placeholder.
- Verified citations through public sources before adding BibTeX entries for AntAWS, ERA5, BRITS, SAITS, and iTransformer.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` succeeds and produces a 3-page PDF.
- The Experiments and Conclusion sections are intentionally provisional until remote GPU results are available.

## Missing Pattern Figure (2026-05-18)

- Implemented `scripts/plot_missing_patterns.py`.
- Generated `paper/figures/fig_missing_patterns.pdf` and `.png`.
- Added the figure to `paper/sections/experiments.tex`.
- Recompiled the paper successfully after adding the figure.

## ECBIT Architecture Figure (2026-05-18)

- Added TikZ architecture figure at `paper/figures/fig_ecbit_arch.tex`.
- Integrated the figure into the Methodology section as Fig. `ecbit-arch`.
- Recompiled the paper successfully after adding TikZ dependencies and the figure.

## Remote Experiment Diagnosis (2026-05-18)

- The GPU server is usable via the `microclimate-experiment-server` credential fallback. The earlier SSH failure was caused by password authentication being disabled by BatchMode-style probing, not by a server or VPN outage.
- Remote `darts` sees 6 RTX 4090 GPUs and imports PyPOTS successfully.
- Initial Round1 failures were implementation/routing issues rather than scientific failures: stateless baselines were sent through the neural trainer, PyPOTS/BRITS code paths were mixed with training logic, and result paths were inconsistent.
- A later Round1 run exposed a scheduling issue: duplicate worker batches launched concurrently and repeatedly started the same LOCF configs, creating many evaluation processes.
- The fix is operational: stateless evaluation now uses single-process DataLoader workers, neural training caps DataLoader workers at 2, and `scripts/worker.py` uses per-run `.lock` files so duplicate launches skip already-running configs.
- `scripts/recover_partial.py` recovered 2 iTransformer `result.json` files from completed `best.pt` checkpoints after interrupted workers.
- Round1 core was relaunched in a `tmux` session named `ecbit_round1_core`.
