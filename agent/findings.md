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
- Complete Round1 core aggregation over 108 runs shows iTransformer mean MAE 0.353, ERA5 direct 0.381, linear interpolation 0.395, and LOCF 0.444.
- The pattern-level trade-off remains informative: linear interpolation is best among stateless methods on short gaps, while ERA5 direct is substantially better than interpolation/LOCF on long gaps. iTransformer is strongest overall but still worse than ERA5 direct on the easiest long-gap low-rate setting. This supports ECBIT's conditional-use framing for ERA5.

## Round2 Partial Ablation Snapshot (2026-05-19)

- Round2 is still incomplete, so these are monitoring signals rather than final conclusions.
- Over 26 completed runs, `full` has mean MAE 0.258 over 16 block-missing runs and `no_cross` has mean MAE 0.256 over 5 block-missing runs.
- `no_blockmask` has lower mean MAE over 5 runs, but those runs use MCAR masks and should not be compared directly against block-missing `full/no_cross` runs.
- Active Round2 logs show continued convergence and no new runtime failures.
- Updated 58-run snapshot: `full` mean MAE 0.255 over 26 block-missing runs, `no_cross` 0.256 over 8 block-missing runs, and `no_era5` 0.347 over 3 block-missing runs.
- Current interpretation: ERA5 conditioning appears useful relative to no-ERA5, but the cross-attention mechanism has not yet separated from the no-cross fusion ablation. Wait for the full 108-run Round2 matrix before final claims.

## Round2 Complete Ablation Results (2026-05-19)

- Round2 is complete: 108/108 ablation runs are available locally and aggregated in `experiments/results/tables/round2_core_runs.csv`.
- Block-missing MAE over 27 runs per variant:
  - `no_cross`: 0.2546 +/- 0.0055.
  - `full`: 0.2553 +/- 0.0059.
  - `no_era5`: 0.3429 +/- 0.0238.
- MCAR/no-blockmask runs are much easier and should not be compared directly to block-missing runs: mean MAE 0.1882 +/- 0.0072 over 27 runs.
- Scientific interpretation: ERA5 conditioning is clearly useful relative to no-ERA5, reducing mean MAE by about 25.6% versus the no-ERA5 ablation. The current cross-attention mechanism does not outperform the simpler no-cross fusion variant; any paper claim should emphasize ERA5-conditioned block imputation rather than cross-attention superiority unless later analysis identifies a narrower regime where cross-attention helps.

## Metric Semantics Correction (2026-05-19)

- Diagnosed a critical evaluation bug while checking Round3: neural `train_impute.py` saved the best validation metrics to `result.json`, while stateless baselines used `evaluate_impute.py` and reported test metrics.
- Symptom: early Round3 held-out results were identical across different stations for the same missing pattern and seed, because all runs were reporting main-station validation metrics rather than station-specific held-out test metrics.
- Fix implemented:
  - `train_impute.py` now builds a test loader, reloads the best checkpoint, evaluates test split, and stores metrics under `result["test"]`.
  - `aggregate_results.py` now uses `result["test"]` when present and keeps validation metrics only as auxiliary columns.
  - `recover_partial.py` can overwrite completed neural `result.json` files from `best.pt` checkpoints with true test metrics.
- Corrected partial Round3 test metrics over 17 completed runs are now station-specific: Butcher Ridge mean MAE 0.2893, Mount Sidley 0.3657, Nico 0.1921. This confirms the held-out station filter is functioning when evaluated on the test split.
- Consequence: prior Round1 iTransformer and Round2 ECBIT numbers must be treated as invalid until recovered from checkpoints and re-aggregated. A remote recovery job is running in tmux session `ecbit_recover_test_metrics`.
- Recovery completed for Round1 and Round2. Corrected Round1 test ranking: iTransformer 0.3599, ERA5 direct 0.3811, linear interpolation 0.3955, LOCF 0.4443.
- Corrected Round2 test ablation result: `no_cross` 0.2581 +/- 0.0067, `full` 0.2596 +/- 0.0074, `no_era5` 0.3462 +/- 0.0247, MCAR/no-blockmask 0.1910 +/- 0.0072.
- The corrected conclusion is unchanged in direction: ERA5 conditioning is useful, but the implemented cross-attention block does not beat simpler no-cross fusion.

## Gated Feature Injection Pivot (2026-05-19)

- Current evidence does not support cross-attention as the primary fusion contribution: cross-attention `full` is tied with, and slightly worse than, the no-cross concat fusion ablation.
- The project direction remains viable because the ERA5 information signal is large: no-ERA5 is far worse than either ERA5-conditioned variant.
- Architectural response: introduce gated feature injection as the new primary ECBIT fusion path. It projects ERA5 tokens into the AWS token space and uses a learned vector gate multiplied by the missing-variable mask, so observed variables remain independent of ERA5 while missing variables can draw on reanalysis context.
- New experiment matrix `round2_gated` will test whether gated feature injection improves over concat no-cross. If it does not, the paper claim should shift to: lightweight ERA5 conditioning is sufficient; the value lies in the ERA5 information and block-missing setup rather than a complex fusion mechanism.
