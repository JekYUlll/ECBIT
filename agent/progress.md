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
- Implemented `src/utils/block_missing.py` with block masks constrained to real observed positions, MCAR ablation masks, apply-mask helper, artificial-label loss mask, and variable-level gate helper.
- Added package markers and `src/tests/test_block_missing.py`.
- Test result: `/home/horeb/miniconda3/bin/conda run -n darts pytest -q src/tests/test_block_missing.py` -> 5 passed.
- Phase 1 data engineering is complete; next phase is model implementation.
- Implemented `src/models/ecbit.py` and `src/tests/test_ecbit.py`.
- Test result: `/home/horeb/miniconda3/bin/conda run -n darts pytest -q src/tests/test_ecbit.py src/tests/test_block_missing.py` -> 10 passed.
- Real-data CPU forward smoke test passed using `data/antaws/processed/aws06_impute.npz`: `B=4`, `T=168`, `C=5`, finite `(4,168,5)` output.
- Implemented simple baselines: `src/baselines/linear_interp.py` and `src/baselines/locf.py`.
- Added `src/tests/test_baselines.py`.
- Test result: `/home/horeb/miniconda3/bin/conda run -n darts pytest -q src/tests/test_baselines.py src/tests/test_ecbit.py src/tests/test_block_missing.py` -> 14 passed.
- Checked PyPOTS availability: local `darts` environment is missing `pypots`.
- Implemented lazy PyPOTS wrappers for SAITS and BRITS plus ECBIT-to-PyPOTS dataset conversion.
- Added `src/tests/test_pypots_wrappers.py`.
- Test result: `/home/horeb/miniconda3/bin/conda run -n darts pytest -q src/tests/test_pypots_wrappers.py src/tests/test_baselines.py src/tests/test_ecbit.py src/tests/test_block_missing.py` -> 17 passed.
- Implemented `src/baselines/itransformer_impute.py` and `masked_mse_loss()`.
- Added `src/tests/test_itransformer_impute.py`.
- Test result: `/home/horeb/miniconda3/bin/conda run -n darts pytest -q src/tests` -> 20 passed.
- Implemented `src/baselines/era5_direct.py` with train-set bias correction.
- Added `src/tests/test_era5_direct.py`.
- Test result: `/home/horeb/miniconda3/bin/conda run -n darts pytest -q src/tests` -> 23 passed.
- Implemented training/evaluation framework: `src/data/impute_dataset.py`, `src/metrics.py`, `src/train_impute.py`, and `src/evaluate_impute.py`.
- Added `src/tests/test_training_framework.py`.
- Test result: `/home/horeb/miniconda3/bin/conda run -n darts pytest -q src/tests` -> 27 passed.
- Implemented `scripts/generate_experiment_configs.py` and `src/tests/test_generate_configs.py`.
- Fixed held-out generalization filtering by adding split-specific station IDs (`train_station_ids`, `val_station_ids`, `test_station_ids`) to the data-loading path.
- Generated experiment configs: Round 1 = 162, Round 2 = 108, Round 3 = 45.
- Test result after config generation: `/home/horeb/miniconda3/bin/conda run -n darts pytest -q src/tests` -> 29 passed.
- Remote check: server `192.168.10.47` responds to ping, but SSH BatchMode fails with `Permission denied (publickey,password)` and local `SSHPASS` is not set, so remote smoke submission is blocked by authentication rather than server availability.
- Used parallel paper-writing time to create `paper/main.tex`, section files, and `paper/references.bib`.
- Verified core citations via public sources before adding BibTeX entries.
- Compile result: `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` succeeds and produces a 3-page PDF with only underfull-box warnings.
- Implemented `scripts/plot_missing_patterns.py` and generated `paper/figures/fig_missing_patterns.pdf/.png`.
- Added Fig. missing-patterns to the Experiments section.
- First compile attempt after plotting failed because `latexmk` was run from the repository root and could not find `main.tex`; reran from `paper/` and compilation succeeded.
- Added TikZ ECBIT architecture figure at `paper/figures/fig_ecbit_arch.tex`.
- Integrated the architecture figure into `paper/sections/methodology.tex`.
- Compile result after architecture figure: `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` succeeds.

## Blocked Issues
| Timestamp | Issue | Status | Action |
|-----------|-------|--------|--------|
| 2026-05-18 | Full AntAWS station-level ERA5 alignment not yet confirmed | resolved | Downloaded and validated ERA5 3h files for all 32 selected AntAWS stations |
| 2026-05-18 | `python` missing from PATH; system `python3` lacks NumPy | resolved | Use `/home/horeb/miniconda3/bin/conda run -n darts python` for local data scripts |
| 2026-05-18 | `src` not importable during pytest collection | resolved | Added `src/__init__.py`, `src/utils/__init__.py`, and `src/tests/__init__.py` |
| 2026-05-18 | PyPOTS missing from local `darts` environment | open | Wrappers use lazy import; verify/install dependency before remote SAITS/BRITS jobs |
| 2026-05-18 | Remote SSH authentication unavailable | resolved | Verified skill credential fallback; server login, GPUs, conda `darts`, and PyPOTS import work |
| 2026-05-18 | `latexmk` run from wrong directory after figure generation | resolved | Reran `latexmk` from `paper/`; compile succeeded |

### Remote Round1 Recovery
- Verified that `.claude/skills/microclimate-experiment-server` can use the server normally when its credential fallback is followed.
- Diagnosed the apparent server problem as an SSH probing issue plus duplicated remote worker launches, not a GPU-server outage.
- Stopped 56 duplicate remote worker/evaluation processes that were repeatedly launching the same LOCF configs.
- Added per-run lock files to `scripts/worker.py` and capped DataLoader workers to prevent file descriptor exhaustion.
- Added `scripts/recover_partial.py`; recovered 2 completed iTransformer results from `best.pt`.
- Test result after local scheduler fixes: `/home/horeb/miniconda3/bin/conda run -n darts pytest -q src/tests` -> 29 passed.
- Relaunched Round1 core on the remote server in tmux session `ecbit_round1_core` with 5 locked workers.
- Synced 85 completed Round1 core metrics back locally and generated `experiments/results/tables/round1_partial_runs.csv` plus `round1_partial_summary.csv`.
- Partial Round1 signal: ERA5 direct is competitive for long gaps, linear interpolation is strongest for short gaps, and iTransformer is still incomplete so no final neural-baseline conclusion yet.
- Started Round2 ECBIT ablation worker on idle GPU5 in tmux session `ecbit_round2_gpu5`.
- Updated `scripts/worker.py` to support configurable `--world-size` and `--cuda-id` for cleaner future single-GPU or non-5-way batches.
- Checked remote after Round1 core completion: 108/108 core results are present.
- Synced complete Round1 core metrics locally and generated `experiments/results/tables/round1_core_runs.csv` and `round1_core_summary.csv`.
- Complete Round1 baseline ranking by mean MAE: iTransformer 0.353, ERA5 direct 0.381, linear interpolation 0.395, LOCF 0.444.
- Launched four additional Round2 workers on GPU1-GPU4 while avoiding GPU0, which is occupied by another user. Round2 now uses GPU1-GPU5.
