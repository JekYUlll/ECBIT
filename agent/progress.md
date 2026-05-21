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

## Session: 2026-05-19

### Phase 3 · Experiments
- Checked remote experiment status: Round1 core remains complete at 108/108.
- Round2 ablations are actively running on GPU1-GPU5; GPU0 is occupied by another user's process and is intentionally avoided.
- Synced 26 completed Round2 metrics locally and generated `experiments/results/tables/round2_partial_runs.csv` plus `round2_partial_summary.csv`.
- Partial Round2 status: `full` 16 runs, `no_cross` 5 runs, `no_blockmask` 5 runs. `no_blockmask` uses MCAR masks, so it is not directly comparable to block-missing `full/no_cross` results.
- No new remote Traceback, OOM, or RuntimeError found in active Round2 logs.
- Continued monitoring: Round2 advanced to 58/108 completed runs with no active-log failures.
- Synced and regenerated `round2_partial` tables for 58 runs.
- Current block-missing ablation signal: `full` mean MAE 0.255 over 26 runs, `no_cross` 0.256 over 8 runs, `no_era5` 0.347 over 3 runs. Full vs no-cross remains too close to call, but no-ERA5 is clearly worse in the completed subset.
- Added `scripts/plot_round1_baselines.py` and generated `paper/figures/fig_round1_baselines.pdf/.png`.
- Inserted the Round1 baseline comparison figure into `paper/sections/experiments.tex`.
- Recompiled the paper successfully; output is currently 4 pages.
- Added `scripts/make_round1_table.py` and generated `paper/tables/tab_round1_baselines.tex`.
- Inserted the Round1 core baseline table into the Experiments section and recompiled successfully.
- Detected a worker cleanup bug after Round2 was nearly complete: `scripts/worker.py` still referenced the removed `gpu_id` variable in the final DONE print. The bug occurs after results are written, but it prevents clean worker termination.
- Fixed the final worker status print to use `worker_id/world_size/cuda_id`, synced it to the remote server, and launched `ecbit_round2_remaining` to cover the four remaining Round2 configs.
- Started Round3 held-out station generalization on idle GPU5 in tmux session `ecbit_round3_gpu5` while the final four Round2 jobs continue on GPU1-GPU4.
- Added `scripts/plot_round2_ablations.py` and generated a preliminary `paper/figures/fig_round2_ablations.pdf/.png` from the current partial Round2 table. This script is ready to rerun when Round2 reaches 108/108.
- Added `scripts/make_round2_table.py` and generated a preliminary `paper/tables/tab_round2_ablations.tex`; keep it out of the manuscript body until Round2 is complete.
- Confirmed Round2 is complete locally: 108/108 result files, with regenerated `round2_core_runs.csv`, `round2_core_summary.csv`, `fig_round2_ablations`, and `tab_round2_ablations`.
- Complete Round2 result: block-missing `no_cross` MAE 0.2546, `full` MAE 0.2553, and `no_era5` MAE 0.3429. ERA5 conditioning is strongly supported; the current cross-attention variant is not better than no-cross fusion.
- Inserted the Round2 ablation figure and table into the Experiments section and updated the implementation-status text.
- Launched additional Round3 held-out station workers on remote GPU1-GPU4 in tmux session `ecbit_round3_gpus1_4`, while the original GPU5 worker continues.
- Checked Round3 after multi-GPU launch: remote progress reached 17/45, with no Traceback/OOM/Killed in Round3 logs.
- Diagnosed a critical neural evaluation bug: `train_impute.py` wrote validation metrics to `result.json`, so neural Round1/Round2/Round3 aggregates were not test metrics. Stateless baselines were already test metrics.
- Fixed `train_impute.py` to evaluate the best checkpoint on the configured test split, fixed `aggregate_results.py` to prefer `result["test"]`, and enhanced `recover_partial.py` to overwrite completed neural runs from `best.pt`.
- Local tests passed after the fix: `/home/horeb/miniconda3/bin/conda run -n darts pytest -q src/tests` -> 30 passed.
- Synced the fix to the remote server and verified remote targeted tests pass: 6 passed.
- Recovered 17 completed Round3 runs on CPU with true held-out test metrics and re-synced them locally. Corrected partial Round3 mean MAE by station: Butcher Ridge 0.2893, Mount Sidley 0.3657, Nico 0.1921.
- Started remote tmux session `ecbit_recover_test_metrics` to recover Round1 neural and Round2 ECBIT result files from checkpoints using true test metrics.
- Completed remote recovery for Round1 iTransformer and Round2 ECBIT runs; synced corrected result JSON files locally.
- Regenerated Round1 and Round2 aggregate CSVs, tables, and figures using test metrics.
- Corrected Round1 test ranking: iTransformer 0.3599, ERA5 direct 0.3811, linear interpolation 0.3955, LOCF 0.4443.
- Corrected Round2 block-missing test MAE: `no_cross` 0.2581, `full` 0.2596, `no_era5` 0.3462. The prior qualitative conclusion still holds: ERA5 helps; cross-attention is tied with no-cross.
- Reframed the architecture response to the Round2 result: keep ERA5 conditioning as the core contribution, but replace the new primary ECBIT fusion path with gated feature injection for the next ablation round.
- Implemented `GatedFeatureInjection` in `src/models/ecbit.py` with missing-variable gating, retained `cross_attn` for backward-compatible old configurations, and kept `concat` as the explicit no-cross lightweight fusion baseline.
- Added `fusion_type` support to `train_impute.py` and generated a new `experiments/configs/round2_gated/` matrix with 108 runs targeting `experiments/results/metrics/round2_gated/`.
- Local verification after gated injection implementation: `/home/horeb/miniconda3/bin/conda run -n darts pytest -q src/tests` -> 31 passed.
- Synced gated-injection code/configs to the remote server and verified remote targeted tests pass: 7 passed.
- Launched remote tmux session `ecbit_round2_gated_gpus1_5` on GPU1-GPU5 for the 108-run gated ablation matrix. Initial status: all five workers entered their first `ecbit_full_*` configs with no immediate Traceback/OOM/Killed errors.

## Session: 2026-05-20

### Phase 3 · Experiments
- Checked remote experiments: `round2_gated` is running in tmux session `ecbit_round2_gated_gpus1_5`; Round3 continues in `ecbit_round3_gpu5` and `ecbit_round3_gpus1_4`.
- Remote status at check: `round2_gated` 5/108 complete, Round3 23/45 complete, no Traceback/OOM/Killed entries in active logs.
- Synced 5 gated-injection results and generated `experiments/results/tables/round2_gated_partial_runs.csv` plus `round2_gated_partial_summary.csv`.
- Smoke signal for gated injection is too early to conclude: first 5 long-gap full runs have mean MAE 0.2673, about +0.0015 versus matched old no-cross rows.
- Detected that 4 newly completed Round3 results did not yet contain `test` metrics because some workers were started before the metric-semantics patch. Ran remote `recover_partial.py --metrics-dir experiments/results/metrics/round3 --overwrite --completed-only --device cpu`; Round3 now has 23/23 completed result files with `test` metrics.
- Re-synced and regenerated `round3_partial` tables. Current held-out station mean MAE: Butcher Ridge 0.2893 over 9 runs, Mount Sidley 0.3500 over 9 runs, Nico 0.1936 over 5 runs.
- Checked remote experiments again: `round2_gated` advanced to 14/108 and Round3 advanced to 31/45. Active logs still show no Traceback/OOM/Killed.
- Recovered newly completed Round3 checkpoints so 31/31 completed Round3 result files contain test metrics.
- Synced and regenerated `round2_gated_partial` and `round3_partial` tables. Current gated full runs cover long and medium patterns only; matched against old no-cross rows, the overall MAE delta is approximately +0.00003, effectively tied at this early stage.
- Current Round3 held-out station mean MAE: Butcher Ridge 0.2893 over 9 runs, Mount Sidley 0.3500 over 9 runs, Nico 0.1775 over 8 runs, Sabrina 0.2525 over 5 runs.
- Investigated slow `round2_gated` throughput. Diagnosis: CPU is not the bottleneck (`load average` about 10 on a 224-thread server, memory abundant); each active training job uses only about two DataLoader workers because `train_impute.py` caps workers at 2. The main slowdown is GPU oversubscription: Round3 and gated workers are both running on GPU1-GPU5, so each GPU has two training processes. GPUs are highly utilized but each process is small (~536 MiB) and shares a 150W power-capped 4090.
- Launched an extra gated worker on currently idle GPU0 in tmux session `ecbit_round2_gated_gpu0_extra` using `worker_id=5`, `world_size=6`, and lock-file protection. It entered `ecbit_full_short_r20_s44` without immediate errors.
- Synced latest results: `round2_gated` is now 15/108 and Round3 is 32/45. Round3 32/32 completed results have test metrics after recovery.
- Updated partial signal: gated full results remain tied with old no-cross on matched rows, with overall MAE delta about -0.00010 across 15 completed full runs.

## Session: 2026-05-21

### Phase 3 · Experiments
- Checked remote ECBIT experiments at 02:34 CST: `round2_gated` has 53/108 completed results and Round3 held-out has 42/45 completed results. All completed result files contain `test` metrics.
- Active log scan found no Traceback, RuntimeError, CUDA OOM, Killed, or Error lines in the `round2_gated` and Round3 logs.
- Diagnosed the idle GPU0 state: `ecbit_round2_gated_gpu0_extra` completed its 15 assigned configs successfully (`ok=15 fail=0 skipped=0`) and exited normally. It was not a crash or CPU-utilization problem.
- Launched a GPU0 catch-up worker in tmux session `ecbit_round2_gated_gpu0_catchup` with lock-file protection. It skipped locked runs and entered remaining `round2_gated` configs.
- Synced latest remote metrics locally and regenerated `round2_gated_partial` and `round3_partial` aggregate tables.
- Current gated ablation snapshot over 53 completed runs: `full` mean MAE 0.2589 over 24 runs, `no_cross` mean MAE 0.2586 over 11 runs, `no_era5` mean MAE 0.3484 over 4 runs, and MCAR/no-blockmask mean MAE 0.1902 over 14 runs.
- Matched gated-full vs old no-cross rows: 24 matched runs, overall delta -0.00018 MAE; by pattern, long +0.00133, medium -0.00146, short -0.00051. Within the new gated matrix, matched full vs concat/no-cross delta is -0.00017 MAE over 11 rows.
- Current Round3 held-out station means over 42 runs: Butcher Ridge 0.2893, Mount Sidley 0.3500, Nico 0.1732, Sabrina 0.2458, Zhongshan 0.3243. Zhongshan is still partial with 6/9 runs.

### Planning Update
- Added a consolidated idea/validation summary to `agent/findings.md`.
- Current working thesis recorded: the publishable contribution is ERA5-conditioned block-missing AWS imputation, not cross-attention superiority.
- The result narrative is now: ERA5 conditioning is strongly validated; cross-attention and gated injection are tied with no-cross lightweight fusion; station-level held-out difficulty must be reported explicitly.

### Final Round2-Gated and Round3 Check
- Remote check at 17:39 CST found `round2_gated` complete at 108/108 and Round3 complete at 45/45. All completed files contain `test` metrics.
- All ECBIT tmux experiment sessions have exited. Active log scan found no Traceback, RuntimeError, CUDA OOM, Killed, or Error lines for the gated and held-out logs.
- Synced full remote metrics locally and generated `round2_gated_final_*`, `round3_final_*`, and refreshed partial aggregate CSVs.
- Final gated ablation result: gated injection MAE 0.2577, concat/no-cross MAE 0.2580, no-ERA5 MAE 0.3457. Gated injection is tied with no-cross; removing ERA5 causes a 34.14% relative MAE degradation.
- Final Round3 held-out station means: Nico 0.1732, Sabrina 0.2458, Butcher Ridge 0.2893, Zhongshan 0.3073, Mount Sidley 0.3500.
- Updated the paper experiments/conclusion sections, regenerated the Round2 ablation figure/table from `round2_gated_final`, added a Round3 held-out table, and verified `latexmk -pdf` succeeds with a 4-page PDF.
