# CHANGELOG

This file records algorithm and experiment-path changes after the submission-ready ECBIT manuscript snapshot. It is intended to keep exploratory changes distinguishable from the Polar Science submission version.

## 2026-06-04 - Parallel Exploration Tracks

Status: exploratory, not manuscript-ready.

Two lower-risk tracks are now separated from the submission snapshot:

- `explorations/2026-06-04-uncertainty/`: inference-only split-conformal residual intervals for retained checkpoints.
- `explorations/2026-06-04-pcr/`: physical-consistency regularization for thermodynamic q consistency.

Uncertainty changes:

- Added `scripts/analyze_conformal_uncertainty.py`.
- The script uses validation residuals to compute finite-sample absolute-residual quantiles and checks test coverage by variable.
- A one-config/one-batch CPU smoke run completed successfully with pooled 90% interval coverage of `0.9055`.
- Full CPU inference is too slow locally; full analysis should run on remote GPU or with an explicit sample limit.

PCR changes:

- Added `src/physical.py` with differentiable specific-humidity and physical-consistency helpers.
- `ImputationWindowDataset` now returns `norm_mean` and `norm_std` in each batch so losses can safely unnormalize meteorological variables.
- `train_impute.py` applies PCR only when `loss.physical_consistency.enabled` is true; existing configs remain behaviorally unchanged.
- Added `scripts/generate_pcr_configs.py`, producing 18 configs under `experiments/configs/exploration_pcr/`.
- Added `scripts/analyze_pcr_results.py` for paired MAE aggregation after remote results complete.
- The 18-run PCR matrix has been launched remotely on GPUs 0, 1, 3, 4, and 5.

## 2026-06-04 - SABC Final Remote Evidence

Status: exploratory, not manuscript-ready.

The metadata-only SABC Stage 1 matrix completed 90/90 remote runs, producing 45 matched gated-baseline/SABC pairs.

Final paired result:

- Overall SABC improvement over matched gated ECBIT: `-0.0039` normalized MAE.
- 95% CI: `[-0.0086, 0.0007]`.
- Current gate decision: do not pass.
- Mount Sidley, Nico, and Zhongshan do not improve under metadata-only SABC.

Interpretation:

- Metadata-only station features are insufficient for a useful SABC claim.
- SABC should not be added to the current manuscript.
- A future SABC v2 would need target-station residual summaries, station-month calibration features, or terrain/grid-elevation mismatch inputs rather than only static metadata.

## 2026-06-04 - SABC Partial Remote Evidence

Status: exploratory, not manuscript-ready.

Remote Stage 1 SABC training is still running. At the latest synchronized checkpoint, 58 of 90 configurations had completed and 13 matched baseline/SABC pairs were available.

Preliminary paired result:

- Overall SABC improvement over matched gated ECBIT: `0.0012` normalized MAE.
- 95% CI: `[-0.0065, 0.0088]`.
- Current gate decision: do not pass yet.
- Butcher Ridge shows a small positive mean difference (`0.0067` normalized MAE), while Mount Sidley is currently negative (`-0.0112` normalized MAE).

Interpretation:

- These partial results do not yet support adding SABC to the paper.
- The evidence should be revisited only after all 90 configurations complete.
- No manuscript claim should be changed based on this partial result.

## 2026-06-03 - Metadata-Only SABC Trainable Path

Status: implemented on branch `exploration/sabc-20260603`.

Algorithm changes:

- Added `ecbit_sabc`, a separate model path from the original `ecbit`.
- Added `StationAdaptiveBiasCorrection`, inserted after the ERA5 variate-token encoder and before gated feature injection.
- The SABC layer conditions ERA5 variable tokens on static station metadata.
- The SABC residual projection is zero-initialized, so the model starts as an identity correction to the ERA5 token stream.
- The original ECBIT gated, concat, no-ERA5, and baseline paths are unchanged.

Station metadata features added to each batch:

- latitude scaled by 90 degrees;
- longitude sine;
- longitude cosine;
- elevation z-score;
- record-length z-score;
- temperature completeness;
- pressure completeness;
- wind-speed completeness;
- relative-humidity completeness.

Training/evaluation changes:

- `ImputationWindowDataset` now returns `station_features` for every window.
- `train_impute.py` and `evaluate_impute.py` pass `station_features` only to models declaring `uses_station_features`.
- Existing models ignore the new batch field.

Experiment configuration changes:

- Added `scripts/generate_sabc_configs.py`.
- Generated 90 Stage 1 configs under `experiments/configs/exploration_sabc/`.
- The matrix compares matched `gated_baseline` and `sabc_metadata` variants over:
  - five held-out stations;
  - short, medium, and long block regimes;
  - 40% artificial missing rate;
  - seeds 42, 43, and 44.

Result-analysis changes:

- Added `scripts/analyze_sabc_results.py`.
- The script writes SABC run tables, summary tables, matched-pair rows, paired tests, and a go/no-go gate report under `experiments/results/exploration_sabc/`.

## 2026-06-03 - SABC Feasibility Probe

Status: local stateless analysis complete.

Motivation:

- The submission-ready paper identifies station-specific ERA5-AWS mismatch as a remaining limitation.
- SABC was selected as the first exploration direction because it directly targets this limitation without changing the paper's benchmark-first narrative.

Local probe:

- Compared ERA5-AWS residual correction modes on held-out station test periods:
  - no correction;
  - global mean residual correction;
  - target-station mean correction;
  - target-station month correction;
  - target-station state-linear correction.

Finding:

- The state-linear target-station probe improved held-out residual MAE over the global mean correction by `0.048` to `0.221` normalized MAE across the five held-out stations.
- This justified testing a learned SABC layer, but did not itself establish a neural-model improvement.

## 2026-06-03 - Submission Snapshot Preserved

Status: completed before exploratory changes.

Preserved state:

- Git tag: `paper-submission-ready-20260603`.
- Local backup: `backups/paper_submission_ready_20260603/`.

Submission-version algorithm:

- ERA5-conditioned block imputation of Antarctic AWS records.
- Variate-token observation encoder.
- Variate-token ERA5 encoder.
- Gated ERA5 feature injection as the controlled ECBIT variant.
- Block-missing curriculum matched to prolonged Antarctic AWS outages.
- Final output preserves observed AWS values by copy-back.

Submission-version interpretation:

- The paper's main claim remains that ERA5 conditioning and failure-mode-matched block masking dominate recovery performance.
- The paper does not claim that the gated fusion block is uniquely superior.
- SABC is not part of the submission-ready manuscript unless the full exploration matrix later provides clear support.
