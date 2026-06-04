# CHANGELOG

This file records algorithm and experiment-path changes after the submission-ready ECBIT manuscript snapshot. It is intended to keep exploratory changes distinguishable from the Polar Science submission version.

## 2026-06-04 - Residual-Aware SABC v2

Status: implemented and locally unit-tested; remote pilot pending.

Motivation:

- Metadata-only SABC v1 failed its gate, suggesting static station metadata alone is not enough to correct held-out ERA5-AWS mismatch.
- The earlier residual probe was positive, so SABC v2 adds train-period residual summary features rather than repeating metadata-only adaptation.

Algorithm changes:

- `scripts/generate_sabc_residual_features.py`: generates train-period ERA5-AWS residual summaries by station, month, and variable.
- `ImputationWindowDataset`: optionally loads `residual_feature_csv` and returns `residual_features` with shape `(n_vars, 8)` for each window.
- `StationAdaptiveBiasCorrection`: optionally projects per-variable residual features and combines them with ERA5 tokens, station metadata tokens, and variable embeddings.
- `ECBITSABC`: remains backward-compatible for metadata-only SABC, while configs with `n_residual_features > 0` declare `uses_residual_features`.
- Training/evaluation dispatch now passes residual features only to models that require them.

Residual features:

- station-level AWS-minus-ERA5 bias, standard deviation, MAE, and scaled log-count;
- station-month AWS-minus-ERA5 bias, standard deviation, MAE, and scaled log-count;
- features are estimated from chronological train windows only.

Experiment path:

- `scripts/generate_sabc_v2_configs.py` writes 27 pilot configs under `experiments/configs/exploration_sabc_v2/`.
- Pilot scope: Mount Sidley, Nico, Zhongshan x short/medium/long block regimes x seeds 42/43/44 at 40% missingness.
- `scripts/analyze_sabc_v2_results.py` compares SABC v2 results against the completed matched gated-baseline rows from `experiments/results/metrics/exploration_sabc/`.

Boundary:

- For held-out stations, residual summaries use target-station chronological train history. This is a target-station historical calibration setting, not strict zero-shot station generalization.
- No SABC v2 training or evaluation result exists yet. Per the current project rule, all training and evaluation must run on the server.

Verification:

- `python -m py_compile` passed for the changed code and new scripts.
- `conda run -n darts python -m pytest -q src/tests` passed with 54 tests.

## 2026-06-04 - Operational Hybrid Baseline

Status: exploratory, closed as negative for endpoint anchoring.

Added a local-code, server-evaluated operational hybrid baseline:

- `src/baselines/era5_hybrid.py`: endpoint residual anchoring for calibrated ERA5 trajectories.
- `scripts/evaluate_operational_hybrid_baseline.py`: evaluates station-month ERA5 direct substitution versus station-month ERA5 plus endpoint anchoring under the same 27 Round 1 block-missing configurations.
- `src/tests/test_era5_hybrid.py`: unit tests for endpoint residual interpolation, single-endpoint fallback, no-endpoint fallback, and shape validation.

Final server-side result:

- Station-month calibrated ERA5 direct: `0.2699` normalized MAE.
- Endpoint-anchored station-month ERA5: `0.3000` normalized MAE.
- Endpoint anchoring helps short blocks (`0.2567` vs `0.2702`) but substantially worsens medium (`0.3037` vs `0.2699`) and long blocks (`0.3396` vs `0.2694`).
- Compared with neural ERA5 variants, endpoint anchoring is clearly behind: SAITS+ERA5 is better by `0.0551` MAE, and ECBIT/iTransformer ERA5 variants are better by about `0.042` MAE.

Interpretation:

- Endpoint anchoring over-constrains long Antarctic AWS outages. A residual estimated at the visible boundary is not a reliable correction across multi-day gaps.
- The strongest simple operational baseline remains station-month calibrated ERA5 direct substitution.
- This result supports the current neural sparse-context interpretation: the neural ERA5-conditioned models are not merely reproducing endpoint continuity.

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
- The 18-run PCR matrix completed remotely on GPUs 0, 1, 3, 4, and 5.
- Final paired MAE result over 9 matched gated-baseline/PCR pairs: mean improvement `0.0004` normalized MAE, 95% CI `[-0.0000, 0.0009]`, p=`0.0589`. This passes the non-degradation gate but is not evidence of a meaningful accuracy gain.
- Added `scripts/evaluate_physical_consistency.py` for checkpoint-based thermodynamic diagnostics. A bounded remote GPU diagnostic with `--max-batches 10` found lower q-consistency residual for PCR than the matched gated baseline: `0.1237` versus `0.1796` normalized q MAE, or `0.0836` versus `0.1214` g/kg. This supports PCR as a physical-consistency diagnostic direction, not as a main MAE-improvement claim.
- A bounded remote GPU conformal run over the 9 retained r40 iTransformer+ERA5 configs with `--max-batches 10` produced pooled coverage `0.8892` for nominal 90% intervals. RH coverage is lower (`0.8650`), so uncertainty needs variable-aware calibration before it can support a manuscript claim.

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
