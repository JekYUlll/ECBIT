# 2026-06-03 SABC Exploration

This module is intentionally separate from the submission-ready manuscript and public release code path.

Goal: explore whether a station-adaptive ERA5 bias-correction layer is worth adding to ECBIT after the current Polar Science submission version is backed up.

Source design memo: `agent/06-03-01.md` (local planning file, not part of the public repository).

## Scope

The first exploration focuses on Station-Adaptive Bias Correction (SABC):

- diagnose held-out ERA5-AWS residual structure;
- test simple station-adaptive calibration probes without neural training;
- implement a minimal metadata-only neural SABC layer between the ERA5 encoder and gated feature injection;
- define a go/no-go gate for remote GPU experiments.

The module does not modify the submission manuscript. The trainable implementation is registered as a separate `ecbit_sabc` model name so the original `ecbit` path remains unchanged.

## Contents

```text
experiment_plan.md                  Experiment design and decision gates
sabc_layer_spec.md                  Neural layer interface and implementation boundary
scripts/analyze_sabc_feasibility.py Local stateless feasibility probe
../../scripts/generate_sabc_configs.py
                                    Repository-level generator for remote YAML configs
results/                            Generated local analysis artifacts
```

## Backup Boundary

The submission-ready manuscript snapshot is backed up by:

- Git tag: `paper-submission-ready-20260603`
- Local backup: `backups/paper_submission_ready_20260603/`

The local backup directory is ignored by Git.

## First Local Probe

Run:

```bash
conda run -n darts python explorations/2026-06-03-sabc/scripts/analyze_sabc_feasibility.py
```

The script compares ERA5-AWS normalized residual MAE on held-out station test periods under increasingly adaptive calibration probes:

- `none`: aligned ERA5 without additional correction;
- `global_mean`: main-station train mean bias by variable;
- `target_mean`: target-station train mean bias by variable;
- `target_month`: target-station train month-variable bias with fallback;
- `target_state_linear`: target-station train ridge-linear residual model using ERA5 state and time features.

These probes are not a final model result. They test whether station-specific residual structure is strong enough to justify a learned SABC layer.

## Trainable SABC Path

Implemented code paths:

- `src/models/ecbit_sabc.py`: `StationAdaptiveBiasCorrection` and `ECBITSABC`;
- `src/data/impute_dataset.py`: 9-dimensional station metadata features attached to every batch;
- `src/train_impute.py` and `src/evaluate_impute.py`: conditional station-feature forwarding for models that declare `uses_station_features`;
- `scripts/generate_sabc_configs.py`: isolated remote config generator.

Generate the Stage 1 metadata-only remote matrix:

```bash
conda run -n darts python scripts/generate_sabc_configs.py
```

This writes 90 YAML files under `experiments/configs/exploration_sabc/`: matched gated ECBIT baselines and metadata-only SABC variants over five held-out stations, three block regimes, and three random seeds. Outputs are routed to `experiments/results/metrics/exploration_sabc/`.
