# CLAUDE.md — ECBIT Session Configuration

## Mandatory Session Start
1. Read `agent/task_plan.md`.
2. Read the latest entries in `agent/progress.md`.
3. Read `agent/findings.md`.
4. Work on one atomic task unless the user explicitly expands scope.

## Mandatory Session End
1. Mark completed task in `agent/task_plan.md`.
2. Append a short summary to `agent/progress.md`.
3. Append discoveries to `agent/findings.md` when applicable.
4. Run the task acceptance check.
5. Commit with the project git convention.

## Fixed Design Decisions
- Block missing simulation: `min_block=6`, `max_block=72` at 3h resolution.
- ERA5 conditioning: variable-level gating from missing-variable masks.
- Observable variables must not be overwritten by ERA5.
- Backbone: iTransformer-style variate-token encoder.
- Loss: missing positions only.
- Normalization: per-variable z-score fit on train split.

## Forbidden Locally
- `python src/train_impute.py`
- Any large-scale neural training or evaluation

Use the remote GPU server for training.

## W&B Convention
- Project: `ecbit-polar`
- Run naming: `{model}_{station_group}_{miss_pattern}_{miss_rate}_s{seed}`
- Metrics: MAE/RMSE per variable (`T`, `RH`, `wspd`, `P`, `q`) plus mean.
