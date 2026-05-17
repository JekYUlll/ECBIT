# ECBIT

ERA5-Conditioned Block Imputation Transformer for sparse Antarctic weather station time series.

## Research Goal

ECBIT studies block-wise imputation for Antarctic AWS meteorological variables under realistic polar sensor outages. The core hypothesis is that ERA5 should be used as a conditional prior for missing variables, not as an always-on parallel stream.

## Motivation

The preceding ECAFT project showed that naive ERA5-AWS cross-attention is not enough: ERA5 meteorological variables are often redundant with AWS observations, while ERA5 relative humidity can be biased. ECBIT changes the problem setting from forecasting to missing-data imputation and constrains ERA5 influence through variable-level missingness gates.

## Current Status

Phase 0 initialization is in progress. See:

- `PLAN.md`
- `agent/task_plan.md`
- `agent/findings.md`
- `agent/progress.md`

## Compute Policy

Preprocessing and plotting run locally. Model training runs only on the remote GPU server.
