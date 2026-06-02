# Reproducibility Notes

This document maps the main manuscript claims to the retained code and artifacts.

## Environment

The local analysis environment used for preprocessing, aggregation, plotting, and LaTeX compilation is summarized in:

```text
experiments/results/analysis/environment/local_environment.txt
```

The remote GPU training environment is summarized in:

```text
experiments/results/analysis/environment/remote_environment.txt
```

Use `environment.yml` or `requirements.txt` for a close local environment. SAITS and BRITS require `requirements-optional.txt`.

## Data Construction

```bash
python scripts/select_antaws_stations.py --raw-dir data/antaws/raw --output data/station_meta_ecbit.csv --target-total 32 --heldout 5
python src/resample_era5.py --meta-csv data/station_meta_ecbit.csv --raw-dir data/antaws/raw --era5-raw-dir data/era5 --output-dir data/era5_3h
python scripts/validate_era5_3h.py --meta-csv data/station_meta_ecbit.csv --era5-dir data/era5_3h --output data/era5_3h_manifest.csv
python src/preprocess_antaws_impute.py --meta-csv data/station_meta_ecbit.csv --era5-dir data/era5_3h --output-dir data/antaws/processed --manifest-csv data/antaws_impute_manifest.csv --scaler-npz data/antaws_impute_scaler.npz
```

Benchmark diagnostics:

```bash
python scripts/analyze_benchmark_protocol.py
python scripts/analyze_split_overlap.py
python scripts/make_reproducibility_inventory.py
```

## Experiment Configurations

YAML configurations are retained under:

```text
experiments/configs/
```

The major configuration groups are:

- `round1`: interpolation, LOCF, ERA5-direct, SAITS, BRITS, and iTransformer baselines.
- `round2` and `round2_gated`: ECBIT ablations.
- `round3`: held-out station evaluation.
- `fair_era5_baselines`: SAITS+ERA5 and iTransformer+ERA5 controls.
- `followup_blocklen`: outage-length sensitivity.
- `followup_mcar_noera5`: MCAR/no-ERA5 follow-up.
- `revision`: calibration, non-overlap, real-gap, and held-out diagnostics.

## Training and Evaluation

Train neural models:

```bash
python src/train_impute.py --config <config.yaml>
```

Evaluate stateless baselines or trained checkpoints:

```bash
python src/evaluate_impute.py --config <config.yaml> --split test
python src/evaluate_impute.py --config <config.yaml> --split test --checkpoint <best.pt>
```

The manuscript's full neural experiment matrix was run on a GPU server; no model checkpoints are included in this code repository.

## Aggregation

Per-run JSON files are retained under `experiments/results/metrics/` when available. Aggregate them with:

```bash
python scripts/aggregate_results.py --metrics-dir experiments/results/metrics/round1 --config-dir experiments/configs/round1 --prefix round1_core
python scripts/aggregate_results.py --metrics-dir experiments/results/metrics/round2_gated --config-dir experiments/configs/round2_gated --prefix round2_gated_final
python scripts/aggregate_results.py --metrics-dir experiments/results/metrics/round3 --config-dir experiments/configs/round3 --prefix round3_final
```

Additional analysis scripts:

```bash
python scripts/evaluate_era5_direct_calibration.py
python scripts/evaluate_era5_robustness.py
python scripts/evaluate_mcar_on_block.py
python scripts/evaluate_observed_consistency.py
python scripts/analyze_real_gap_plausibility.py
python scripts/analyze_heldout_station_bias.py
```

## Manuscript Tables

```bash
python scripts/make_round1_table.py
python scripts/make_round2_table.py
python scripts/make_round3_table.py
python scripts/make_fair_baseline_table.py
python scripts/make_stat_tests_table.py
python scripts/make_station_month_neural_tests.py
python scripts/make_curriculum_factorial_table.py
python scripts/make_non_overlap_eval_table.py
python scripts/make_raw_unit_tables.py
python scripts/make_station_metadata_table.py
```

## Manuscript Figures

```bash
python scripts/plot_missing_patterns.py
python scripts/plot_main_results.py
python scripts/plot_round1_baselines.py
python scripts/plot_round2_ablations.py
python scripts/plot_followup_analyses.py
python scripts/plot_imputation_case.py
```

## Artifact Inventory

The retained reproducibility artifacts are listed in:

```text
experiments/results/tables/reproducibility_inventory.csv
```

Each row reports an artifact name, path, byte size, short SHA-256 digest when applicable, and a description.
