# ECBIT

Code and reproducibility artifacts for:

> Block-missing imputation of Antarctic automatic weather station records using ERA5 reanalysis conditioning

This repository supports a Polar Science submission on block-missing imputation of sparse Antarctic automatic weather station (AWS) records. The main contribution is a reproducible AntAWS-derived benchmark and controlled evidence that ERA5 conditioning and failure-mode-matched block masking are the dominant factors for recovering prolonged Antarctic AWS outages.

## What Is Included

- ECBIT model implementation and ERA5-conditioned imputation baselines.
- Linear interpolation, LOCF, ERA5-direct, SAITS, BRITS, and iTransformer-style baseline wrappers.
- Station selection, ERA5 point-series alignment, sparse-window preprocessing, mask simulation, training, evaluation, aggregation, and plotting scripts.
- Experiment YAML configurations used for the reported benchmark comparisons.
- Summary result CSV/JSON artifacts, generated manuscript tables, and publication figures.
- The LaTeX manuscript source under `paper/`.

The repository does not redistribute raw AntAWS station files, downloaded ERA5 NetCDF files, processed station windows, model checkpoints, or CDS credentials.

## Repository Layout

```text
src/                         Model, dataset, baseline, training, and evaluation code
scripts/                     Preprocessing, configuration, aggregation, and plotting scripts
experiments/configs/         YAML experiment configurations
experiments/results/         Retained summary metrics, diagnostics, and result tables
data/                        Public manifests and empty placeholders for generated data
paper/                       Manuscript source, tables, and figures
docs/                        Data access and reproducibility notes
```

## Installation

Python 3.12 is recommended. A minimal environment for local preprocessing, analysis, and plotting can be created with:

```bash
conda env create -f environment.yml
conda activate ecbit
```

or with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

SAITS and BRITS baselines require PyPOTS. Install the optional dependencies when reproducing those baselines:

```bash
pip install -r requirements-optional.txt
```

## Data Preparation

Raw inputs must be obtained from their original providers:

- AntAWS 3-hourly station CSV files from the AntAWS data product.
- ERA5 single-level point time series through the Copernicus Climate Data Store.

Place AntAWS 3-hourly CSV files under `data/antaws/raw/`. Configure CDS API credentials outside the repository, for example in `~/.cdsapirc`; do not commit credentials.

The main local data workflow is:

```bash
python scripts/select_antaws_stations.py \
  --raw-dir data/antaws/raw \
  --output data/station_meta_ecbit.csv \
  --target-total 32 \
  --heldout 5

python src/resample_era5.py \
  --meta-csv data/station_meta_ecbit.csv \
  --raw-dir data/antaws/raw \
  --era5-raw-dir data/era5 \
  --output-dir data/era5_3h

python scripts/validate_era5_3h.py \
  --meta-csv data/station_meta_ecbit.csv \
  --era5-dir data/era5_3h \
  --output data/era5_3h_manifest.csv

python src/preprocess_antaws_impute.py \
  --meta-csv data/station_meta_ecbit.csv \
  --era5-dir data/era5_3h \
  --output-dir data/antaws/processed \
  --manifest-csv data/antaws_impute_manifest.csv \
  --scaler-npz data/antaws_impute_scaler.npz
```

See [docs/DATA.md](docs/DATA.md) for data-source, licensing, and non-redistribution notes.

## Running Experiments

Stateless baselines can be evaluated directly:

```bash
python src/evaluate_impute.py \
  --config experiments/configs/round1/linear_interp_short_r40_s42.yaml \
  --split test
```

Neural models are trained with:

```bash
python src/train_impute.py \
  --config experiments/configs/round2/ecbit_full_short_r40_s42.yaml
```

Training was performed on a remote GPU server. Local CPU execution is appropriate for preprocessing, stateless baselines, aggregation, plotting, and tests, but not for full neural experiment batches.

## Reproducing Tables and Figures

The retained summary artifacts can regenerate the manuscript tables and figures without rerunning neural training:

```bash
python scripts/make_round1_table.py
python scripts/make_round2_table.py
python scripts/make_round3_table.py
python scripts/make_station_month_neural_tests.py
python scripts/make_raw_unit_tables.py

python scripts/plot_main_results.py
python scripts/plot_round1_baselines.py
python scripts/plot_round2_ablations.py
python scripts/plot_followup_analyses.py
python scripts/plot_imputation_case.py
```

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for a fuller command map.

## Tests

Run the lightweight unit tests with:

```bash
python -m pytest -q src/tests
```

The tests cover mask simulation, baseline helpers, ERA5-direct substitution, model shape contracts, config generation, and the training/evaluation framework.

## Citation

If you use this repository, cite the accompanying paper and the code repository. A machine-readable citation file is provided in [CITATION.cff](CITATION.cff).

## License

This code is released under the MIT License. See [LICENSE](LICENSE).
