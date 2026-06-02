# Data Availability

This repository contains code, configuration files, result summaries, generated figures, and selected metadata manifests. It does not redistribute the raw or processed meteorological data needed to train the models.

## External Data Sources

### AntAWS

The AWS observations are based on the AntAWS data product:

Wang, Y. et al. (2023). The AntAWS dataset: a compilation of Antarctic automatic weather station observations. *Earth System Science Data*, 15, 411-429. https://doi.org/10.5194/essd-15-411-2023

Users should download the AntAWS station files from the original data source and follow that product's citation and license requirements.

### ERA5

ERA5 single-level point time series are obtained through the Copernicus Climate Data Store. The alignment script requests station-location point series and converts them into the benchmark variables used in the paper: temperature, relative humidity, wind speed, pressure, and specific humidity.

The local CDS API credentials file, typically `~/.cdsapirc`, must remain outside this repository.

## Local Data Layout

The expected local layout is:

```text
data/antaws/raw/          AntAWS 3-hourly CSV files, not tracked
data/era5/                Downloaded ERA5 NetCDF point time series, not tracked
data/era5_3h/             ERA5 arrays aligned to AntAWS 3-hourly timestamps, not tracked
data/antaws/processed/    Sparse imputation windows, not tracked
```

The repository tracks only lightweight metadata and manifests:

```text
data/station_meta_ecbit.csv
data/era5_3h_manifest.csv
data/antaws_impute_manifest.csv
data/antaws_impute_scaler.npz
```

These files document the benchmark construction and normalization state but do not replace the original data products.

## Why Processed Data Are Not Included

The processed station windows combine AntAWS observations with ERA5-derived variables and can be regenerated from the public data sources. To keep the code repository lightweight and to respect source-data terms, processed `.npz` windows, downloaded ERA5 `.nc` files, and model checkpoints are excluded by `.gitignore`.

## Regeneration

After obtaining AntAWS files and configuring the CDS API, run:

```bash
python scripts/select_antaws_stations.py --raw-dir data/antaws/raw --output data/station_meta_ecbit.csv --target-total 32 --heldout 5
python src/resample_era5.py --meta-csv data/station_meta_ecbit.csv --raw-dir data/antaws/raw --era5-raw-dir data/era5 --output-dir data/era5_3h
python scripts/validate_era5_3h.py --meta-csv data/station_meta_ecbit.csv --era5-dir data/era5_3h --output data/era5_3h_manifest.csv
python src/preprocess_antaws_impute.py --meta-csv data/station_meta_ecbit.csv --era5-dir data/era5_3h --output-dir data/antaws/processed --manifest-csv data/antaws_impute_manifest.csv --scaler-npz data/antaws_impute_scaler.npz
```

The regenerated manifests should match the retained benchmark metadata apart from path strings and any provider-side updates to the downloaded ERA5 service.
