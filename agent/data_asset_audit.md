# ECBIT Data Asset Audit

Date: 2026-05-18

## Local Assets Found

### AntAWS

Path:

```text
/home/horeb/_code/microclimate_demo/data/AntAWS
```

Status:

- Present locally.
- Size: 305 MB.
- File count within 3 levels: 268.
- Main raw station directory appears to be:

```text
/home/horeb/_code/microclimate_demo/data/AntAWS/3_hourly/
```

Example files:

- `Zhongshan_3h.csv`
- `Kunlun_3h.csv`
- `Taishan_3h.csv`
- `Cape Bird_3h.csv`
- `Main characteristics of automatic weather station.csv`

### Existing AntAWS Processed Demo Assets

Path:

```text
/home/horeb/_code/microclimate_demo/data/processed/
```

Status:

- Present locally.
- Includes demo processed splits for Cape Hallett, Zhongshan, Taishan, and Kunlun.
- Useful as format reference, not sufficient for the full ECBIT station-selection experiment.

### ECAFT / IMAU ERA5 Assets

Path:

```text
/home/horeb/_code/ECAFT/data/processed/
```

Status:

- Present locally.
- Contains repaired IMAU AWS station `.npz` and `_era5.npz` files for AWS04, AWS05, AWS06, AWS14, AWS15, AWS16, AWS17, AWS18, AWS19.
- These are scientifically valid but cover IMAU stations, not the full 267-station AntAWS dataset.

## Remote Compute

GPU server:

- Host reachable by ping.
- SSH reachable.
- Conda env: `darts`.
- Python: 3.12.12.
- GPUs: 6 x NVIDIA GeForce RTX 4090, 49140 MiB each.

## Implications

The first ECBIT data task should not assume aligned ERA5 exists for all AntAWS stations. Before implementing full preprocessing, we need to:

1. Parse AntAWS station metadata and determine station coordinates.
2. Select 30-45 stations by completeness and record length.
3. Check whether ERA5 can be reused or must be downloaded/aligned for selected AntAWS stations.
4. Decide whether to symlink/copy AntAWS raw files into `ECBIT/data/antaws/raw/` or keep external paths configurable.
