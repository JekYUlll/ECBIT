# ECBIT Findings

## Starting Context

ECBIT starts from the archived ECAFT diagnosis:

- Real ERA5 data has now been validated in the parent ECAFT project.
- Naive ERA5-AWS cross-attention did not outperform single-stream PatchTST.
- ERA5 and AWS share highly correlated T, pressure, and q signals.
- ERA5 RH is biased and weakly correlated with AWS RH at several Antarctic stations.
- The new design should condition on ERA5 only where observations are missing.

## Initial Hypothesis

Block-wise imputation is a better fit than forecasting for ERA5 conditioning:

- Missing variables require external priors.
- Observed variables should remain anchored to AWS observations.
- ERA5 bias can be limited with variable-level missingness gates.
- Real polar AWS outages are block structured rather than MCAR point missing.

## Open Questions

- Which AntAWS stations satisfy completeness and record-length criteria?
- Are aligned ERA5 files available for the selected AntAWS stations, or only for IMAU stations?
- Does ERA5 conditioning help most for long block gaps and high-missingness regimes?
- Which variables benefit from ERA5, and where does RH bias hurt?

## Data Asset Audit (2026-05-18)

- AntAWS raw 3h CSV files exist locally at `/home/horeb/_code/microclimate_demo/data/AntAWS/3_hourly/`.
- AntAWS asset count: 268 files within 3 directory levels, total size about 305 MB.
- Existing processed AntAWS demo splits exist for Cape Hallett, Zhongshan, Taishan, and Kunlun under `/home/horeb/_code/microclimate_demo/data/processed/`.
- Repaired ECAFT ERA5 files exist only for 9 IMAU stations in `/home/horeb/_code/ECAFT/data/processed/`.
- Full AntAWS station-level ERA5 alignment is not yet confirmed; station selection must happen before ERA5 download/alignment scope is fixed.
- Remote GPU server is reachable; `darts` environment has Python 3.12.12 and 6 x RTX 4090.
