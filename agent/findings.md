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
