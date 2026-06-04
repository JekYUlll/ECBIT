# Uncertainty and Conformal Prediction Exploration

Status: exploratory, not part of the submission-ready manuscript.

Goal: add quality flags and prediction intervals around deterministic AWS gap fills without changing the ECBIT architecture. The first pass uses split-conformal absolute residual intervals:

1. load an existing trained checkpoint;
2. collect artificial-hidden residuals on the validation split;
3. compute finite-sample conformal quantiles by variable;
4. evaluate interval coverage on the test split.

The implementation is `scripts/analyze_conformal_uncertainty.py`. It is inference-only and must not be interpreted as a new training result.

Decision gate:

- useful if 90% target intervals achieve at least 88% empirical test coverage for the pooled hidden positions;
- operationally useful if interval width is strongly larger for difficult variables/stations and can serve as a quality flag;
- do not add to the paper if coverage is badly under-calibrated or depends on unavailable checkpoints.
