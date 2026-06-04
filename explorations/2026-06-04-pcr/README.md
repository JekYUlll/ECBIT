# Physical Consistency Regularization Exploration

Status: exploratory, not part of the submission-ready manuscript.

Goal: test whether a thermodynamic self-consistency term improves or regularizes imputed humidity. The regularizer penalizes disagreement between predicted specific humidity and the value implied by predicted temperature, relative humidity, and pressure using the same Bolton-style formula used during preprocessing.

Implementation:

- `src/physical.py` contains differentiable humidity and PCR loss helpers;
- `src/train_impute.py` applies the loss only when `loss.physical_consistency.enabled` is true;
- `scripts/generate_pcr_configs.py` writes an isolated 18-run matrix under `experiments/configs/exploration_pcr/`.

Matrix:

- model: gated ECBIT baseline versus gated ECBIT + PCR;
- regimes: short, medium, long block missing;
- missing rate: 40%;
- seeds: 42, 43, 44.

Decision gate:

- useful if physical inconsistency decreases without degrading mean MAE by more than 0.003 normalized units;
- stronger if q or RH MAE improves while temperature and pressure do not degrade;
- reject for the current manuscript if it mainly shifts humidity consistency while worsening the headline imputation metric.
