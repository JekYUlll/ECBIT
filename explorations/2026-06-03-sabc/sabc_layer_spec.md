# SABC Layer Specification

## Placement

The Station-Adaptive Bias Correction layer is inserted after the ERA5 variate-token encoder and before gated feature injection:

```text
AWS encoder output       -> Z_aws
ERA5 encoder output      -> Z_era5
Station-adaptive SABC    -> Z_era5_tilde
Gated feature injection  -> fused representation
Decoder                  -> imputed trajectory
```

This placement keeps the submission paper's main conclusion intact: the layer targets station-specific ERA5 bias rather than claiming a generally superior fusion mechanism.

## Inputs

For each station `s` and variable token `c`:

- `Z_era5[c]`: encoded ERA5 token, shape `(B, C, D)`;
- `station_meta[s]`: normalized metadata vector;
- optional `residual_summary[s, c]`: train-period ERA5-AWS residual features.

Implemented metadata fields:

- latitude scaled by 90 degrees;
- longitude sine and cosine;
- station metadata elevation z-score;
- record-length z-score;
- observed completeness for temperature, pressure, wind speed, and relative humidity.

Not yet implemented:

- institution/network embeddings;
- train-period ERA5-AWS residual summaries;
- ERA5 grid elevation minus station elevation.

## Minimal Form

```python
delta = mlp(torch.cat([Z_era5_c, station_embedding, residual_embedding_c], dim=-1))
Z_era5_tilde_c = Z_era5_c + alpha * delta
```

Initialization:

- initialize final MLP layer weights near zero;
- set `alpha = 0.1` initially or learn a sigmoid-scaled residual gate.

Reason: the current ERA5-conditioned models already work. SABC should start as a small correction, not a disruptive replacement of ERA5 tokens.

## Config Sketch

```yaml
model:
  name: ecbit_sabc
  variant: sabc_metadata
  fusion_type: gated
  n_station_features: 9
  sabc:
    hidden_dim: 64
    dropout: 0.1
    residual_scale_init: 0.1
```

## Implementation Boundary

The first implementation should be isolated:

- add a new model variant name, `ecbit_sabc`;
- preserve the existing `ecbit` path unchanged;
- generate a separate config directory, e.g. `experiments/configs/exploration_sabc/`;
- write outputs to `experiments/results/metrics/exploration_sabc/`.

This makes negative results easy to discard without affecting the submission-ready code path.

## Current Implementation

The metadata-only SABC path is implemented in `src/models/ecbit_sabc.py`. The final residual projection is zero-initialized, so the correction starts as an identity mapping and must earn any deviation during training. `ImputationWindowDataset` now attaches a `station_features` tensor to each batch; existing models ignore it, and `ecbit_sabc` consumes it through the `uses_station_features` dispatch in the training and evaluation entry points.
