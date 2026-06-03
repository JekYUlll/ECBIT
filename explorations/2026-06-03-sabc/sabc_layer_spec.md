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

Candidate metadata fields:

- station elevation;
- latitude and longitude;
- institution/network indicator;
- train-period ERA5-AWS mean residual by variable;
- train-period ERA5-AWS residual standard deviation by variable;
- optional ERA5 grid elevation minus station elevation if available later.

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
  name: ecbit
  use_era5: true
  fusion_type: gated
  sabc:
    enabled: true
    hidden_dim: 64
    dropout: 0.1
    use_station_meta: true
    use_residual_summary: true
    residual_scale_init: 0.1
```

## Implementation Boundary

The first implementation should be isolated:

- add a new model variant name, e.g. `ecbit_sabc`;
- preserve the existing `ecbit` path unchanged;
- generate a separate config directory, e.g. `experiments/configs/exploration_sabc/`;
- write outputs to `experiments/results/metrics/exploration_sabc/`.

This makes negative results easy to discard without affecting the submission-ready code path.
