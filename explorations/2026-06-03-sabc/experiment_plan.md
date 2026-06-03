# Experiment Plan: Station-Adaptive Bias Correction

## Motivation

The current submission-ready paper shows that, once ERA5 alignment and block-missing curriculum are correct, several ERA5-augmented architectures achieve similar accuracy. The remaining high-risk failure mode is station-specific ERA5-AWS mismatch, especially for difficult held-out stations.

The design memo `06-03-01.md` identifies Station-Adaptive Bias Correction (SABC) as the highest-priority extension because it targets the strongest observed residual source without relying on another generic fusion block.

## Hypothesis

A lightweight station-conditioned correction of ERA5 representations can reduce imputation error at stations with large ERA5-AWS mismatch. The correction should be state-dependent rather than only a static station-month offset.

## Stage 0: Local Feasibility Probe

Run a stateless calibration analysis on observed AWS values:

1. Fit calibrators only on training portions.
2. Evaluate on held-out station test portions.
3. Compare normalized ERA5-AWS residual MAE before and after correction.
4. Report variable-level and station-level improvements.

Calibration probes:

- `none`: no correction.
- `global_mean`: mean AWS-minus-ERA5 residual by variable, fitted on all main-station train periods.
- `target_mean`: mean AWS-minus-ERA5 residual by variable, fitted on the target held-out station train period.
- `target_month`: month-variable residual table fitted on the target held-out train period.
- `target_state_linear`: ridge-linear residual model fitted on the target held-out train period with features `[ERA5_c, sin_doy, cos_doy, sin_hour, cos_hour]`.

This stage requires no neural training and is safe to run locally.

## Stage 0 Decision Gate

Proceed to neural SABC if either condition holds:

- `target_state_linear` reduces held-out mean mismatch by at least 0.02 normalized MAE over `target_month`; or
- `target_month`/`target_state_linear` reduces Mount Sidley or Zhongshan mismatch by at least 0.04 normalized MAE over `global_mean`.

Reject or postpone neural SABC if all adaptive probes improve by less than 0.01 normalized MAE relative to `global_mean`.

## Stage 1: Minimal Neural Layer

Insert SABC between the ERA5 variate-token encoder and gated feature injection:

```text
ERA5 values -> ERA5 encoder -> SABC -> gated feature injection -> decoder
```

Inputs:

- ERA5 variable token `Z_era5[c]`;
- station metadata embedding `e_s`;
- optional historical residual summary vector `r_s`.

Output:

```text
Z_era5_corrected[c] = Z_era5[c] + g_phi(Z_era5[c], e_s, r_s)
```

Minimal implementation:

- two-layer MLP;
- hidden size 64;
- dropout 0.1;
- residual scale initialized near zero.

## Stage 1 Remote Matrix

Run only after Stage 0 is positive:

| Model | Stations | Pattern | Rate | Seeds |
|---|---|---|---|---|
| ECBIT gated baseline reuse | held-out | short/medium/long | 40% | 42/43/44 |
| ECBIT + SABC metadata-only | held-out | short/medium/long | 40% | 42/43/44 |
| ECBIT + SABC residual-summary | held-out | short/medium/long | 40% | 42/43/44 |

Primary metric: held-out normalized MAE by station and variable.

Secondary metrics:

- Mount Sidley MAE;
- Zhongshan MAE;
- q/T/RH physical-consistency residual;
- SABC correction magnitude by variable.

## Stage 1 Decision Gate

Treat SABC as worth manuscript integration only if:

- held-out mean MAE improves by at least 0.015 normalized MAE; and
- at least one of Mount Sidley or Zhongshan improves by at least 0.03 normalized MAE; and
- Nico does not degrade by more than 0.01 normalized MAE.

## Risks

- Target-station adaptation may require labels from historical observed segments, which changes the deployment assumption.
- Static station metadata may be too weak without ERA5 grid elevation or terrain descriptors.
- A neural SABC layer may learn the same correction already captured by station-month ERA5 direct calibration.
- Held-out station count is small; any success must be framed as exploratory unless additional splits are run.
