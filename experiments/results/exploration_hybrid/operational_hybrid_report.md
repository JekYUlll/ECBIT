# Operational Hybrid Baseline Report

Station-month ERA5 MAE: 0.2699.
Endpoint-anchored station-month ERA5 MAE: 0.3000.
Gap from endpoint-anchored hybrid to best neural ERA5 method: 0.0551.
Gate decision: clearly behind the best neural ERA5 variants.

## Summary

| mode | mae_mean | mae_std | rmse_mean | rmse_std | n |
| --- | --- | --- | --- | --- | --- |
| station_month_bias | 0.2699 | 0.0006 | 0.3917 | 0.0012 | 27 |
| endpoint_anchor | 0.3000 | 0.0353 | 0.4550 | 0.0490 | 27 |

## Pattern Summary

| mode | pattern | mae_mean | rmse_mean | n |
| --- | --- | --- | --- | --- |
| endpoint_anchor | long | 0.3396 | 0.5100 | 9 |
| endpoint_anchor | medium | 0.3037 | 0.4600 | 9 |
| endpoint_anchor | short | 0.2567 | 0.3950 | 9 |
| station_month_bias | long | 0.2694 | 0.3914 | 9 |
| station_month_bias | medium | 0.2699 | 0.3920 | 9 |
| station_month_bias | short | 0.2702 | 0.3918 | 9 |

## Paired Test: Station-Month vs Endpoint Anchor

| comparison | group | n | mean_diff | ci95_low | ci95_high | t | p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| station_month_minus_endpoint_anchor | overall | 27 | -0.0301 | -0.0442 | -0.0160 | -4.3878 | 0.0002 |
| station_month_minus_endpoint_anchor | pattern:long | 9 | -0.0702 | -0.0726 | -0.0677 | -66.5478 | 0.0000 |
| station_month_minus_endpoint_anchor | pattern:medium | 9 | -0.0338 | -0.0402 | -0.0274 | -12.1502 | 0.0000 |
| station_month_minus_endpoint_anchor | pattern:short | 9 | 0.0136 | 0.0064 | 0.0208 | 4.3518 | 0.0024 |

## Paired Tests Against Neural ERA5 Variants

| comparison | method | method_mae | endpoint_anchor_mae | n | mean_diff | ci95_low | ci95_high | t | p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAITS+ERA5 minus endpoint_anchor | SAITS+ERA5 | 0.2449 | 0.3000 | 27 | -0.0551 | -0.0716 | -0.0385 | -6.8522 | 0.0000 |
| ECBIT gated minus endpoint_anchor | ECBIT gated | 0.2577 | 0.3000 | 27 | -0.0423 | -0.0535 | -0.0310 | -7.7426 | 0.0000 |
| ECBIT concat minus endpoint_anchor | ECBIT concat | 0.2580 | 0.3000 | 27 | -0.0420 | -0.0537 | -0.0303 | -7.3760 | 0.0000 |
| iTransformer+ERA5 minus endpoint_anchor | iTransformer+ERA5 | 0.2581 | 0.3000 | 27 | -0.0419 | -0.0541 | -0.0296 | -7.0472 | 0.0000 |
