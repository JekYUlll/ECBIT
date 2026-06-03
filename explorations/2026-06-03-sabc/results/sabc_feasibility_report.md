# SABC Feasibility Probe

This local analysis evaluates whether held-out stations contain station-specific ERA5-AWS residual structure that can be reduced by simple train-period adaptation.

## Station-Level Summary

| station_id | station | global_mean | none | target_mean | target_month | target_state_linear | global_mean_gain_vs_none | global_mean_pct_gain_vs_none | target_mean_gain_vs_none | target_mean_pct_gain_vs_none | target_month_gain_vs_none | target_month_pct_gain_vs_none | target_state_linear_gain_vs_none | target_state_linear_pct_gain_vs_none | target_mean_gain_vs_global | target_month_gain_vs_global | target_state_linear_gain_vs_global |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| butcher_ridge | Butcher Ridge | 0.3145 | 0.3593 | 0.2672 | 0.2403 | 0.2183 | 0.0449 | 12.4848 | 0.0921 | 25.6279 | 0.1191 | 33.1415 | 0.1411 | 39.2573 | 0.0472 | 0.0742 | 0.0962 |
| mount_sidley | Mount Sidley | 0.4741 | 0.4701 | 0.3078 | 0.2926 | 0.2858 | -0.0040 | -0.8541 | 0.1623 | 34.5231 | 0.1775 | 37.7553 | 0.1843 | 39.2059 | 0.1663 | 0.1815 | 0.1883 |
| nico | Nico | 0.3681 | 0.2355 | 0.1633 | 0.1493 | 0.1471 | -0.1325 | -56.2688 | 0.0722 | 30.6507 | 0.0863 | 36.6349 | 0.0885 | 37.5599 | 0.2047 | 0.2188 | 0.2210 |
| sabrina | Sabrina | 0.2185 | 0.1903 | 0.1837 | 0.1815 | 0.1709 | -0.0283 | -14.8661 | 0.0066 | 3.4515 | 0.0088 | 4.6145 | 0.0194 | 10.1722 | 0.0349 | 0.0371 | 0.0476 |
| zhongshan | Zhongshan | 0.3288 | 0.2751 | 0.2607 | 0.2578 | 0.2553 | -0.0537 | -19.5373 | 0.0144 | 5.2308 | 0.0173 | 6.2736 | 0.0198 | 7.1873 | 0.0681 | 0.0710 | 0.0735 |

## Largest State-Linear Gains Over Global Mean

| station_id | global_mean | target_month | target_state_linear | target_state_linear_gain_vs_global |
| --- | --- | --- | --- | --- |
| nico | 0.3681 | 0.1493 | 0.1471 | 0.2210 |
| mount_sidley | 0.4741 | 0.2926 | 0.2858 | 0.1883 |
| butcher_ridge | 0.3145 | 0.2403 | 0.2183 | 0.0962 |
| zhongshan | 0.3288 | 0.2578 | 0.2553 | 0.0735 |
| sabrina | 0.2185 | 0.1815 | 0.1709 | 0.0476 |

## Variable-Level Mean MAE by Mode

| mode | variable | mae |
| --- | --- | --- |
| global_mean | P | 0.0786 |
| global_mean | RH | 0.6762 |
| global_mean | T | 0.2938 |
| global_mean | mean | 0.3408 |
| global_mean | q | 0.2327 |
| global_mean | wspd | 0.4960 |
| none | P | 0.0654 |
| none | RH | 0.6047 |
| none | T | 0.3000 |
| none | mean | 0.3061 |
| none | q | 0.2136 |
| none | wspd | 0.4297 |
| target_mean | P | 0.0177 |
| target_mean | RH | 0.4841 |
| target_mean | T | 0.1906 |
| target_mean | mean | 0.2366 |
| target_mean | q | 0.1783 |
| target_mean | wspd | 0.4147 |
| target_month | P | 0.0174 |
| target_month | RH | 0.4541 |
| target_month | T | 0.1683 |
| target_month | mean | 0.2243 |
| target_month | q | 0.1771 |
| target_month | wspd | 0.4102 |
| target_state_linear | P | 0.0173 |
| target_state_linear | RH | 0.4220 |
| target_state_linear | T | 0.1684 |
| target_state_linear | mean | 0.2155 |
| target_state_linear | q | 0.1732 |
| target_state_linear | wspd | 0.3757 |

## Initial Interpretation

The probe is positive under the Stage 0 gate: station-adaptive residual structure is large enough to justify a neural SABC experiment.
