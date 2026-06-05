# SABC v2 Gate Report

Improvement is matched gated baseline MAE minus residual-aware SABC v2 MAE.

Matched pairs: 27
Overall mean improvement: 0.0102 normalized MAE

## Paired Tests

| group | n | mean_improvement | ci95_low | ci95_high | t | p |
| --- | --- | --- | --- | --- | --- | --- |
| overall | 27 | 0.0102 | 0.0026 | 0.0178 | 2.7633 | 0.0104 |
| station:mount_sidley | 9 | 0.0228 | 0.0117 | 0.0339 | 4.7363 | 0.0015 |
| station:nico | 9 | -0.0105 | -0.0168 | -0.0043 | -3.8760 | 0.0047 |
| station:zhongshan | 9 | 0.0184 | 0.0078 | 0.0291 | 3.9861 | 0.0040 |
| pattern:long | 9 | 0.0210 | 0.0012 | 0.0408 | 2.4463 | 0.0402 |
| pattern:medium | 9 | 0.0062 | -0.0071 | 0.0195 | 1.0700 | 0.3158 |
| pattern:short | 9 | 0.0035 | -0.0015 | 0.0086 | 1.6271 | 0.1424 |

## Station Mean Improvements

| station_id | mean_improvement |
| --- | --- |
| mount_sidley | 0.0228 |
| nico | -0.0105 |
| zhongshan | 0.0184 |

## Pattern Mean Improvements

| pattern | mean_improvement |
| --- | --- |
| long | 0.0210 |
| medium | 0.0062 |
| short | 0.0035 |
