# SABC Gate Report

Matched pairs: 41
Overall mean improvement (gated baseline - SABC): -0.0040 normalized MAE
Mount Sidley mean improvement: -0.0097
Zhongshan mean improvement: -0.0156
Nico mean improvement: -0.0174

| Gate | Threshold | Value | Pass |
|---|---:|---:|:---:|
| Held-out mean improvement | >= 0.015 | -0.0040 | no |
| Mount Sidley or Zhongshan improvement | >= 0.030 | -0.0097 | no |
| Nico degradation guard | >= -0.010 | -0.0174 | no |

Decision: do not pass yet

## Paired Tests

| group | n | mean_improvement | ci95_low | ci95_high | t | p |
| --- | --- | --- | --- | --- | --- | --- |
| overall | 41 | -0.0040 | -0.0091 | 0.0010 | -1.6041 | 0.1166 |
| station:butcher_ridge | 9 | 0.0067 | -0.0016 | 0.0149 | 1.8530 | 0.1010 |
| station:mount_sidley | 9 | -0.0097 | -0.0152 | -0.0043 | -4.1143 | 0.0034 |
| station:nico | 9 | -0.0174 | -0.0304 | -0.0044 | -3.0914 | 0.0149 |
| station:sabrina | 9 | 0.0108 | 0.0062 | 0.0154 | 5.3854 | 0.0007 |
| station:zhongshan | 5 | -0.0156 | -0.0341 | 0.0029 | -2.3458 | 0.0789 |
| pattern:long | 15 | -0.0077 | -0.0210 | 0.0056 | -1.2391 | 0.2357 |
| pattern:medium | 13 | -0.0035 | -0.0096 | 0.0027 | -1.2256 | 0.2439 |
| pattern:short | 13 | -0.0004 | -0.0049 | 0.0041 | -0.1863 | 0.8553 |

## Station Mean Improvements

| station_id | mean_improvement |
| --- | --- |
| butcher_ridge | 0.0067 |
| mount_sidley | -0.0097 |
| nico | -0.0174 |
| sabrina | 0.0108 |
| zhongshan | -0.0156 |
