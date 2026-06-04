# SABC Gate Report

Matched pairs: 45
Overall mean improvement (gated baseline - SABC): -0.0039 normalized MAE
Mount Sidley mean improvement: -0.0097
Zhongshan mean improvement: -0.0099
Nico mean improvement: -0.0174

| Gate | Threshold | Value | Pass |
|---|---:|---:|:---:|
| Held-out mean improvement | >= 0.015 | -0.0039 | no |
| Mount Sidley or Zhongshan improvement | >= 0.030 | -0.0097 | no |
| Nico degradation guard | >= -0.010 | -0.0174 | no |

Decision: do not pass yet

## Paired Tests

| group | n | mean_improvement | ci95_low | ci95_high | t | p |
| --- | --- | --- | --- | --- | --- | --- |
| overall | 45 | -0.0039 | -0.0086 | 0.0007 | -1.7015 | 0.0959 |
| station:butcher_ridge | 9 | 0.0067 | -0.0016 | 0.0149 | 1.8530 | 0.1010 |
| station:mount_sidley | 9 | -0.0097 | -0.0152 | -0.0043 | -4.1143 | 0.0034 |
| station:nico | 9 | -0.0174 | -0.0304 | -0.0044 | -3.0914 | 0.0149 |
| station:sabrina | 9 | 0.0108 | 0.0062 | 0.0154 | 5.3854 | 0.0007 |
| station:zhongshan | 9 | -0.0099 | -0.0200 | 0.0002 | -2.2553 | 0.0541 |
| pattern:long | 15 | -0.0077 | -0.0210 | 0.0056 | -1.2391 | 0.2357 |
| pattern:medium | 15 | -0.0035 | -0.0089 | 0.0020 | -1.3715 | 0.1918 |
| pattern:short | 15 | -0.0006 | -0.0044 | 0.0033 | -0.3227 | 0.7517 |

## Station Mean Improvements

| station_id | mean_improvement |
| --- | --- |
| butcher_ridge | 0.0067 |
| mount_sidley | -0.0097 |
| nico | -0.0174 |
| sabrina | 0.0108 |
| zhongshan | -0.0099 |
