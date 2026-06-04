# SABC Gate Report

Matched pairs: 13
Overall mean improvement (gated baseline - SABC): 0.0012 normalized MAE
Mount Sidley mean improvement: -0.0112
Zhongshan mean improvement: nan
Nico mean improvement: nan

| Gate | Threshold | Value | Pass |
|---|---:|---:|:---:|
| Held-out mean improvement | >= 0.015 | 0.0012 | no |
| Mount Sidley or Zhongshan improvement | >= 0.030 | -0.0112 | no |
| Nico degradation guard | >= -0.010 | nan | yes |

Decision: do not pass yet

## Paired Tests

| group | n | mean_improvement | ci95_low | ci95_high | t | p |
| --- | --- | --- | --- | --- | --- | --- |
| overall | 13 | 0.0012 | -0.0065 | 0.0088 | 0.3319 | 0.7457 |
| station:butcher_ridge | 9 | 0.0067 | -0.0016 | 0.0149 | 1.8530 | 0.1010 |
| station:mount_sidley | 4 | -0.0112 | -0.0214 | -0.0010 | -3.4987 | 0.0395 |
| pattern:long | 5 | 0.0062 | -0.0157 | 0.0282 | 0.7854 | 0.4761 |
| pattern:medium | 5 | -0.0064 | -0.0143 | 0.0014 | -2.2882 | 0.0840 |
| pattern:short | 3 | 0.0055 | -0.0074 | 0.0183 | 1.8256 | 0.2095 |

## Station Mean Improvements

| station_id | mean_improvement |
| --- | --- |
| butcher_ridge | 0.0067 |
| mount_sidley | -0.0112 |
