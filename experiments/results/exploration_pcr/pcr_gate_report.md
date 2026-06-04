# PCR Gate Report

Completed runs: 18

Overall mean improvement (gated baseline - PCR): 0.0004 normalized MAE with 95% CI [-0.0000, 0.0009].

MAE non-degradation gate (`>= -0.003`): pass

## Paired Tests

| group | n | mean_improvement | ci95_low | ci95_high | t | p |
| --- | --- | --- | --- | --- | --- | --- |
| overall | 9 | 0.0004 | -0.0000 | 0.0009 | 2.2011 | 0.0589 |
| pattern:long | 3 | 0.0006 | -0.0014 | 0.0026 | 1.3272 | 0.3157 |
| pattern:medium | 3 | 0.0005 | -0.0008 | 0.0019 | 1.6785 | 0.2352 |
| pattern:short | 3 | 0.0001 | -0.0009 | 0.0012 | 0.5007 | 0.6662 |
