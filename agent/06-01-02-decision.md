# 06-01-02 Review Triage Decision

## Decision

The next revision should be a manuscript-only pass plus local aggregation from existing CSV artifacts. Do not launch new GPU training unless explicitly requested.

## Immediate Edits

1. Further reduce any architecture-forward wording in the Introduction so ECBIT remains a controlled implementation/testbed rather than a primary architectural contribution.
2. Add station-selection attrition counts in Data/Benchmark: 267 raw station files to 117 after record length, 38 after temperature completeness, 36 after core-mean completeness, and 32 after per-variable completeness.
3. Define normalized MAE/RMSE early in Experimental Protocol as errors in training-window standard-deviation units; cross-reference raw-unit Table XIII.
4. Report empirical clipped long-block diagnostics in Methodology/Protocol: long sampled median 155--156 steps, clipped median 80 steps, p10 17, p90 139, max 168; supervised label segments can be shorter because original AWS gaps split observed labels.
5. Clarify that AWS-derived specific humidity inherits source-variable observation masks, but artificial block masks are sampled per model channel and do not automatically hide q when T/RH/P is artificially hidden unless q itself is sampled.
6. Make the held-out ERA5-AWS mismatch result explicitly qualitative; avoid leaning on Pearson r=0.66 as an inferential finding.
7. Add an explanation of SAITS+ERA5's larger variance: it is mainly regime/rate sensitivity, especially long/high-missing settings, not only seed noise.
8. Clarify that the ERA5 temporal downsampling analysis is inference-time perturbation of a 3-hourly-trained model.
9. Add practical significance wording for the 0.012--0.025 normalized MAE gain over station-month calibrated ERA5: useful but modest, and calibrated ERA5 remains the operational default if neural deployment complexity is unacceptable.
10. Strengthen limitations around missing probabilistic imputers, ERA5 orographic-height/lapse-rate correction, and absent runtime benchmarking.

## No New Experiment For Now

- Do not train CSDI or GP-VAE now. Treat them as a future benchmark expansion because fair block-curriculum and ERA5-conditioned versions would require substantial additional implementation and GPU time.
- Do not retrain alternate held-out station splits now. The current five-station held-out result should be framed as a limited diagnostic rather than a strong generalization estimate.
- Do not add ERA5 orographic-height correction unless a local ERA5 geopotential/elevation artifact is already available. The current data download does not include ERA5 surface geopotential.
- Do not add compute-time claims unless trustworthy remote logs exist. It is safer to state that runtime benchmarking is not part of the current evidence.

## Already Sufficient

- Title, Abstract, and Conclusion already mostly demote ECBIT architecture dominance.
- CDS point extraction wording is already conservative.
- ERA5 specific humidity derivation from temperature/dewpoint/pressure is already stated.
- MCAR in-distribution vs MCAR-on-block rows are already separated in captions; only a small cross-reference edit may be useful.
- PROMICE/GC-NET 2026 citation already includes DOI metadata.
