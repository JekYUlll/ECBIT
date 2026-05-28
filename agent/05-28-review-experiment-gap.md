# Review: Experiment Gap Check — 2026-05-28

## Verdict

No submission-blocking new experiment is required for an application-oriented polar or meteorological journal. The manuscript now has enough empirical coverage for the central claim: realistic block-missing curriculum and ERA5 conditioning matter more than the specific gated ECBIT fusion layer.

The paper already covers the major reviewer-risk categories:

- realistic block-missing benchmark and mask-generator transparency;
- fair non-ERA5 baselines and fair ERA5-augmented SAITS/iTransformer baselines;
- station-month calibrated ERA5 direct baseline and paired tests against neural variants;
- block-vs-MCAR transfer matrix;
- outage-length, ERA5-variable, and ERA5-temporal-resolution analyses;
- held-out station detail and ERA5-AWS mismatch diagnostic;
- real historical-gap plausibility audit;
- split/window overlap audit and non-overlap iTransformer+ERA5 sensitivity;
- observed-position copy-back audit;
- raw-unit error interpretation and reference audit.

## Highest-Value Optional Experiment

If only one more experiment is added before submission, make it a local stateless baseline rather than a new GPU training run:

**Station-month calibrated ERA5 + endpoint/interpolation hybrid baseline.**

Rationale: the strongest remaining reviewer question is not whether gated ECBIT is the best neural architecture; the manuscript already says it is not. The sharper question is whether a simple operational method can close the remaining 0.012 MAE gap between station-month calibrated ERA5 direct substitution (0.270) and variate-token ERA5 neural models (about 0.258).

Suggested variants:

1. `linear_or_era5`: use linear interpolation for short gaps and station-month calibrated ERA5 for medium/long gaps.
2. `era5_endpoint_anchored`: station-month calibrated ERA5 inside the gap plus a linear endpoint residual correction so the fill connects to observed AWS boundary values.
3. `weighted_linear_era5`: convex blend between linear interpolation and station-month calibrated ERA5, with weight chosen on training/validation masks only.

Decision rule:

- If the hybrid remains above neural models, it strengthens the statement that sparse AWS context adds modest value beyond calibrated ERA5.
- If the hybrid matches or beats neural models, the paper can still be publishable, but the conclusion should pivot further toward operational calibrated ERA5/hybrid baselines and treat neural conditioning as a benchmark comparator rather than the recommended deployment default.

This experiment should be local-only and stateless. It should not require model training.

## Optional, Not Required

- **Season/month-level error analysis.** Useful for Theoretical and Applied Climatology, especially because Antarctic winter boundary-layer conditions are explicitly discussed. It is not currently submission-blocking because the paper already acknowledges seasonal/event analysis as future work.
- **Multiple held-out station splits.** Stronger spatial-generalization evidence, but expensive and not necessary for the current conservative framing. The current five held-out stations plus station-level table are acceptable if claims remain cautious.
- **BRITS/CSDI block-curriculum baselines.** Useful for a method-heavy ML venue, but not needed for the current application-oriented target because SAITS and iTransformer already cover two attention-style baselines with and without ERA5, and the paper explicitly limits architecture-dominance claims.
- **Gated ECBIT strict non-overlap re-evaluation.** Not available from retained artifacts because the Round 2 gated checkpoints were not retained. The iTransformer+ERA5 non-overlap sensitivity is sufficient as a transparency check unless the authors decide to retrain.
- **Uncertainty-aware/probabilistic outputs.** Good future work, not needed for the deterministic benchmark claim.

## Reviewer-Risk Ranking

| Risk | Need new experiment? | Current mitigation |
|---|---:|---|
| Simple calibrated ERA5 baseline nearly matches neural models | Optional local baseline recommended | Station-month ERA5 table + paired tests + modest-gain wording |
| Architecture novelty weak | No | Fair ERA5 baselines and conservative framing |
| Single held-out split | No, unless targeting stronger spatial-generalization claim | Per-station/variable table, mismatch diagnostic, cautious wording |
| No labels inside natural historical gaps | No | Real-gap plausibility audit + limitation |
| BRITS/CSDI missing under block retraining | No for application journal | SAITS/iTransformer controls + future-work limitation |
| Seasonal physical interpretation | Optional writing/analysis enhancement | Failure-mode discussion already flags season/event analysis as future work |

## Recommendation

Proceed without new GPU experiments. For a final pre-submission hardening pass, implement the station-month calibrated ERA5 hybrid baseline if time is available, because it directly targets the only remaining high-salience reviewer question and can be done locally.
