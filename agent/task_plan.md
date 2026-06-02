# ECBIT Task Plan
Goal: Prepare a journal-style SCI manuscript on ERA5-conditioned Antarctic AWS block imputation.
Current Phase: Phase 17 — journal selection and submission targeting
Last Updated: 2026-06-02
Current Claim: ERA5-conditioned block-missing AWS imputation is strongly supported; the journal narrative should emphasize realistic outage modeling, ERA5 information value, block-curriculum interaction, and station generalization rather than a specific fusion-module novelty.
Active Follow-up: 06-01-02 manuscript revision is complete. The paper now incorporates the selected local/statistical clarifications without adding a defensive reviewer-response layer. No new GPU training was launched. Current work is selecting a journal target and defining the smallest venue-specific submission edits.

## Phase 0 · Initialization
- [x] Add ECBIT as project submodule under archived ECAFT repository
- [x] Create project directory structure
- [x] Write `AGENTS.md`, `CLAUDE.md`, and `README.md`
- [x] Initialize `agent/task_plan.md`, `agent/findings.md`, and `agent/progress.md`
- [x] Verify skill/tool availability for ECBIT workflow
- [x] Test microclimate-experiment-server connectivity
- [x] Audit available AntAWS and ERA5 data assets

## Phase 1 · Data Engineering
- [x] Implement AntAWS station selection script
- [x] Resample ERA5 2h data to AntAWS 3h resolution
- [x] Analyze real missing block patterns
- [x] Implement AntAWS preprocessing for imputation
- [x] Implement block missing simulator and unit tests

## Phase 2 · Model Implementation
- [x] Implement ECBIT model
- [x] Implement interpolation and carry-forward baselines
- [x] Implement PyPOTS SAITS and BRITS wrappers
- [x] Implement iTransformer imputation baseline
- [x] Implement ERA5 direct substitution baseline
- [x] Implement unified remote training entrypoint
- [x] Generate experiment configs

## Phase 3 · Experiments
- [x] Run baseline experiments
  - [x] Complete SAITS block-missing third-party baseline matrix (27/27 complete as of 2026-05-24)
- [x] Run ECBIT ablations
- [x] Run gated-injection ECBIT ablations (108/108 complete as of 2026-05-21 17:39 CST)
- [x] Run held-out station generalization experiments (45/45 complete as of 2026-05-21 17:39 CST)
- [x] Aggregate results for completed Round1 core, Round2 cross-attention, Round2 gated, and Round3 held-out matrices

## Phase 4 · Analysis and Visualization
- [x] Missing pattern analysis figure
- [x] Main result comparison figure
- [x] ERA5 conditioning effect figure (follow-up synthesis figure added to paper)
- [x] ERA5 variable importance analysis
- [x] ERA5 temporal-resolution robustness analysis
- [x] MCAR-trained ERA5 baseline evaluated on block-missing tests
- [x] Block length sensitivity training matrix (18/18 complete as of 2026-05-22 14:05 CST)
- [x] MCAR-trained no-ERA5 factorial completion for block-test 2x2 matrix
- [x] Imputation curve visualization

## Phase 5 · Paper Writing
- [x] Introduction and Related Work
- [x] Methodology
- [x] Experiments updated with completed gated and held-out results
- [x] Conclusion updated with current evidence
- [x] Journal-style data/benchmark section
- [x] Journal-style experimental protocol section
- [x] Journal-style discussion and limitations section
- [x] Format and submission checks

## Phase 6 · Major Revision 05-24-2
- [x] Plan and triage major-review tasks into local analysis, remote GPU experiments, and manuscript-only edits
- [x] Replace strong causal/fusion novelty wording with controlled predictive-value and benchmark/curriculum framing
- [x] Add mask-generator pseudocode and realized block-length diagnostics for short/medium/long regimes
- [x] Disclose window generation details: stride, chronological split fractions, minimum observation thresholds, and per-station window counts
- [x] Clarify specific humidity derivation, missingness propagation, and normalization
- [x] Clarify final inference rule for preserving observed AWS values; add observed-position consistency audit
- [x] Expand ERA5 direct bias-correction definition and add no-bias / mean-bias / linear-calibration ablation if feasible
- [x] Generate and submit SAITS+ERA5 and iTransformer+ERA5 fair-baseline configs on remote GPUs
- [x] Monitor fair ERA5 baseline batch, sync results, aggregate tables, and update manuscript
- [x] Add paired statistics with effect size, 95% CI, and Holm correction for expanded comparisons
- [x] Add real-gap plausibility diagnostics and case figures for historical gaps without treating them as supervised labels
- [x] Add strict chronological split-overlap audit and deterministic non-overlap test subset/config for sensitivity evaluation
- [x] Run remote non-overlap iTransformer+ERA5 checkpoint evaluation and report sensitivity delta; document ECBIT-gated checkpoint limitation
- [x] Add station x variable and raw-unit error tables
- [x] Update reproducibility artifacts: station metadata, split/window index, configs, raw result inventory, environment notes
- [x] Compile, visually check figures/tables, and commit each completed atomic task

## Phase 7 · Medium Revision 05-26-01
- [x] Read `agent/05-26-01-codex.md` and triage requested edits into local analysis, manuscript changes, and optional experiments
- [x] Reframe Abstract, Contributions, Results, Discussion, and Conclusion around benchmark/protocol evidence rather than ECBIT architecture superiority
- [x] Add station-level metadata table and cite it from Data/Benchmark and reproducibility text
- [x] Expand ERA5 extraction/alignment details: single-level time-series source, point extraction, temporal interpolation, unit conversion, q/RH derivation, and train-only normalization
- [x] Add station-month ERA5 direct calibration baseline and update ERA5 calibration table/results
- [x] Add held-out station detail table with per-variable MAE/RMSE and ERA5-AWS mismatch diagnostic
- [x] Add held-out MAE vs ERA5-AWS mismatch figure and cautious interpretation
- [x] Strengthen historical real-gap audit wording so it cannot be read as supervised historical-gap accuracy
- [x] Expand Limitations with architecture, ERA5 bias, artificial-mask, station coverage, variable coverage, and operational constraints
- [x] Add/update revision output artifacts under `experiments/results/revision/`
- [x] Compile, visually check new tables/figures, update planning files, and commit atomic changes

## Phase 8 · Post-Review Cleanup 05-26
- [x] Add paired tests comparing station-month calibrated ERA5 direct substitution against SAITS+ERA5, iTransformer+ERA5, ECBIT concat, and ECBIT gated
- [x] Reframe station-month ERA5 direct as a strong operational baseline and neural ERA5 conditioning as a modest additional gain from sparse AWS context
- [x] Update title toward application-oriented AWS records wording
- [x] Correct Abstract wording from generic humidity channels to specific-humidity channels
- [x] Clarify CDS point extraction and local interpolation wording
- [x] Move full 32-station metadata table to appendix and keep only a compact benchmark summary table in the main text
- [x] Check Discussion B continuity and compile/visual-check the revised PDF

## Phase 9 · Post-Review Verification 05-27
- [x] Re-read planning files and verify latest re-review items against actual LaTeX/table artifacts
- [x] Regenerate station-month calibrated ERA5 vs neural paired-test table and confirm values match the manuscript
- [x] Harmonize specific-humidity wording in the contribution list and Abstract
- [x] Re-run LaTeX compile, log scan, reference count, placeholder scan, and targeted local tests
- [x] Record residual constraints: gated ECBIT strict non-overlap re-evaluation is not available from retained artifacts

## Phase 10 · Reference Audit 05-27
- [x] Perform online verification of all 30 BibTeX entries against publisher, DOI, arXiv, OpenReview, NeurIPS, AAAI, and PMLR pages
- [x] Correct AntAWS author names, IMAU page range, PROMICE/GC-NET author name, and Informer official proceedings metadata
- [x] Add `agent/reference_audit_2026-05-27.md` with audit scope, corrections, and verified-without-change categories
- [x] Re-run BibTeX/LaTeX and verify `paper/main.bbl` contains 30 references with no BibTeX or LaTeX citation warnings

## Phase 11 · Experiment-Gap Review 05-28
- [x] Re-check manuscript evidence coverage and compile/log state from a reviewer perspective
- [x] Decide whether additional experiments are submission-blocking
- [x] Rank optional experiments by reviewer-risk reduction
- [x] Add `agent/05-28-review-experiment-gap.md`


## Phase 12 · Figure/Table Audit 05-29
- [x] Inventory all manuscript figures and tables and map them to source scripts/artifacts
- [x] Re-run local non-training table and figure generation scripts
- [x] Cross-check headline numbers against CSV artifacts and manuscript claims
- [x] Fix factual or potentially misleading labels in Fig. 3, station metadata, raw-unit q errors, and generation-script captions
- [x] Render and visually inspect the figure/table pages for overlaps, clipping, and legend placement
- [x] Add `agent/05-29-figure-table-audit.md`

## Phase 13 · Narrative/Prose Audit 05-29
- [x] Scan the manuscript for residual early ECAFT/radiation/forecasting direction cues
- [x] Remove explicit revision-response wording and reviewer-facing prose
- [x] Compress repeated architecture-disclaimer passages while preserving the conservative claim
- [x] Recompile and scan LaTeX logs
- [x] Add `agent/05-29-prose-audit.md`
- [x] Commit the atomic edit

## Phase 14 · Citation Truth/Content Audit 05-29
- [x] Spawn independent subagents for bibliography metadata, citation-context fit, and coverage/reasonableness
- [x] Re-verify bibliography metadata against primary online sources
- [x] Re-check all citation contexts against cited-paper content
- [x] Patch BibTeX or prose if mismatches are found
- [x] Compile and scan LaTeX/BibTeX logs
- [x] Add `agent/05-29-citation-truth-audit.md`
- [x] Commit the atomic edit

## Phase 15 · Reference Follow-up 06-01
- [x] Read `agent/06-01-01.md` and triage its confirmed errors, suspected issues, and no-change items
- [x] Verify the disputed Wang et al. imputation-survey author list against arXiv metadata before editing BibTeX
- [x] Correct the Lee et al. auxiliary-fusion citation context to use `Controlled Fusion Adapter (CFA)` and time-series forecasting wording
- [x] Confirm the manuscript does not cite an incorrect `154 station-years` value for the IMAU AWS archive
- [x] Recompile the manuscript and scan LaTeX/BibTeX logs

## Phase 16 · Review Triage 06-01-02
- [x] Read `agent/06-01-02-review.md` and compare the requested revisions with the current manuscript
- [x] Decide which items require immediate manuscript edits versus optional experiments
- [x] Confirm station-selection attrition counts from `data/station_meta_ecbit.csv`
- [x] Confirm realized long-block clipped-length statistics from the existing diagnostics
- [x] Confirm SAITS+ERA5 variance source from existing fair-baseline run tables
- [x] Apply the selected manuscript-only edits and local table/text updates
- [x] Recompile, scan logs, update progress, and commit

## Phase 17 · Journal Selection 06-01
- [x] Re-read planning context and current manuscript framing
- [x] Check current official journal scopes and publishing models for leading candidates
- [x] Rank target journals by fit to the current benchmark/application narrative
- [x] Record venue-specific risks and required pre-submission edits in `agent/06-01-03-journal-selection.md`
- [x] Convert manuscript references to Polar Science author-year style with natbib and Elsevier Harvard bibliography style
- [x] Recompile manuscript after citation-style conversion and verify no undefined citations or references remain
- [x] Convert LaTeX manuscript from IEEEtran two-column layout to Elsevier `elsarticle` single-column preprint format with 12 pt type, double spacing, line numbers, frontmatter abstract, and keywords
- [x] Reorganize top-level section numbering toward Polar Science IMRAD style: Introduction, Data and methods, Results, Interpretations and discussions, Conclusions
- [x] Recompile Elsevier-format manuscript and scan final logs for citation, reference, and fatal compile errors
- [ ] Replace placeholder author, affiliation, corresponding-author, acknowledgements, funding, and data/code statements with final submission information
- [ ] Compress display items and move dense statistical tables to appendix/supplementary material

## Blocked Issues
| Issue | Status | Action |
|-------|--------|--------|
| Full-station AntAWS ERA5 alignment not yet available | resolved | Downloaded/aligned ERA5 point time series for all 32 selected stations |
| PyPOTS not installed in local `darts` environment | open | Wrappers implemented with lazy import; verify/install PyPOTS before SAITS/BRITS remote jobs |
| Remote SSH authentication unavailable | resolved | microclimate-experiment-server credential fallback works; server login, conda `darts`, PyPOTS import, and GPU visibility verified |
| Duplicate Round1 worker launches caused repeated LOCF jobs | resolved | Stopped duplicate remote processes, added per-run lock files to `scripts/worker.py`, and relaunched Round1 core in tmux |
| Neural training results reported validation metrics instead of test metrics | resolved for completed Round1/Round2 and partial Round3 | Patched training/aggregation/recovery scripts; recovered completed neural result JSON files from checkpoints and regenerated aggregates |

## Completion Criteria
Journal-style SCI manuscript package ready for venue selection, with complete experiment tables, reproducible figures, compiled LaTeX PDF, and a defensible narrative centered on Antarctic AWS block imputation rather than a specific fusion-module claim.
