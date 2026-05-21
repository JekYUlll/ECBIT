# ECBIT Task Plan
Goal: Submit ERA5-Conditioned Block Imputation paper to IEEE TGRS or Remote Sensing.
Current Phase: Phase 3 — Experiments / Phase 5 — Paper Writing
Last Updated: 2026-05-21
Current Claim: ERA5-conditioned block-missing AWS imputation is strongly supported; complex cross-attention/gated fusion has not yet shown stable gains over lightweight no-cross conditioning.

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
- [ ] Run baseline experiments
- [x] Run ECBIT ablations
- [x] Run gated-injection ECBIT ablations (108/108 complete as of 2026-05-21 17:39 CST)
- [x] Run held-out station generalization experiments (45/45 complete as of 2026-05-21 17:39 CST)
- [x] Aggregate results for completed Round1 core, Round2 cross-attention, Round2 gated, and Round3 held-out matrices

## Phase 4 · Analysis and Visualization
- [x] Missing pattern analysis figure
- [ ] Main result comparison figure
- [ ] ERA5 conditioning effect figure
- [ ] Imputation curve visualization

## Phase 5 · Paper Writing
- [x] Introduction and Related Work
- [x] Methodology
- [x] Experiments updated with completed gated and held-out results
- [x] Conclusion updated with current evidence
- [ ] Format and submission checks

## Blocked Issues
| Issue | Status | Action |
|-------|--------|--------|
| Full-station AntAWS ERA5 alignment not yet available | resolved | Downloaded/aligned ERA5 point time series for all 32 selected stations |
| PyPOTS not installed in local `darts` environment | open | Wrappers implemented with lazy import; verify/install PyPOTS before SAITS/BRITS remote jobs |
| Remote SSH authentication unavailable | resolved | microclimate-experiment-server credential fallback works; server login, conda `darts`, PyPOTS import, and GPU visibility verified |
| Duplicate Round1 worker launches caused repeated LOCF jobs | resolved | Stopped duplicate remote processes, added per-run lock files to `scripts/worker.py`, and relaunched Round1 core in tmux |
| Neural training results reported validation metrics instead of test metrics | resolved for completed Round1/Round2 and partial Round3 | Patched training/aggregation/recovery scripts; recovered completed neural result JSON files from checkpoints and regenerated aggregates |

## Completion Criteria
Paper submitted to IEEE TGRS or Remote Sensing.
