# ECBIT Task Plan
Goal: Submit ERA5-Conditioned Block Imputation paper to IEEE TGRS or Remote Sensing.
Current Phase: Phase 1 — Data Engineering
Last Updated: 2026-05-18

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
- [ ] Resample ERA5 2h data to AntAWS 3h resolution
- [ ] Analyze real missing block patterns
- [ ] Implement AntAWS preprocessing for imputation
- [ ] Implement block missing simulator and unit tests

## Phase 2 · Model Implementation
- [ ] Implement ECBIT model
- [ ] Implement interpolation and carry-forward baselines
- [ ] Implement PyPOTS SAITS and BRITS wrappers
- [ ] Implement iTransformer imputation baseline
- [ ] Implement ERA5 direct substitution baseline
- [ ] Implement unified remote training entrypoint
- [ ] Generate experiment configs

## Phase 3 · Experiments
- [ ] Run baseline experiments
- [ ] Run ECBIT ablations
- [ ] Run held-out station generalization experiments
- [ ] Aggregate results

## Phase 4 · Analysis and Visualization
- [ ] Missing pattern analysis figure
- [ ] Main result comparison figure
- [ ] ERA5 conditioning effect figure
- [ ] Imputation curve visualization

## Phase 5 · Paper Writing
- [ ] Introduction and Related Work
- [ ] Methodology
- [ ] Experiments
- [ ] Conclusion
- [ ] Format and submission checks

## Blocked Issues
| Issue | Status | Action |
|-------|--------|--------|
| Full-station AntAWS ERA5 alignment not yet available | open | First data task must select stations, then verify/download ERA5 for those coordinates |

## Completion Criteria
Paper submitted to IEEE TGRS or Remote Sensing.
