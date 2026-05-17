# AGENTS.md — ECBIT Research Project

## Project Goal
Complete and submit "ERA5-Conditioned Block Imputation for Sparse Antarctic
Weather Station Time Series" to IEEE TGRS or Remote Sensing.
Target submission: within 4 weeks of project start.

## Project Context
This project builds on two prior experiments:
1. AntAWS benchmarking: iTransformer was strongest for forecasting, and variate-token attention captured meteorological coupling.
2. ECAFT recovery: ERA5 meteorological variables were mostly redundant with AWS, ERA5 RH showed systematic bias, and naive parallel cross-attention failed.

ECBIT is the direct response:
- ERA5 is a condition, not a parallel stream.
- ERA5 activates only for missing variables/timesteps.
- Block missing simulation matches polar AWS sensor failure.
- Variable-level gating contains ERA5 bias to affected variables.

## Data Assets
- AntAWS raw CSVs: `data/antaws/raw/`
- ERA5 aligned data: `data/era5/`
- ERA5 resampled 3h data: `data/era5_3h/`
- Optional IMAU AWS data: `data/imau/`

Do not re-download existing data unless integrity checks show it is missing or corrupted.

## Compute Routing
- LOCAL: data preprocessing, station analysis, script writing, plotting, LaTeX compilation, result analysis.
- REMOTE: all neural model training and evaluation.
- Never run `src/train_impute.py` or large imputation training locally.
- Use the `microclimate-experiment-server` skill for remote submissions.

## Parallel Work Protocol
After submitting a remote experiment batch:
- Immediately switch to paper writing or literature tasks.
- Poll experiment status every 2 hours.
- Pull results as soon as jobs complete.
- Update `agent/findings.md` immediately.

## Skill Usage
- `planning-with-files`: read at every session start.
- `autoresearch`: literature search on imputation and polar AWS gaps.
- `ml-paper-writing`: LaTeX writing and citation checks.
- `academic-plotting`: publication figures and architecture diagrams.
- `weights-and-biases`: experiment tracking, project `ecbit-polar`.
- `microclimate-experiment-server`: remote training and monitoring.

## Paper Toolchain
- LaTeX: `paper/main.tex`
- Sections: `paper/sections/`
- Figures: `paper/figures/`
- Compile locally: `latexmk -pdf paper/main.tex`

## Error Recovery
If a blocking error occurs:
1. Run `scripts/release_resources.sh` if applicable.
2. Log the issue in `agent/progress.md` under blocked issues.
3. Stop cleanly if root cause requires external action.

## Git Convention
Commit after each completed atomic task.

Format:

```text
[phase] action: description
```

Examples:

```text
[data] feat: AntAWS station selection, 38 stations retained
[model] feat: ECBIT conditional cross-attention module
[exp] result: Round 1 baselines complete
[paper] write: methodology section
```
