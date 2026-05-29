# 2026-05-29 Figure and Table Audit

## Scope

Audited all paper figures and tables in `paper/main.tex` and `paper/sections/*.tex`.

Figures checked: Fig. 1 `fig_missing_patterns.pdf`; Fig. 2 `fig_ecbit_arch.tex`; Fig. 3 `fig_main_results.pdf`; Fig. 4 `fig_imputation_case.pdf`; Fig. 5 `fig_round1_baselines.pdf`; Fig. 6 `fig_round2_ablations.pdf`; Fig. 7 `fig_followup_analyses.pdf`; Fig. 8 `fig_heldout_bias_vs_mae.pdf`.

Tables checked: Tables I--XVII, including station metadata, split overlap, ERA5 calibration, station-month paired tests, real-gap plausibility, Round 1/2/3 results, fair ERA5 baselines, paired tests, non-overlap sensitivity, observed-position consistency, raw-unit errors, curriculum/ERA5 transfer matrix, held-out station detail, and appendix station metadata.

## Regeneration and Cross-Checks

Re-ran local, non-training generation scripts for tables and figures. No local model training was run. Neural checkpoint evaluation scripts were not re-run locally; tables derived from remote evaluation artifacts were checked through their CSV outputs and manuscript values.

Key scripts re-run or checked:

- `scripts/make_round1_table.py`
- `scripts/make_round2_table.py`
- `scripts/make_round3_table.py`
- `scripts/make_curriculum_factorial_table.py`
- `scripts/make_fair_baseline_table.py`
- `scripts/make_stat_tests_table.py`
- `scripts/make_station_month_neural_tests.py`
- `scripts/make_non_overlap_eval_table.py`
- `scripts/make_raw_unit_tables.py`
- `scripts/make_station_metadata_table.py`
- `scripts/analyze_split_overlap.py`
- `scripts/analyze_heldout_station_bias.py`
- `scripts/evaluate_era5_direct_calibration.py`
- `scripts/plot_missing_patterns.py`
- `scripts/plot_main_results.py`
- `scripts/plot_round1_baselines.py`
- `scripts/plot_round2_ablations.py`
- `scripts/plot_followup_analyses.py`
- `scripts/plot_imputation_case.py`

The ERA5 direct calibration recomputation confirmed the manuscript values: station-month mean bias MAE 0.269853, global linear MAE 0.378153, global mean bias MAE 0.381058, and no calibration MAE 0.387378.

The curriculum/ERA5 transfer matrix regenerated as Block+ERA5 0.258, Block/no-ERA5 0.346, MCAR+ERA5 on block tests 0.407, and MCAR/no-ERA5 on block tests 0.529.

The station-month calibrated ERA5 versus neural paired-test table regenerated with the same rounded values: SAITS+ERA5 -0.0249 MAE, iTransformer+ERA5 -0.0117, ECBIT concat -0.0119, and ECBIT gated -0.0121.

## Issues Found and Fixed

1. Fig. 3 group label was factually misleading. It labeled the first group as `No-ERA5 baselines`, but that group includes the `ERA5 direct` baseline. Fixed to `Core baselines`. Also changed the y-axis to `Mean test MAE (normalized)`.

2. The main station summary table used the median station elevation but labeled it only as `Elev.`. Fixed the column header to `Median elev.` and added the median-elevation definition to the caption.

3. The appendix metadata table showed `aws11  ` with a trailing-space artifact. The table generator now strips station-name whitespace before LaTeX emission.

4. Raw-unit specific-humidity errors were displayed with two decimals, turning the standard deviation into `0.00`. The table now uses three decimals for `q`, showing `0.145 +/- 0.004` MAE and `0.212 +/- 0.005` RMSE.

5. `scripts/make_round2_table.py` still defaulted to stale `round2_partial_runs.csv`. Fixed it to use `round2_gated_final_runs.csv`, and clarified the table caption so the MCAR row is explicitly an in-distribution MCAR-test reference distinct from the MCAR-on-block rows in the curriculum transfer table.

6. `scripts/make_curriculum_factorial_table.py` would regenerate a weaker caption. Fixed it to preserve the manuscript's out-of-distribution MCAR-on-block interpretation.

7. Fig. 7 panel (a)'s first `+0.037` gap annotation was close to the y-axis. Shifted only that label rightward. No data values changed.

## Visual Audit

Rendered the compiled PDF pages containing figures and tables. No obvious factual label mismatch, legend-data mismatch, table clipping, or unreadable overlap remains.

Specific visual checks:

- Fig. 4 no longer has a redundant bold title above the plot; the top legend is readable and does not overlap the curve region.
- Fig. 5 and Fig. 6 legends are above the panels and do not obscure curves.
- Fig. 3 now correctly labels the baseline group as `Core baselines`.
- Fig. 7 labels and panel captions match the manuscript text: temperature, wind speed, and specific humidity are the top ERA5 channel-importance variables.
- Appendix Table XVII is small but readable; the `aws11` station-name artifact is removed.

## Verification

Final `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` completed successfully.

Final PDF: 18 pages; no undefined references; no undefined citations; no BibTeX warnings; no overfull boxes; no fatal errors or emergency stops; only non-blocking underfull hbox/vbox layout warnings.

## Residual Notes

- Re-running direct calibration and held-out bias scripts caused tiny floating-point-only CSV differences at approximately 1e-8 scale. Rounded table values and manuscript claims are unchanged.
- Matplotlib emitted cache-directory/fontconfig warnings because `/home/horeb/.config/matplotlib` is not writable in the sandbox. All figure outputs were still written successfully.
- `paper/main.tex` already contained an unrelated author-line modification before this audit; this audit does not rely on or revert that change.
