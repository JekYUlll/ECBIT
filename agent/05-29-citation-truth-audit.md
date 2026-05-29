# 2026-05-29 Citation Truth and Content Audit

## Scope
- Audited `paper/references.bib`, `paper/main.tex`, and `paper/sections/*.tex`.
- Used one coverage/reasonableness subagent, one metadata-focused subagent, and one citation-context subagent.
- Performed local DOI/arXiv/Crossref checks and LaTeX/BibTeX verification.

## Subagent Findings
- Coverage subagent: all 30 original BibTeX entries were cited; no missing or uncited keys. Main risks were missing method citations for MCAR, Holm correction, humidity derivation, and sparse just-in-time baseline/data citations.
- Metadata subagent: no mandatory metadata fixes for the high-risk/new entries. Notes:
  - `stekhoven2012missforest` is acceptable as 2012 because the formal Bioinformatics issue is 28(1), January 2012, despite online-first publication in 2011.
  - `chen2025exost` uses the 2025 arXiv submission year; current arXiv version is later but the year is acceptable.
  - `cao2026ecto` and `lee2026constrainedfusion` are arXiv preprints; no publisher metadata is available.
  - `song2026stdan` metadata are acceptable; `PLOS One` styling may be changed to `PLOS ONE` later if desired but is not mandatory.
- Citation-context subagent: one mandatory mismatch was found. Antarctic ERA5 temperature/wind validation citations should not be used to support a humidity-specific ERA5 bias claim.

## Changes Applied
- Added `rubin1976inference` for Missing Completely At Random (MCAR) terminology.
- Added `holm1979simple` for Holm multiple-comparison correction.
- Added `bolton1980computation` for the saturation-vapor-pressure approximation used in RH/q derivation.
- Added near-section citations for AntAWS, ERA5, SAITS, iTransformer-style baseline, and BRITS where these are described in the data and experimental-protocol sections.
- Softened/excised humidity-specific ERA5 bias claims not directly supported by the cited Antarctic ERA5 temperature/wind validation papers.
- Softened exogenous-fusion preprint language from broad consensus wording to "emerging studies suggest" / direct author-specific descriptions.

## Verification
- `paper/references.bib`: 33 unique BibTeX keys.
- Manuscript citations: 33 unique cited keys.
- No uncited BibTeX keys and no missing BibTeX entries.
- `paper/main.bbl`: 33 compiled references.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` passed.
- Log scan found no undefined references/citations, BibTeX warnings, overfull boxes, fatal errors, emergency stops, rerun warnings, or LaTeX warnings. Remaining underfull messages are non-blocking IEEE layout artifacts.

## Residual Judgment
- No known fabricated or unsupported citation remains.
- No citation is currently carrying a claim outside its verified scope.
- The remaining arXiv preprints are clearly presented as adjacent/emerging exogenous-conditioning literature rather than as settled core evidence.
