# Reference Audit — 2026-05-27

Scope: online verification of all 30 entries in `paper/references.bib` against publisher, arXiv, OpenReview, NeurIPS, PMLR, or DOI-indexed pages.

## Corrections Applied

| Key | Issue | Correction | Verification source |
|---|---|---|---|
| `wang2023antaws` | Several first names were wrong (`Yixiao`, `Wenjing`, `Dominik`, `Leah`, `Zhen`, `Yi`) | Corrected to Yetang Wang, Wentao Ning, David Mikolajczyk, Lee J. Welhouse, Zhaosheng Zhai, Yuqi Sun | ESSD article page, DOI `10.5194/essd-15-411-2023` |
| `vantiggelen2025imau` | Page range ended at 4960 | Corrected page range to 4933--4955 | ESSD/UU article page, DOI `10.5194/essd-17-4933-2025` |
| `fausto2026promicegcnet` | Author listed as `Razan Bahbah` | Corrected to `Rasmus Bahbah` | University of Copenhagen record and ESSD DOI `10.5194/essd-18-2829-2026` |
| `zhou2021informer` | Used arXiv DOI despite official AAAI DOI; volume/pages missing | Added AAAI volume 35(12), pages 11106--11115, and DOI `10.1609/aaai.v35i12.17325` | AAAI proceedings page |

## Verified Without Changes

- Polar/AWS and ERA5 references: `hersbach2020era5`, `fausto2021promice`, `zhu2021era5antarctica`, `tetzner2019era5antarcticpeninsula`, `catonharrison2022antarcticwinds`, `ma2025antarcticsat`.
- Imputation references: `cao2018brits`, `du2023saits`, `du2024tsibench`, `stekhoven2012missforest`, `che2018grud`, `tashiro2021csdi`, `fang2020imputationsurvey`, `wang2024imputationsurvey`, `fortuin2020gpvae`.
- Transformer/time-series architecture references: `liu2024itransformer`, `vaswani2017attention`, `zhou2022fedformer`, `nie2023patchtst`, `wu2023timesnet`, `wu2021autoformer`.
- Exogenous/reanalysis-conditioning references: `lee2026constrainedfusion`, `cao2026ecto`, `chen2025exost`, `song2026stdan`, `li2026era5superresolution`.

## Notes

- `liu2024itransformer` is verified through OpenReview as ICLR 2024 spotlight; no publisher DOI is needed.
- Several conference papers keep arXiv DOIs for stable resolution when the proceedings entry has no DOI or the arXiv version is the canonical public identifier.
- `and others` is retained for long author lists where the current bibliography style abbreviates correctly.

## Primary Online Sources Used

- AntAWS ESSD article: https://essd.copernicus.org/articles/15/411/2023/index.html
- ERA5 QJRMS DOI reference: https://doi.org/10.1002/qj.3803
- BRITS NeurIPS page: https://papers.nips.cc/paper/7911-brits-bidirectional-recurrent-imputation-for-time-series
- SAITS ScienceDirect page: https://www.sciencedirect.com/science/article/pii/S0957417423001203
- iTransformer OpenReview page: https://openreview.net/forum?id=JePfAI8fah
- IMAU Antarctic AWS ESSD article: https://essd.copernicus.org/articles/17/4933/2025/
- PROMICE/GC-NET article record: https://researchprofiles.ku.dk/en/publications/promice-gc-net-automatic-weather-station-data/
- Informer AAAI proceedings page: https://ojs.aaai.org/index.php/AAAI/article/view/17325
- ArXiv pages for preprints and arXiv-DOI records: https://arxiv.org/
