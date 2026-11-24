# Sector-aware headline drift

This optional candidate for integration issue #4 scored **4.22 Sharpe** on the shared 1,500-session selection set. The volatility-sized text candidate in PR #8 scored **3.48** on those same sessions. The paired difference was **+0.75** with a 95% session-bootstrap interval of **+0.17 to +1.31** (2,500 paired resamples, seed 42, assuming independent sessions). The interval is conditional on the fitted models. The shared selection set has also been used to choose earlier models.

## How the model trades

Companies mentioned together in training headlines form five clusters. For each session the model counts distinct companies by cluster. It gives the most represented cluster the largest weight. Earlier price changes in training show a roughly ten-bar response to a headline about that cluster. It estimates each headline template's effect from the part of that response visible before bar 50. At bar 49, a headline from bar 40 has half a bar of drift left, while one from bar 49 has 9.5 bars left.

The model estimates volatility from observed high–low ranges, calibrates the return forecast using labeled training sessions and recent price changes unexplained by headlines, and sizes the position against estimated squared return. All parameters are fit on the training partition. Inference can run on one session at a time and uses its first 50 bars. Session IDs only align rows.

## Comparison and model choice

Four-fold cross-validation inside the 7,000-session training partition produced the following pooled Sharpes. These development comparisons informed model choice. The shared selection set was opened once for this candidate after the model was frozen.

| Development variant | Pooled Sharpe |
| --- | ---: |
| Earlier drift-weighted TF-IDF before volatility sizing | 2.38 |
| Historical price reaction, all sectors | 2.86 |
| Historical price reaction, dominant sector | 3.09 |
| Dominant sector, inverse volatility | 3.15 |
| Dominant sector, inverse squared volatility | 3.10 |
| Inferred per-company sensitivity, raw positions | 2.98 |
| Inferred sensitivity, inverse variance | 3.15 |
| Inferred sensitivity, second-moment sizing | 3.23 |
| Inferred sensitivity with supervised calibration | 3.39 |
| Direct sector model with calibration, first implementation | 3.38 |
| Frozen direct sector model | **3.37** |

The frozen model standardizes historical price moves by session volatility before learning template effects. Its fold Sharpes were 3.39, 3.23, 3.39 and 3.47. It needs one sparse impact regression and one small calibration regression. The latent-company alternative required repeated inference over 24 role assignments for a development gain of 0.02. Sector classification, response duration and the risk model describe this synthetic data and may generalize differently to new regimes.

## Reproduction and export

Run from the repository root after installing its declared dependencies.

```sh
python -m arena.evaluate --model arena.models.sector_drift --out reports/sector_drift
python reports/sector_drift/cross_validate.py
python -m pytest -q
```

The common evaluator reports 4.22324 with population standard deviation. The supplied challenge helper uses sample standard deviation and adds 1e-12 to the denominator, giving 4.22184. Selection prediction-to-return correlation was 0.259. The ten largest profitable sessions contributed 14.3% of net selection PnL. The paired selection forecasts correlated 0.744 with PR #8. The fixed seed was 42.

The 8,500-session development refit produced 10,000 finite predictions with the exact unique official IDs. The official submission writer accepted the export. Independent loading and the repository evaluator produced byte-identical CSVs on the local Apple Silicon CPU. Fit, selection scoring, refit and export took about 2.5 seconds with the prefiltered loader and about 5 seconds with the shared evaluator. The local environment used Python 3.15. CI uses the repository's Python 3.11 environment.

Model source commit `5abb2e3b1261f35f9ad3eb06ba4d73347a8f2eb9`. CSV SHA-256 `f9e2ef47c5e4547bca2e92d3a7045adb224d81e56ac9593d4e9b8f09cc5e5c51`. The coordinator owns the reserved 15% audit, final all-label refit and dashboard submission.
