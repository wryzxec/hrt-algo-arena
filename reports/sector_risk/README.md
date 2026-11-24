# Sector risk calibration

Closes #13. Depends on sector_drift from PR #11; merge that dependency first. This candidate increases the common 1,500-session selection Sharpe from **4.22324347 to 4.26962215** (+0.04638). Four-fold CV inside the fixed 7,000 training IDs increased from **3.37093772 to 3.38804551**. The gain is small, not evidence of a statistically established or leaderboard improvement.

The calibration regression now weights training examples by inverse estimated variance, with weights normalized to mean one. Ridge alpha is 0.01 times sample count. Position sizing uses the calibrated conditional mean in the second-moment denominator, instead of the uncalibrated news forecast. No new package dependencies.

## Complete experiment history

User explicitly authorized continuing beyond the original cutoff. First stage: baseline, nine calibrated-mean sizing choices (volatility powers 1/1.5/2, squared-mean weights 0/0.5/1), and three event blends (10/25/50 percent). All comparisons used four-fold training-only CV with seed 42. Best first-stage CV 3.38136; blends did not improve the baseline. Results: cv.json.

Second stage: nine calibration choices with inverse-volatility weighting powers 0/1/2 and alpha multipliers 0.01/0.1/1. Best CV 3.38805 at power 2, alpha 0.01. Results: calibration_cv.json. These adaptive development comparisons reuse CV folds, so the winning CV estimate is selection-biased. Configuration was frozen before a single selection evaluation. Audit outcomes were not scored or used by these experiments. The inherited sector model's broader historical research-provenance limitations in PR #11 still apply.

## Validation and artifacts

20 tests passed before merging current main; future price/headline cutoff and batch/ID invariance tested for this model. The evaluation took 4.37 seconds on Apple Silicon CPU, Python 3.9.6 with repository-pinned dependencies. Some native linear algebra warnings appeared, but predictions were finite and validation passed. Source was uncommitted during the initial experiment and metrics explicitly mark it dirty.

Export: frozen/predictions.csv, refit on 8,500 development sessions. All 10,000 exact expected OOS IDs, correct columns, finite predictions. SHA256: a463593bf10902f6b9eff98f08d3c035741409a60a11b9cf3624a7b8868f7101. No upload or leaderboard score. The harness uses ddof=0; official sample-standard-deviation scoring is slightly lower.

## Reproduction

```sh
PYTHONPATH=src .venv/bin/python scripts/sector_risk_search.py --out artifacts/sector-risk-cv-reproduction
PYTHONPATH=src .venv/bin/python scripts/sector_risk_search.py --calibration --out artifacts/sector-risk-calibration-reproduction
.venv/bin/python -m arena.evaluate --model arena.models.sector_risk --out artifacts/sector-risk-reproduction
.venv/bin/python -m arena.validate artifacts/sector-risk-reproduction/predictions.csv
.venv/bin/python -m pytest -q
```
