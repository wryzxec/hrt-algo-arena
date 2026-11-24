# Second iteration

Explicit user request authorized continued experiments. Frozen candidate selection Sharpe is 4.27153238 versus previous 4.26962215 (+0.00191023). Training-only four-fold pooled CV increased from 3.38804551 to 3.39691048; three of four folds improved. This is a tiny exploratory gain, not established out-of-sample or leaderboard improvement. Repeated development comparisons induce selection bias. Audit labels were not used; inherited provenance limitations still apply.

The only active forecast change is volatility exponent 2.5 instead of 2 in position sizing. Mean calibration remains unchanged. Optional residual features are retained only to reproduce rejected experiments; defaults disable them. SectorRisk() retains the prior baseline; build() selects the frozen new candidate.

## Complete trial record

1. residual_cv.json: nine configurations, residual windows 1/5/10/20, optional interactions with relative volatility, alpha 0.001/0.01/0.1, fixed calibration weight power 2. Baseline won at 3.38804551; best augmented candidate 3.35975797. All augmented candidates rejected without selection evaluation. Runtime 30.26 seconds.
2. sizing_cv.json: twelve configurations, baseline mean forecast, volatility powers 2/2.5/3 and squared-mean coefficients 1/2/4/8. Winner power 2.5, coefficient 1. Runtime 3.05 seconds. Grid announced before running; winner frozen in issue #13 before one selection evaluation.

Both stages used only fixed 7000 training IDs and four shuffled folds with seed 42. Features stop at bar 49. No ID features, future labels, or audit scoring. Existing fallback artifacts remain intact.

## Validation

30 tests passed, including future-data exclusion and batch/ID invariance across feature variants and sizing powers. Native matrix-operation runtime warnings occurred, as in prior runs; all scores and exported predictions were finite. Evaluation on Apple Silicon CPU took 4.33 seconds. Metrics record dirty source plus exact source hashes. Export refit uses 8500 development sessions; all 10000 OOS IDs validate. CSV SHA256: caa925693a93599ce70da17e8e9a5b209aa092900e30f94d4d16d6b93a97f94e. No upload performed.

## Reproduce in fresh directories

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/sector_risk_search.py --residual --out artifacts/sector-risk-iteration2-residual
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/sector_risk_search.py --sizing --out artifacts/sector-risk-iteration2-sizing
.venv/bin/python -m arena.evaluate --model arena.models.sector_risk --out artifacts/sector-risk-iteration2-frozen
.venv/bin/python -m arena.validate artifacts/sector-risk-iteration2-frozen/predictions.csv
.venv/bin/python -m pytest -q
```
