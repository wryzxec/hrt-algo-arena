# Text lane (issue #3): TF-IDF 1–2 grams + Ridge, drift-weighted

**Selection Sharpe: 3.34** (common 70/15/15 split, seed 42, ddof=0, n=1500). Source SHA 36aedd3.

Reproduce: `python -m arena.evaluate --model arena.models.text --out reports/text`
Runtime: ~4 s end to end (fit + validation + 85% refit + OOS), Apple M4 Pro CPU. No new dependencies.

## Method
- Word TF-IDF, 1–2 grams, `min_df=5`, sublinear tf; dollar amounts/numbers normalised to `AMT`/`NUM`. Vocabulary fit on fit data only.
- **Recent aggregation weighted by remaining drift.** Research finding: a headline moves the price with a
  roughly linear drift for ~10 bars, then stops, regardless of company named. A headline at bar *b* therefore
  only matters through drift still left after bar 49: weight 0 for *b* ≤ 39, 0.5 at 40, … 9.5 at 49.
  Session vector = Σ weight × headline TF-IDF. Ridge on that; intercept dropped so no-news sessions predict 0.
- Only `X` (bars/headlines ≤ 49) and `y` are used; headlines with bar > 49 are ignored defensively (tested).

## All trials (selection split)
| aggregation | alpha | Sharpe | note |
|---|---|---|---|
| recent | 1 | 1.665 | issue grid |
| recent | 10 | 2.092 | issue grid |
| recent | 100 | 2.762 | issue grid |
| all | 1 | 0.533 | issue grid |
| all | 10 | 0.672 | issue grid |
| all | 100 | 0.873 | issue grid |
| recent | 300 | 3.056 | beyond issue grid |
| recent | 1000 | 3.27 | beyond issue grid |
| recent | 3000 | 3.337 | beyond issue grid |
| recent | 10000 | 3.218 | beyond issue grid |
| recent | 30000 | 3.006 | beyond issue grid |

`all` = unweighted sum of every headline: ~0.9 — timing is what carries the signal.
Alpha was chosen on the selection split (curve is flat 1000–10000), so the reported Sharpe is slightly optimistic;
the audit split should be used to confirm. For reference, a one-hot *template* × drift Ridge on the same split scores 2.70.

## Limitations
- Ran locally on Python 3.15 with newer numpy/pandas/sklearn than the pinned versions in pyproject (pins have no 3.15 wheels); CI on 3.11 uses the pins.
- Linear only; drift length (10 bars) fixed from research on in-sample data, not re-estimated per fit.
