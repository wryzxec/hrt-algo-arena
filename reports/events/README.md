# Event model

The single Ridge trial (alpha 500) achieved an annualized selection Sharpe of
3.12219 on 1,500 sessions. A constant forecast of the 7,000 training returns'
mean scored −0.51271 on the same sessions. Both use the repository metric
with population standard deviation. The full evaluation took 15.51 seconds
on an Apple Silicon CPU, including the development refit and OOS export.

Training headlines contained recurring contract, guidance, product, legal,
operations and corporate-event phrases. Contract sizes, buybacks and percentages
supply magnitude features. The training inspection found no explicit no/not/never/
denies/without tokens. The parser handles local negation and groups unseen
templates under “other”. Company and session identifiers are excluded.

Per-template features count events, weight them by 5- and 15-bar half-lives,
and sum close-to-close moves from each event bar to bar 49. The latest headline
contributes its category, age, sign and price response. The model learns
feature scales and Ridge coefficients on the fit partition. Selection measures
synthetic held-out sessions. Performance on the competition dashboard is unknown.

## Reproduction

From repository root with Python 3.12 and the pinned project dependencies installed:

```sh
python -m arena.evaluate --model arena.models.events --out reports/events
python -m pytest -q
```

Source commit 452804d22361f53b6a3d4af4b1ecbe1ccd60e354. Seed 42.
The evaluator fits on 70% for validation.csv and on 85% for predictions.csv.
No audit outcomes were used. metrics.json records the selection score.
Leakage tests alter future prices, append an unreadable future headline, inject
a returns column, change IDs and reverse row order. A separate test checks the
bar-49 response boundary, half-life, latest event, magnitude and negation.
