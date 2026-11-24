# Frozen interfaces

`arena.data.load_sessions("ins" | "oos") -> (X,y)`: X indexed by integer session;
open/high/low/close are 50-element lists; headlines and headline_bars are aligned lists.
y is a separate aligned return Series, or None for OOS. Never use session IDs as features.
Raw training prices contain 100 bars: filtering to <=49 is essential.

Model module exports `build()` returning an object with `fit(X,y)` and `predict(X)`.
Predict returns a finite 1D array preserving row order. All learned transforms fit only
on fit data; model code must not independently read parquet or labels. Seed 42.

`partitions(X.index)` returns fixed 70% train, 15% selection, 15% audit IDs. Workers use
first two only. Coordinator opens audit once after candidate freeze. This assumes
independent synthetic sessions; examine regime effects on development data if needed.

Run `python -m arena.evaluate --model arena.models.text --out reports/text` (replace lane).
Produces validation.csv keyed by session, metrics.json, and predictions.csv refitted on
85% development data. Audit remains untouched. Final all-label refit is coordinator work.
Report runtime, device, every trial, dependencies, source SHA and limitations in lane README.
Worker scope: their module, tests/test_<lane>.py, reports/<lane>/; optional lane requirements.
Large weights go in ignored artifacts/. Submit small predictions CSVs with the PR.

Submission columns exactly headline_id,pred; headline_id is session number, one per OOS
session, no dataframe index. Local Sharpe uses ddof=0 pending official helper verification.
