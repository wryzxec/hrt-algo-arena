# Shared validation gates (#6)

The evaluator now refuses to overwrite previous report files, rejects invalid session
identity/schema/nonfinite predictions, and records runtime, package versions, source and
artifact hashes plus selection PnL tails/concentration. The standalone validator checks
the exact 10,000 OOS IDs and prints the submission SHA256 without loading training labels.

Validation: 16 tests passed on local arm64 CPU, Python 3.9.6 with repository-pinned packages.
A fixed-configuration events reproduction scored selection Sharpe 3.122185731987107 on
1,500 sessions. Its OOS file is byte-identical to reports/events/predictions.csv:
6d070f87bf8c18cbadc4e9fe94cf398df5ea958a8db1393f698a63d7046d065b.
Top 1% of sessions account for 8.70% of absolute selection PnL. No model tuning or audit
scoring was performed. metrics.json records the measured runtime and committed source SHA.

Reproduce from repository root into a new directory each time:

```sh
python -m pytest -q
python -m arena.evaluate --model arena.models.events --out artifacts/coordinator-reproduction-02
python -m arena.validate artifacts/coordinator-reproduction-02/predictions.csv
```

Two verification runs used the same configuration: precommit (2.99 seconds, dirty source)
and committed-source reproduction recorded in metrics.json. Selection predictions also
match reports/events/validation.csv. No duplicated CSVs are committed in this report.

The recorded source fingerprint covers src/arena Python files; dependency versions are
recorded separately. Dirty-source status covers src and pyproject.toml. These are
reproduction aids, not proof of compliance by earlier exploratory scripts. In particular,
audit_scored=false describes only this evaluator invocation. Dashboard deadline/quota and
the official ddof convention remain unverified; this work neither uploads nor reserves an
attempt. Text PR #8 fit-time filtering and audit-provenance questions remain with its owner.
