# HRT Algo Arena — 45-minute swarm sprint

Predict bar 49→59 synthetic-stock returns from bars/headlines 0–49.
Score: annualized Sharpe of prediction × return. Notebook limit: **10 uploads**.

Read [onboarding](docs/SWARM.md), [strategy](docs/STRATEGY.md), [contracts](docs/CONTRACTS.md),
and [assigned tasks](docs/TASKS.md).

```sh
git pull --ff-only
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pytest -q
```

Models live in `src/arena/models/`; small results in `reports/<lane>/`; ignored weights
and caches in `artifacts/`. Root parquet files/notebook are read-only inputs.
Official sessions.py is absent; local loader implements the notebook's documented inputs.
Dashboard rules, deadline, remaining quota and official std convention need verification.

Reproduce a lane into a new directory to preserve previous results:

```sh
python -m arena.evaluate --model arena.models.events --out artifacts/events-reproduction-01
python -m arena.validate artifacts/events-reproduction-01/predictions.csv
```

The report records source/artifact hashes, runtime, environment and selection PnL
concentration. The CSV check enforces the expected 10,000 session IDs and prints its
SHA256; dashboard quota/deadline and actual upload receipt remain separate requirements.
