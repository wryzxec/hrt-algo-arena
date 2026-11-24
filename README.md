# HRT Algo Arena — 45-minute swarm sprint

Predict bar 49→59 synthetic-stock returns from bars/headlines 0–49.
Score: annualized Sharpe of prediction × return. Notebook limit: **10 uploads**.

Read [onboarding](docs/SWARM.md), [strategy](docs/STRATEGY.md), [contracts](docs/CONTRACTS.md),
and [assigned tasks](docs/TASKS.md).

```sh
git pull --ff-only
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

Models live in `src/arena/models/`; small results in `reports/<lane>/`; ignored weights
and caches in `artifacts/`. Root parquet files/notebook are read-only inputs.
Official sessions.py is absent; local loader implements the notebook's documented inputs.
Dashboard rules, deadline, remaining quota and official std convention need verification.
