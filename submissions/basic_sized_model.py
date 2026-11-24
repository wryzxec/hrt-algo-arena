"""Basic model + Sharpe-optimal sizing. No momentum, no regression beyond one scale factor.

1. s   = sum over main-sector headlines (bars <= 49) of effect[template] x remaining_drift(bar)
         (effect = plain average next-10-bar drift per bar in vol units; see basic_model.py)
2. m   = k * s, where k is one least-squares number linking s to return/vol on in-sample sessions
3. pred = m / (vol * (Var(return/vol) + m^2))      i.e. E[r] / E[r^2], which maximises Sharpe

5-fold CV: 3.399 (seed 0), 3.386 (seed 1)  vs 3.327 / 3.315 for basic_template.csv (scored 3.4).

    .venv/bin/python submissions/basic_sized_model.py    # CV + write basic_sized.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from basic_model import predict, template_effects  # noqa: E402
from sector_model import DATA, load_all, sharpe  # noqa: E402


def fit(ins, y, idx):
    eff = template_effects(ins, idx)
    g = ins["gk"][idx]
    s = predict(ins, idx, eff) * g                      # raw signal (vol units)
    ya = y[idx] / g
    return eff, (s @ ya) / (s @ s), ya.var()


def predict_sized(d, idx, eff, k, var_y):
    g = d["gk"][idx]
    m = k * predict(d, idx, eff) * g                    # E[return / vol]
    return m / (g * (var_y + m ** 2))


if __name__ == "__main__":
    ins, oos, y = load_all()
    N = len(ins["S"])
    for seed in (0, 1):
        folds = list(KFold(5, shuffle=True, random_state=seed).split(np.arange(N)))
        p = np.zeros(N)
        for a, b in folds:
            p[b] = predict_sized(ins, b, *fit(ins, y, a))
        print(f"seed {seed}: 5-fold Sharpe {sharpe(p, y):.3f}  folds {[round(sharpe(p[b], y[b]), 2) for _, b in folds]}")

    allo = np.arange(len(oos["S"]))
    eff, k, var_y = fit(ins, y, np.arange(N))
    pred = predict_sized(oos, allo, eff, k, var_y)
    out = Path(__file__).resolve().parent / "basic_sized.csv"
    pd.DataFrame({"headline_id": oos["S"].astype(np.int64), "pred": pred}).to_csv(out, index=False, lineterminator="\n")

    sub = pd.read_csv(out)
    assert list(sub.columns) == ["headline_id", "pred"], "wrong columns"
    assert len(sub) == 10000 and sub.headline_id.is_unique, "wrong row count or duplicate ids"
    assert set(sub.headline_id) == set(pd.read_parquet(DATA / "oos_bars.parquet").session), "ids do not match OOS"
    assert np.isfinite(sub.pred).all(), "non-finite predictions"
    basic = pd.read_csv(Path(__file__).resolve().parent / "basic_template.csv").pred
    print(f"wrote {out}: rows={len(sub)} k={k:.3f} corr with basic_template={np.corrcoef(sub.pred, basic)[0, 1]:.3f}")
