import numpy as np
import pandas as pd
from arena.models.text import build, remaining_drift


def _toy(n=40, seed=0):
    rng = np.random.default_rng(seed)
    good, bad = "Acme Corp secures $500M contract", "Acme Corp misses quarterly revenue estimates"
    rows, y = [], []
    for i in range(n):
        g = i % 2 == 0
        rows.append(dict(open=[1.0] * 50, high=[1.0] * 50, low=[1.0] * 50, close=[1.0] * 50,
                         headlines=[good if g else bad] * 5, headline_bars=[45, 46, 47, 48, 49]))
        y.append((0.01 if g else -0.01) + rng.normal(0, 1e-3))
    return pd.DataFrame(rows), pd.Series(y)


def test_remaining_drift():
    w = remaining_drift([0, 39, 40, 41, 49])
    assert w[0] == 0 and w[1] == 0 and w[2] == 0.5 and w[3] == 1.5 and w[4] == 9.5


def test_fit_predict_finite_and_signed():
    X, y = _toy()
    m = build(); m.alpha = 0.1
    p = m.fit(X, y).predict(X)
    assert p.shape == (len(X),) and np.isfinite(p).all()
    assert np.corrcoef(p, y)[0, 1] > 0.9


def test_empty_and_old_headlines_predict_zero():
    X, y = _toy()
    m = build().fit(X, y)
    Z = X.iloc[:2].copy()
    Z.at[Z.index[0], "headlines"] = []; Z.at[Z.index[0], "headline_bars"] = []
    Z.at[Z.index[1], "headline_bars"] = [1, 2, 3, 4, 5]  # too old: drift finished before bar 49
    assert np.allclose(m.predict(Z), 0.0)


def test_future_headlines_ignored():
    X, y = _toy()
    m = build().fit(X, y)
    Z = X.iloc[:1].copy()
    Z.at[Z.index[0], "headline_bars"] = [55, 56, 57, 58, 59]
    assert np.allclose(m.predict(Z), 0.0)
