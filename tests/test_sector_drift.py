import numpy as np
import pandas as pd

from arena.models.sector_drift import build


def sample():
    rng = np.random.default_rng(42)
    records, targets = [], []
    for i in range(60):
        direction = (-1) ** i
        returns = rng.normal(0, 0.001, 50)
        returns[6:16] += 0.001 * direction
        close = np.exp(np.cumsum(returns))
        headline = ("Acme Labs signs multi-year partnership" if direction > 0
                    else "Acme Labs loses key contract")
        records.append(dict(close=close, high=close * 1.001, low=close / 1.001,
                            headlines=[headline, headline], headline_bars=[5, 49]))
        targets.append(direction * 0.009)
    return pd.DataFrame(records), np.array(targets)


def test_cutoff_applies_to_fit_and_predict():
    X, y = sample()
    altered = X.copy()
    for col in ["close", "high", "low"]:
        altered[col] = [np.r_[x, np.nan, 1e12] for x in X[col]]
    altered["headlines"] = [x + ["Leak Inc gains ground"] for x in X.headlines]
    altered["headline_bars"] = [x + [50] for x in X.headline_bars]
    altered["returns"] = -y
    model = build().fit(X, y)
    expected = model.predict(X)
    np.testing.assert_allclose(build().fit(altered, y).predict(altered), expected)
    assert np.corrcoef(expected, y)[0, 1] > 0.9


def test_order_identity_and_batch_independence():
    X, y = sample()
    model = build().fit(X, y)
    expected = model.predict(X)
    changed = X.iloc[[19, 3]].copy()
    changed.index = [-8, 500000]
    np.testing.assert_allclose(model.predict(changed), expected[[19, 3]])
    np.testing.assert_allclose(model.predict(changed.iloc[:1]), expected[[19]])


def test_no_news_and_unknown_news_are_finite():
    X, y = sample()
    empty = X.assign(headlines=[[] for _ in y], headline_bars=[[] for _ in y])
    model = build().fit(empty, y)
    assert np.isfinite(model.predict(X)).all()
    assert np.isfinite(build().fit(X, y).predict(empty)).all()
