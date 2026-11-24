import numpy as np

from arena.models.sector_risk import build
from test_sector_drift import sample


def test_risk_model_fit_predict_cutoff_and_batch_invariance():
    X, y = sample()
    altered = X.copy()
    for column in ("close", "high", "low"):
        altered[column] = [np.r_[row, np.nan, 1e10] for row in X[column]]
    altered["headlines"] = [row + ["Leak Inc wins major contract"] for row in X.headlines]
    altered["headline_bars"] = [row + [50] for row in X.headline_bars]
    model = build().fit(X, y)
    expected = model.predict(X)
    np.testing.assert_allclose(build().fit(altered, y).predict(altered), expected)
    subset = X.iloc[[8, 2]].copy()
    subset.index = [-10, 900000]
    np.testing.assert_allclose(model.predict(subset), expected[[8, 2]])
    assert np.isfinite(expected).all()
