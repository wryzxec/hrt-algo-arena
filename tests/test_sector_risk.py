import numpy as np
import pytest

from arena.models.sector_risk import SectorRisk
from test_sector_drift import sample


@pytest.mark.parametrize("variant", [0, 1, 2])
@pytest.mark.parametrize("power", [2., 2.5])
def test_risk_model_fit_predict_cutoff_and_batch_invariance(variant, power):
    X, y = sample()
    altered = X.copy()
    for column in ("close", "high", "low"):
        altered[column] = [np.r_[row, np.nan, 1e10] for row in X[column]]
    altered["headlines"] = [row + ["Leak Inc wins major contract"] for row in X.headlines]
    altered["headline_bars"] = [row + [50] for row in X.headline_bars]
    model = SectorRisk(residual_features=variant, sizing_power=power).fit(X, y)
    expected = model.predict(X)
    np.testing.assert_allclose(SectorRisk(residual_features=variant, sizing_power=power).fit(altered, y).predict(altered), expected)
    subset = X.iloc[[8, 2]].copy()
    subset.index = [-10, 900000]
    np.testing.assert_allclose(model.predict(subset), expected[[8, 2]])
    assert np.isfinite(expected).all()
