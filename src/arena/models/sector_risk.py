"""Risk-weighted calibration of the sector headline drift forecast."""
import numpy as np
from sklearn.linear_model import Ridge

from arena.models.sector_drift import SectorDrift, price_arrays


class SectorRisk(SectorDrift):
    def __init__(self, calibration_alpha=.01, weight_power=2.):
        self.calibration_alpha = calibration_alpha
        self.weight_power = weight_power

    def fit(self, X, y):
        super().fit(X, y)
        features, raw, _ = self._state(X)
        _, ranges = price_arrays(X)
        vol = np.clip(ranges / self.typical_range, .45, 2.)
        weights = vol**(-self.weight_power)
        weights /= weights.mean()
        self.calibration = Ridge(alpha=self.calibration_alpha*len(X), fit_intercept=False).fit(
            features / self.scale, np.asarray(y, dtype=float)-raw, sample_weight=weights)
        return self

    def predict(self, X):
        features, raw, _ = self._state(X)
        mu = raw + self.calibration.predict(features / self.scale)
        _, ranges = price_arrays(X)
        vol = np.clip(ranges / self.typical_range, .45, 2.)
        return mu / (vol**2 + mu**2/(10*self.noise**2))


def build():
    return SectorRisk()
