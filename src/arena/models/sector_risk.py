"""Risk-weighted calibration of the sector headline drift forecast."""
import numpy as np
from sklearn.linear_model import Ridge

from arena.models.sector_drift import SectorDrift, price_arrays, observed_events


class SectorRisk(SectorDrift):
    def __init__(self, calibration_alpha=.01, weight_power=2., residual_features=0,
                 sizing_power=2.):
        self.calibration_alpha = calibration_alpha
        self.weight_power = weight_power
        self.residual_features = residual_features
        self.sizing_power = sizing_power

    def _state(self, X):
        features, raw, weight = super()._state(X)
        if self.residual_features:
            changes, ranges = price_arrays(X)
            vol = np.clip(ranges / self.typical_range, .45, 2.)
            path, _ = self._design(observed_events(X))
            residual = changes - self.effects.predict(path).reshape(len(X), 49) * vol[:, None]
            windows = np.column_stack([residual[:, -n:].sum(axis=1) for n in (1, 5, 10, 20)])
            features = np.column_stack((features, windows))
            if self.residual_features == 2:
                features = np.column_stack((features, windows * (vol[:, None] - 1)))
        return features, raw, weight

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
        return mu / (vol**self.sizing_power + mu**2/(10*self.noise**2))


def build():
    return SectorRisk(sizing_power=2.5)
