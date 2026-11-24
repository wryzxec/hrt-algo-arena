"""Sector-filtered headline drift with price calibration and risk-aware position sizing."""
from collections import Counter
import re

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import SpectralClustering
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits


KERNEL = np.array([0.5] + [1.0] * 9 + [0.5])
NORMALIZE_NUMBER = re.compile(r"\$?\d+(?:\.\d+)?[%mb]?")


def parse_headline(text):
    words = text.split()
    if len(words) < 4:
        return "", ""
    start = 3 if words[2] in {"Co", "Group"} else 2
    phrase = NORMALIZE_NUMBER.sub("#", " ".join(words[start:]).lower())
    if "steps down unexpectedly" in phrase:
        key = "executive steps down"
    elif "addresses investor concerns" in phrase:
        key = "executive addresses concerns"
    elif phrase.startswith(("withdraws from ", "launches next-generation ")):
        key = " ".join(phrase.split()[:2])
    else:
        key = " ".join(phrase.split()[:3])
    return words[0], key


def observed_events(X):
    rows = []
    for texts, bars in zip(X["headlines"], X["headline_bars"]):
        rows.append([(int(b), *parse_headline(h)) for b, h in zip(bars, texts) if 0 <= b <= 49])
    return rows


def price_arrays(X):
    # Slice individual rows so longer arrays supplied by other callers are safe.
    close = np.asarray([np.asarray(row)[:50] for row in X["close"]], dtype=float)
    high = np.asarray([np.asarray(row)[:50] for row in X["high"]], dtype=float)
    low = np.asarray([np.asarray(row)[:50] for row in X["low"]], dtype=float)
    if close.shape[1] != 50 or not np.all(np.isfinite(close)) or np.any(close <= 0):
        raise ValueError("Expected 50 finite, positive closes per session")
    changes = np.diff(np.log(close), axis=1)
    ranges = np.log(high / low).mean(axis=1)
    return changes, ranges


class SectorDrift:
    def _learn_sectors(self, rows):
        names = sorted({name for row in rows for _, name, _ in row if name})
        if len(names) <= 5:
            return dict.fromkeys(names, 0)
        index = {name: j for j, name in enumerate(names)}
        counts = np.zeros((len(rows), len(names)))
        for i, row in enumerate(rows):
            for _, name, _ in row:
                if name:
                    counts[i, index[name]] += 1
        with threadpool_limits(limits=1):
            labels = SpectralClustering(
                n_clusters=5, affinity="precomputed", random_state=42
            ).fit_predict(counts.T @ counts)
        return dict(zip(names, labels))

    def _design(self, rows):
        sessions, starts, keys, weights = [], [], [], []
        for i, row in enumerate(rows):
            names = {name for _, name, _ in row if name in self.sectors}
            distinct = Counter(self.sectors[name] for name in names)
            top = max(distinct.values(), default=0)
            sectors = {s for s, count in distinct.items() if count == top}
            for bar, name, key in row:
                if key not in self.keys:
                    continue
                sector = self.sectors.get(name)
                # Unknown companies retain the sector-unconditional average effect.
                weight = 0.75 if sector is None else float(sector in sectors) / max(len(sectors), 1)
                if weight:
                    sessions.append(i)
                    starts.append(bar)
                    keys.append(self.keys[key])
                    weights.append(weight)
        session = np.asarray(sessions, dtype=int)
        start = np.asarray(starts, dtype=int)
        key = np.asarray(keys, dtype=int)
        weight = np.asarray(weights)
        width = max(1, len(self.keys))
        t = start[:, None] + np.arange(len(KERNEL))
        event, lag = np.where((t >= 1) & (t <= 49))
        path = csr_matrix(
            (weight[event] * KERNEL[lag], (49 * session[event] + t[event, lag] - 1, key[event])),
            shape=(49 * len(rows), width),
        )
        future = csr_matrix(
            (weight * np.clip(start - 39.5, 0, 10), (session, key)),
            shape=(len(rows), width),
        )
        return path, future

    def _state(self, X):
        changes, ranges = price_arrays(X)
        vol = np.clip(ranges / self.typical_range, 0.45, 2.0)
        path, future = self._design(observed_events(X))
        raw = (future @ self.effects.coef_) * vol
        fitted_path = self.effects.predict(path).reshape(len(X), 49) * vol[:, None]
        residual = changes - fitted_path
        features = np.column_stack((
            np.ones(len(X)), raw, vol - 1, residual.sum(axis=1), residual[:, -3:].sum(axis=1)
        ))
        variance = 10 * (self.noise * vol) ** 2
        # Expected squared return contains both return variance and squared mean.
        risk_weight = 10 * self.noise**2 / (variance + raw**2)
        return features, raw, risk_weight

    def fit(self, X, y):
        rows = observed_events(X)
        self.sectors = self._learn_sectors(rows)
        self.keys = {key: i for i, key in enumerate(sorted({k for row in rows for _, _, k in row if k}))}
        changes, ranges = price_arrays(X)
        self.typical_range = max(float(np.median(ranges)), 1e-8)
        vol = np.clip(ranges / self.typical_range, 0.45, 2.0)
        path, _ = self._design(rows)
        standardized = changes / vol[:, None]
        self.effects = Ridge(alpha=10.0, solver="lsqr", tol=1e-8).fit(path, standardized.ravel())
        errors = standardized.ravel() - self.effects.predict(path)
        self.noise = max(float(np.sqrt(np.mean(errors**2))), 1e-7)
        features, raw, _ = self._state(X)
        self.scale = features.std(axis=0)
        self.scale[self.scale < 1e-10] = 1.0
        self.calibration = Ridge(alpha=0.1 * len(X), fit_intercept=False).fit(
            features / self.scale, np.asarray(y, dtype=float) - raw
        )
        return self

    def predict(self, X):
        features, raw, weight = self._state(X)
        return (raw + self.calibration.predict(features / self.scale)) * weight


def build():
    return SectorDrift()
