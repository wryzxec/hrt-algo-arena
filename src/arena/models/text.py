"""Issue #3 text lane: word TF-IDF (1-2 grams) + Ridge, with drift-weighted aggregation.

Key finding (from in-sample research on bars <= 49 + labels only via this model's fit):
each headline moves the price with a roughly linear drift for ~10 bars after it is
published, then stops. So a headline at bar b only matters for the 49->59 return through
the part of its drift that is still left after bar 49. Each headline's TF-IDF vector is
weighted by that remaining-drift weight (0 for b <= 39, 0.5 at b = 40, rising to 9.5 at b = 49) and
summed per session ("recent" aggregation). An unweighted "all" aggregate is available too.
Numbers are normalised so "$500M" / "7%" share tokens. Vocabulary is fit on fit data only.

Position sizing: the news drift scales with the stock's volatility, and the Sharpe-optimal
position is E[r]/Var[r] ~ (signal * vol) / vol^2, so predictions are divided by a
Garman-Klass volatility estimate from bars 0-49 OHLC (better forecaster of next-10-bar vol
than close-to-close). In-sample 5-fold CV: +0.04 Sharpe, positive in all folds.
"""
import re

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

HALF, END, DRIFT = 49, 59, 10


def remaining_drift(b):
    b = np.asarray(b, dtype=float)
    w = np.zeros_like(b)
    for k in range(DRIFT + 1):
        wk = 0.5 if k in (0, DRIFT) else 1.0
        w += wk * ((b + k > HALF) & (b + k <= END))
    return w


def _norm(h):
    h = re.sub(r"\$[\d.]+\s?[MBK]?", " AMT ", h)
    return re.sub(r"\d+(\.\d+)?%?", " NUM ", h)


def gk_vol(X):
    """Garman-Klass volatility per session from bars 0..49 OHLC (floored to stay finite)."""
    O, H, L, C = (np.log(np.vstack([np.asarray(v, dtype=float)[:HALF + 1] for v in X[c]]))
                  for c in ("open", "high", "low", "close"))
    v = np.sqrt(np.clip((0.5 * (H - L) ** 2 - (2 * np.log(2) - 1) * (C - O) ** 2).mean(1), 0, None))
    return np.maximum(v, 1e-4)


class TextModel:
    def __init__(self, alpha=10.0, mode="recent", vol_scale=True):
        self.alpha, self.mode, self.vol_scale = alpha, mode, vol_scale

    def _design(self, X):
        texts, rows, weights = [], [], []
        for r, (hs, bs) in enumerate(zip(X["headlines"], X["headline_bars"])):
            bs = np.asarray(bs, dtype=float)
            keep = bs <= HALF  # defensive: never use anything after bar 49
            w = remaining_drift(bs) if self.mode == "recent" else np.ones(len(bs))
            for h, wi, k in zip(hs, w, keep):
                if k and wi > 0:
                    texts.append(_norm(h)); rows.append(r); weights.append(wi)
        n = len(X)
        if not texts:
            return sp.csr_matrix((n, len(self.vec.vocabulary_)))
        H = self.vec.transform(texts)
        A = sp.csr_matrix((weights, (rows, np.arange(len(rows)))), shape=(n, len(rows)))
        return (A @ H).tocsr()

    def fit(self, X, y):
        corpus = [_norm(h) for hs in X["headlines"] for h in hs]
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True).fit(corpus)
        self.ridge = Ridge(alpha=self.alpha).fit(self._design(X), np.asarray(y, dtype=float))
        return self

    def predict(self, X):
        p = self.ridge.predict(self._design(X)) - self.ridge.intercept_  # no-news sessions -> 0
        if self.vol_scale:
            p = p / gk_vol(X)
        return np.nan_to_num(np.asarray(p, dtype=float))


def build():
    return TextModel(alpha=3000.0, mode="recent")
