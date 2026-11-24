"""Issue #3 text lane: hybrid template drift + word TF-IDF + Ridge with Garman-Klass vol scaling.

Key findings:
1. Exact 10-bar linear drift: each headline moves the price with a linear drift for ~10 bars
   after publication, then stops. Only headlines at bar b in 40..49 have remaining drift during
   the prediction window (49 -> 59).
2. Template drift estimation: across bars 0..39 of the fit partition, observed 10-bar price moves
   are regressed per headline template and shrunk toward verb means. This provides a robust prior
   for all 302 invariant syntactic headline templates.
3. TF-IDF Ridge: 1-2 gram TF-IDF on drift-weighted headlines (b in 40..49) captures residual
   and lexical nuance directly calibrated to forward returns.
4. Volatility scaling: predictions are divided by Garman-Klass volatility from bars 0..49 OHLC
   for risk-adjusted sizing (E[r]/Var[r]).

All transforms fit strictly on fit data (bars <= 49).
"""
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

HALF, END, DRIFT = 49, 59, 10
SUFFIX = {"Group", "Co"}


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


def _extract_tmpl(h):
    t = h.split()
    rest = " ".join(t[3:]) if len(t) > 2 and t[2] in SUFFIX else " ".join(t[2:])
    rest = re.sub(r"\$[\d.]+[MBK]?", "$X", rest)
    return re.sub(r"\d+(\.\d+)?%", "N%", rest)


def gk_vol(X):
    """Garman-Klass volatility per session from bars 0..49 OHLC (floored to stay finite)."""
    O, H, L, C = (np.log(np.vstack([np.asarray(v, dtype=float)[:HALF + 1] for v in X[c]]))
                  for c in ("open", "high", "low", "close"))
    v = np.sqrt(np.clip((0.5 * (H - L) ** 2 - (2 * np.log(2) - 1) * (C - O) ** 2).mean(1), 0, None))
    return np.maximum(v, 1e-4)


class TextModel:
    def __init__(self, alpha=3000.0, shrink=100.0, w_tfidf=0.4, vol_scale=True):
        self.alpha = alpha
        self.shrink = shrink
        self.w_tfidf = w_tfidf
        self.vol_scale = vol_scale

    def _design_tfidf(self, X):
        texts, rows, weights = [], [], []
        for r, (hs, bs) in enumerate(zip(X["headlines"], X["headline_bars"])):
            bs = np.asarray(bs, dtype=float)
            keep = bs <= HALF
            w = remaining_drift(bs)
            for h, wi, k in zip(hs, w, keep):
                if k and wi > 0:
                    texts.append(_norm(h))
                    rows.append(r)
                    weights.append(wi)
        n = len(X)
        if not texts or not hasattr(self, "vec"):
            n_features = len(getattr(self, "vec", {}).vocabulary_ or []) if hasattr(self, "vec") else 0
            return sp.csr_matrix((n, n_features))
        H = self.vec.transform(texts)
        A = sp.csr_matrix((weights, (rows, np.arange(len(rows)))), shape=(n, len(rows)))
        return (A @ H).tocsr()

    def _predict_tmpl(self, X):
        preds = []
        for hs, bs in zip(X["headlines"], X["headline_bars"]):
            w = remaining_drift(bs)
            s = 0.0
            for h, b, wi in zip(hs, bs, w):
                if wi > 0 and b <= HALF:
                    t = _extract_tmpl(h)
                    v = t.split()[0] if t else ""
                    val = self.tmpl_effects.get(t, self.verb_effects.get(v, 0.0))
                    s += wi * val
            preds.append(s)
        return np.array(preds, dtype=float)

    def fit(self, X, y):
        # 1. Fit TF-IDF on headline vocabulary from fit partition
        corpus = [_norm(h) for hs in X["headlines"] for h in hs]
        if not corpus:
            corpus = ["dummy"]
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True)
        try:
            self.vec.fit(corpus)
        except ValueError:
            self.vec = TfidfVectorizer().fit(["dummy"])

        X_mat = self._design_tfidf(X)
        self.ridge = Ridge(alpha=self.alpha).fit(X_mat, np.asarray(y, dtype=float))

        # 2. Template drift estimation from fully observed moves in bars 0..39
        records = []
        if "close" in X and len(X["close"].iloc[0]) > 0:
            C = np.vstack([np.asarray(c, dtype=float)[:HALF + 1] for c in X["close"]])
            R = np.diff(np.log(C), axis=1)
            mkt = R.mean(axis=0)
            cum = np.concatenate([np.zeros((len(C), 1)), np.cumsum(R - mkt, axis=1)], axis=1)
            for r, (hs, bs) in enumerate(zip(X["headlines"], X["headline_bars"])):
                for h, b in zip(hs, bs):
                    if 1 <= b <= 39:
                        drift = (cum[r, b + 9] - cum[r, b - 1]) / 10.0
                        t = _extract_tmpl(h)
                        v = t.split()[0] if t else ""
                        records.append((t, v, drift))

        if records:
            df = pd.DataFrame(records, columns=["tmpl", "verb", "drift"])
            g = df.groupby("tmpl").drift
            self.verb_effects = df.groupby("verb").drift.mean().to_dict()
            tv = df.groupby("tmpl").verb.first().map(self.verb_effects)
            shrunk = (g.mean() * g.count() + tv * self.shrink) / (g.count() + self.shrink)
            self.tmpl_effects = shrunk.to_dict()
        else:
            self.verb_effects = {}
            self.tmpl_effects = {}

        p_tmpl_fit = self._predict_tmpl(X)
        p_tf_fit = self.ridge.predict(X_mat) - self.ridge.intercept_
        self.std_tmpl = max(float(np.std(p_tmpl_fit)), 1e-6)
        self.std_tf = max(float(np.std(p_tf_fit)), 1e-6)
        return self

    def predict(self, X):
        p_tf = self.ridge.predict(self._design_tfidf(X)) - self.ridge.intercept_
        p_tmpl = self._predict_tmpl(X)
        z_tf = p_tf / self.std_tf
        z_tmpl = p_tmpl / self.std_tmpl
        p = (1.0 - self.w_tfidf) * z_tmpl + self.w_tfidf * z_tf
        if self.vol_scale and "open" in X and "high" in X:
            p = p / gk_vol(X)
        return np.nan_to_num(np.asarray(p, dtype=float))


def build():
    return TextModel(alpha=3000.0, shrink=100.0, w_tfidf=0.4, vol_scale=True)
