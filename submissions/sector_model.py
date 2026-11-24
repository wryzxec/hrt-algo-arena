"""Sector news-drift model with residual momentum and Sharpe-optimal sizing.

Self-contained implementation (reads only data/*.parquet). Used by make_submission.py.

How it works
------------
1. Sectors. The 80 companies form 5 sectors of 16 (spectral clustering of how often companies
   appear in the same in-sample session). A session's stock belongs to the sector with the most
   distinct companies among its bar 0-49 headlines (ties -> more headlines). Only that sector's
   headlines move the price; the rest are dropped.
2. Template drift. Headlines are reduced to templates (company removed, "$500M" -> "$X",
   "7%" -> "N%"). A headline at bar b drifts the price on bars b..b+10 with weights
   0.5, 1 x 9, 0.5, proportional to the stock's volatility. One coefficient per template is fitted
   jointly (sparse ridge) on vol-normalised, market-adjusted bar returns from all labelled
   sessions' bars 1-99 plus the prediction sessions' bars 1-49 (inputs only, no labels).
3. News signal = sum over main-sector headlines of coef x drift bars left after bar 49.
4. Residual momentum. Bar returns 1-49 not explained by the headlines, summed with exponential
   decay (half-lives 1, 2, 3, 5, 10, 25, 1000 bars).
5. Ridge (alpha 50, standardised features) of return/vol on [news signal, 7 residual sums] gives
   m = E[r/vol | x]. Position = m / (vol * (Var(r/vol) + m^2)), i.e. E[r]/E[r^2], which maximises
   Sharpe. vol = Garman-Klass volatility from bars 0-49 OHLC.

In-sample 5-fold CV Sharpe: 3.546 (seed 0), 3.527 (seed 1), 3.530 (seed 2).

    .venv/bin/python submissions/sector_model.py --cv     # reproduce the CV numbers
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
from sklearn.cluster import SpectralClustering
from sklearn.linear_model import Ridge

DATA = Path(__file__).resolve().parent.parent / "data"

HALF = 49                       # trade at the close of this bar
HORIZON = 10                    # hold until the close of bar 59
DRIFT_LEN = 10                  # a headline drifts bars b .. b+DRIFT_LEN
N_SECTORS = 5
SUFFIX = {"Group", "Co"}        # third word that is still part of a company name
LAM = 200.0                     # template ridge strength (scaled by median vol^2 below)
HALF_LIVES = (1, 2, 3, 5, 10, 25, 1000)
ALPHA = 50.0                    # calibration ridge strength


# ---------------------------------------------------------------- headline parsing / timing
def drift_weight(k):
    """Weight of the k-th bar after a headline (k = 0 is the headline's own bar)."""
    return 0.5 if k in (0, DRIFT_LEN) else 1.0


def remaining_drift(b):
    """Drift bars of a headline at bar b that fall in the holding window (HALF, HALF + HORIZON]."""
    b = np.asarray(b)
    return sum(drift_weight(k) * ((b + k > HALF) & (b + k <= HALF + HORIZON)) for k in range(DRIFT_LEN + 1))


def parse_headlines(hl: pd.DataFrame) -> pd.DataFrame:
    hl = hl.copy()
    words = hl.headline.str.split()
    hl["company"] = [" ".join(w[:3]) if w[2] in SUFFIX else " ".join(w[:2]) for w in words]
    rest = [" ".join(w[3:]) if w[2] in SUFFIX else " ".join(w[2:]) for w in words]
    hl["tmpl"] = [re.sub(r"\d+(\.\d+)?%", "N%", re.sub(r"\$[\d.]+[MBK]?", "$X", r)) for r in rest]
    return hl


def company_sectors(hl: pd.DataFrame) -> dict:
    """Cluster companies by how often they appear in the same session."""
    comp = np.sort(hl.company.unique())
    cix = {c: i for i, c in enumerate(comp)}
    M = np.zeros((len(comp), len(comp)))
    for cs in hl.groupby("session").company.apply(lambda s: sorted(set(s))):
        for a in cs:
            for b in cs:
                if a != b:
                    M[cix[a], cix[b]] += 1
    labels = SpectralClustering(N_SECTORS, affinity="precomputed", random_state=0).fit_predict(M + 1e-3)
    return dict(zip(comp, labels))


# ---------------------------------------------------------------- dataset
def load_split(kind: str, sectors: dict) -> dict:
    """Prices (log), main-sector headlines and Garman-Klass vol for one split ('ins' or 'oos')."""
    bars = pd.read_parquet(DATA / f"{kind}_bars.parquet")
    hl = parse_headlines(pd.read_parquet(DATA / f"{kind}_headlines.parquet"))
    S = np.sort(bars.session.unique())
    P = {c: np.log(bars.pivot(index="session", columns="bar_ix", values=c).reindex(S).to_numpy())
         for c in ("open", "high", "low", "close")}

    hl["grp"] = hl.company.map(sectors)
    if hl.grp.isna().any():
        raise ValueError("headline mentions a company not seen in-sample")
    h49 = hl[hl.bar_ix <= HALF]                                    # sector choice uses bars <= 49 only
    dist = h49.groupby(["session", "grp"]).company.nunique().unstack(fill_value=0).reindex(S, fill_value=0).to_numpy()
    cnt = h49.groupby(["session", "grp"]).size().unstack(fill_value=0).reindex(S, fill_value=0).to_numpy()
    main = np.argmax(dist * 1000 + cnt, 1)
    hl["i"] = hl.session.map({s: i for i, s in enumerate(S)})
    hl = hl[hl.grp.to_numpy() == main[hl.i.to_numpy()]].copy()

    O, H, L, C = (P[c][:, :HALF + 1] for c in ("open", "high", "low", "close"))
    gk = np.sqrt(np.clip((0.5 * (H - L) ** 2 - (2 * np.log(2) - 1) * (C - O) ** 2).mean(1), 1e-10, None))
    return dict(S=S, C=P["close"], hl=hl, gk=gk)


def load_returns(S) -> np.ndarray:
    return pd.read_parquet(DATA / "ins_returns.parquet").set_index("session").returns.reindex(S).to_numpy()


# ---------------------------------------------------------------- model
class SectorDriftModel:
    def __init__(self, ins: dict, oos: dict):
        self.tm = np.sort(pd.concat([ins["hl"].tmpl, oos["hl"].tmpl]).unique())
        self.tix = {t: k for k, t in enumerate(self.tm)}
        self.lam = LAM / np.median(ins["gk"]) ** 2

    # template drift --------------------------------------------------------
    def _design(self, d, idx, maxbar):
        """Rows = (session, bar return 1..maxbar); vol-normalised returns and drift exposures."""
        C = d["C"][idx][:, :maxbar + 1]
        R = np.diff(C, axis=1)
        R = R - R.mean(0)                                          # market-adjust per bar
        nb = R.shape[1]
        h = d["hl"][d["hl"].i.isin(idx)]
        ri = h.i.map({i: r for r, i in enumerate(idx)}).to_numpy()
        b = h.bar_ix.to_numpy()
        ci = h.tmpl.map(self.tix).to_numpy()
        rows, cols, vals = [], [], []
        for k in range(DRIFT_LEN + 1):
            j = b + k - 1                                          # bar b+k's return is column b+k-1
            ok = (j >= 0) & (j < nb)
            rows.append(ri[ok] * nb + j[ok]); cols.append(ci[ok]); vals.append(np.full(ok.sum(), drift_weight(k)))
        X = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(len(idx) * nb, len(self.tm)))
        X = sp.diags(1 / np.repeat(d["gk"][idx], nb)) @ X
        R = R / d["gk"][idx][:, None]
        return X, R.ravel()

    def fit_templates(self, parts):
        """parts: list of (dataset, row indices, last bar usable)."""
        Xs, Rs = zip(*[self._design(d, idx, maxbar) for d, idx, maxbar in parts])
        self.coef = lsqr(sp.vstack(Xs).tocsr(), np.concatenate(Rs), damp=np.sqrt(self.lam))[0]
        return self

    # features ----------------------------------------------------------------
    def news_signal(self, d, idx):
        h = d["hl"][d["hl"].i.isin(idx) & (d["hl"].bar_ix <= HALF)]
        e = h.tmpl.map(dict(zip(self.tm, self.coef))).fillna(0) * remaining_drift(h.bar_ix.to_numpy())
        return e.groupby(h.i).sum().reindex(idx).fillna(0).to_numpy()

    def residuals(self, d, idx):
        """Bar returns 1..49 (vol units) minus the drift explained by headlines at bars <= 49."""
        cm = dict(zip(self.tm, self.coef))
        h = d["hl"][d["hl"].i.isin(idx) & (d["hl"].bar_ix <= HALF)]
        c = h.tmpl.map(cm).fillna(0).to_numpy()
        b = h.bar_ix.to_numpy()
        i = h.i.map({v: k for k, v in enumerate(idx)}).to_numpy()
        R = np.diff(d["C"][idx][:, :HALF + 1], axis=1)
        R = (R - R.mean(0)) / d["gk"][idx][:, None]
        E = np.zeros_like(R)
        for k in range(DRIFT_LEN + 1):
            j = b + k - 1
            ok = (j >= 0) & (j < HALF)
            np.add.at(E, (i[ok], j[ok]), drift_weight(k) * c[ok])
        return R - E

    def features(self, d, idx):
        Z = self.residuals(d, idx)
        age = np.arange(HALF)[::-1]                                # bars before bar 49
        return np.column_stack([self.news_signal(d, idx)] + [(Z * np.exp(-age / h)).sum(1) for h in HALF_LIVES])

    # calibration + sizing ----------------------------------------------------
    def fit_calibration(self, ins, tr_idx, y_tr):
        Xa = self.features(ins, tr_idx)
        ya = y_tr / ins["gk"][tr_idx]
        self.mu, self.sd = Xa.mean(0), Xa.std(0) + 1e-12
        self.ridge = Ridge(alpha=ALPHA).fit((Xa - self.mu) / self.sd, ya)
        self.var_y = ya.var()
        return self

    def predict(self, d, idx):
        m = self.ridge.predict((self.features(d, idx) - self.mu) / self.sd) - self.ridge.intercept_
        return m / (d["gk"][idx] * (self.var_y + m ** 2))          # E[r] / E[r^2]


def train_predict(ins, oos, y, tr_idx, te, te_idx, label_free_parts):
    """Fit on labelled in-sample rows tr_idx, predict rows te_idx of dataset te."""
    model = SectorDriftModel(ins, oos)
    model.fit_templates([(ins, tr_idx, 99)] + label_free_parts)
    model.fit_calibration(ins, tr_idx, y[tr_idx])
    return model.predict(te, te_idx)


def load_all():
    raw_ins = parse_headlines(pd.read_parquet(DATA / "ins_headlines.parquet"))
    sectors = company_sectors(raw_ins)
    ins, oos = load_split("ins", sectors), load_split("oos", sectors)
    return ins, oos, load_returns(ins["S"])


def sharpe(pred, ret):
    pnl = pred * ret
    return float(pnl.mean() / pnl.std() * np.sqrt(252))


def cross_validate(seed=0, n_folds=5):
    from sklearn.model_selection import KFold
    ins, oos, y = load_all()
    N, allo = len(ins["S"]), np.arange(len(oos["S"]))
    folds = list(KFold(n_folds, shuffle=True, random_state=seed).split(np.arange(N)))
    p = np.zeros(N)
    for a, b in folds:   # held-out fold's bars 0-49 and OOS bars 0-49 are label-free inputs
        p[b] = train_predict(ins, oos, y, a, ins, b, [(ins, b, HALF), (oos, allo, HALF)])
    return sharpe(p, y), [round(sharpe(p[b], y[b]), 2) for _, b in folds]


if __name__ == "__main__":
    import sys
    if "--cv" in sys.argv:
        for seed in (0, 1, 2):
            s, per_fold = cross_validate(seed)
            print(f"seed {seed}: 5-fold Sharpe {s:.3f}  folds {per_fold}", flush=True)
    else:
        print(__doc__)
