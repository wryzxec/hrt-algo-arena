"""Second-upload candidate: writes predictions_v2.csv.

Blend (equal weight after scaling to unit std in-sample) of
  C: headline-type drifts fitted jointly -- every bar's market-adjusted return is
     regressed on the drift of every headline still active on that bar (sparse ridge),
     so overlapping headlines don't contaminate each other's estimates;
  7: linear model on the news signal (strategy.py's) + 1/3/5/10/20-bar momentum + volatility.
In-sample 5-fold CV Sharpe 3.03 vs 2.97 for strategy.py; better in all 5 folds, but small.
Run:  .venv/bin/python strategy_v2.py
"""
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
from sklearn.linear_model import Ridge

sys.path.insert(0, "data")
sys.path.insert(0, "research")
from features import headline_table, load  # noqa: E402
from strategy import fit_effects  # noqa: E402

import sessions  # noqa: E402

L, LAM = 10, 50.0


def joint_effects():
    _, hl, close, _ = load("ins")
    S = close.index
    R = np.diff(np.log(close.to_numpy()), axis=1)
    R = R - R.mean(0)
    tmpls = np.sort(hl.tmpl.unique())
    ri = hl.session.map({s: r for r, s in enumerate(S)}).to_numpy()
    ci = hl.tmpl.map({t: k for k, t in enumerate(tmpls)}).to_numpy()
    b = hl.bar_ix.to_numpy()
    rows, cols, vals = [], [], []
    for k in range(L + 1):
        j = b + k - 1
        ok = (j >= 0) & (j < 99)
        rows.append(ri[ok] * 99 + j[ok])
        cols.append(ci[ok])
        vals.append(np.full(ok.sum(), 0.5 if k in (0, L) else 1.0))
    X = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(S) * 99, len(tmpls)),
    )
    return pd.Series(lsqr(X, R.ravel(), damp=np.sqrt(LAM))[0], index=tmpls)


def news_signal(kind, eff):
    hl, close, _ = headline_table(kind)
    t = hl.tmpl.map(eff).fillna(0.0)
    return (t * hl.w).groupby(hl.session).sum().reindex(close.index).fillna(0.0), close


def price_feats(close):
    C = close.to_numpy()[:, :50]
    mom = [np.log(C[:, 49] / C[:, 49 - k]) for k in (1, 3, 5, 10, 20)]
    vol = np.diff(np.log(C), axis=1)[:, -20:].std(1)
    return np.column_stack(mom + [vol])


if __name__ == "__main__":
    tmpl, vmean = fit_effects()
    jt = joint_effects()
    y = load("ins")[3].to_numpy()

    base_ins, close_ins = news_signal("ins", tmpl)
    base_oos, close_oos = news_signal("oos", tmpl)
    joint_ins, _ = news_signal("ins", jt)
    joint_oos, _ = news_signal("oos", jt)

    Xi = np.column_stack([base_ins.to_numpy(), price_feats(close_ins)])
    Xo = np.column_stack([base_oos.to_numpy(), price_feats(close_oos)])
    lin = Ridge(alpha=1e-6).fit(Xi, y)
    lin_ins, lin_oos = lin.predict(Xi), lin.predict(Xo)

    pred = joint_oos.to_numpy() / joint_ins.std() + lin_oos / lin_ins.std()
    pred = pred * 1e-3  # scale is irrelevant to Sharpe; keep numbers small like returns
    print(f"OOS sessions={len(pred)}  corr with strategy.py signal={np.corrcoef(pred, base_oos)[0, 1]:.3f}")
    sessions.write_submission(
        prediction_df=pd.DataFrame({"session": close_oos.index, "pred": pred}),
        path="predictions_v2.csv",
    )
