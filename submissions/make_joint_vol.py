"""Chosen single model (5-fold in-sample CV Sharpe 3.06): joint template drift x remaining drift / GK vol.

1. Every bar's market-adjusted log return (all 10,000 in-sample sessions, bars 1-99) is regressed on the
   drift of every headline active on that bar: one coefficient per headline template (302), sparse ridge.
   A headline drifts on bars b..b+10 (half weight at both ends).
2. Prediction = sum over headlines of coef[template] x drift bars left after bar 49 (only bars <= 49 used).
3. Divide by Garman-Klass vol from bars 0-49 OHLC (Sharpe-optimal sizing; drift scales with vol).
Self-contained: reads only data/*.parquet. Run: .venv/bin/python submissions/make_joint_vol.py
"""
import re, hashlib
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.sparse.linalg import lsqr

D = Path(__file__).resolve().parent.parent / "data"
L, LAM, SUFFIX = 10, 50.0, {"Group", "Co"}

def tmpl(h):
    t = h.split(); rest = " ".join(t[3:]) if t[2] in SUFFIX else " ".join(t[2:])
    return re.sub(r"\d+(\.\d+)?%", "N%", re.sub(r"\$[\d.]+[MBK]?", "$X", rest))

def wk(k): return 0.5 if k in (0, L) else 1.0

def remaining(b):
    return sum(wk(k) * ((b + k > 49) & (b + k <= 59)) for k in range(L + 1))

ib = pd.read_parquet(D / "ins_bars.parquet"); ih = pd.read_parquet(D / "ins_headlines.parquet")
S = np.sort(ib.session.unique()); C = np.log(ib.pivot(index="session", columns="bar_ix", values="close").reindex(S).to_numpy())
R = np.diff(C, axis=1); R -= R.mean(0)
ih["t"] = ih.headline.map(tmpl); T = np.sort(ih.t.unique()); tix = {t: k for k, t in enumerate(T)}
ri = ih.session.map({s: i for i, s in enumerate(S)}).to_numpy(); b = ih.bar_ix.to_numpy(); ci = ih.t.map(tix).to_numpy()
rows, cols, vals = [], [], []
for k in range(L + 1):
    j = b + k - 1; ok = (j >= 0) & (j < 99)
    rows.append(ri[ok] * 99 + j[ok]); cols.append(ci[ok]); vals.append(np.full(ok.sum(), wk(k)))
Xd = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(S) * 99, len(T)))
coef = pd.Series(lsqr(Xd, R.ravel(), damp=np.sqrt(LAM))[0], index=T)

def predict(kind):
    bars = pd.read_parquet(D / f"{kind}_bars.parquet"); bars = bars[bars.bar_ix <= 49]
    hl = pd.read_parquet(D / f"{kind}_headlines.parquet"); hl = hl[hl.bar_ix <= 49]
    ss = np.sort(bars.session.unique())
    P = {c: np.log(bars.pivot(index="session", columns="bar_ix", values=c).reindex(ss).to_numpy()) for c in ("open", "high", "low", "close")}
    gk = np.sqrt(np.clip((0.5 * (P["high"] - P["low"]) ** 2 - (2 * np.log(2) - 1) * (P["close"] - P["open"]) ** 2).mean(1), 1e-10, None))
    eff = hl.headline.map(tmpl).map(coef).fillna(0.0) * remaining(hl.bar_ix.to_numpy())
    sig = eff.groupby(hl.session).sum().reindex(ss).fillna(0.0).to_numpy()
    return pd.Series(sig / gk, index=ss), hl

p_oos, hl_oos = predict("oos")
unseen = (~hl_oos.headline.map(tmpl).isin(coef.index)).sum()
out = Path(__file__).resolve().parent / "joint_vol.csv"
pd.DataFrame({"headline_id": p_oos.index.astype(np.int64), "pred": p_oos.to_numpy()}).to_csv(out, index=False, lineterminator="\n")
# validation
sub = pd.read_csv(out); exp = set(pd.read_parquet(D / "oos_bars.parquet").session.unique())
assert list(sub.columns) == ["headline_id", "pred"] and len(sub) == 10000 and sub.headline_id.is_unique
assert set(sub.headline_id) == exp and np.isfinite(sub.pred).all()
print(f"wrote {out.name}: rows={len(sub)} nonzero={(sub.pred != 0).mean():.2f} unseen OOS templates={unseen} "
      f"sha256={hashlib.sha256(out.read_bytes()).hexdigest()[:16]}")
