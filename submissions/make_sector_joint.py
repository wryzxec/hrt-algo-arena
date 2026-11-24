"""Sector-filtered joint template drift / GK vol.  5-fold in-sample CV Sharpe 3.38 (vs 3.06 for joint_vol.csv).

1. Companies form 5 sectors of 16 (spectral clustering of in-sample company co-occurrence). A session's stock
   belongs to the sector with the most distinct companies in its bar 0-49 headlines (ties -> more headlines).
   Only that sector's headlines move the price (off-sector drift slope = 0.00), so others are dropped.
2. Each headline gives drift on bars b..b+10 with weights 0.5,1,...,1,0.5 (measured exactly), proportional to vol.
   One coefficient per template, fitted jointly by sparse ridge on vol-normalised, market-adjusted bar returns
   from all in-sample bars 1-99 plus OOS bars 1-49 (label-free inputs).
3. pred = sum coef x drift-bars left after bar 49, / GK vol (Sharpe-optimal E[r]/Var[r]).
Run: .venv/bin/python submissions/make_sector_joint.py
"""
import re, hashlib
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.sparse.linalg import lsqr
from sklearn.cluster import SpectralClustering

D = Path(__file__).resolve().parent.parent / "data"; SUFFIX = {"Group", "Co"}; LAM = 200.0

def wk(k): return 0.5 if k in (0, 10) else 1.0
def remaining(b): return sum(wk(k) * ((b + k > 49) & (b + k <= 59)) for k in range(11))

def parse(hl):
    t = hl.headline.str.split()
    hl["company"] = [" ".join(x[:3]) if x[2] in SUFFIX else " ".join(x[:2]) for x in t]
    rest = [" ".join(x[3:]) if x[2] in SUFFIX else " ".join(x[2:]) for x in t]
    hl["tmpl"] = [re.sub(r"\d+(\.\d+)?%", "N%", re.sub(r"\$[\d.]+[MBK]?", "$X", r)) for r in rest]
    return hl

raw = {k: (pd.read_parquet(D / f"{k}_bars.parquet"), parse(pd.read_parquet(D / f"{k}_headlines.parquet"))) for k in ("ins", "oos")}
# sectors from in-sample co-occurrence
hl_i = raw["ins"][1]; comp = np.sort(hl_i.company.unique()); cix = {c: i for i, c in enumerate(comp)}
M = np.zeros((len(comp), len(comp)))
for cs in hl_i.groupby("session").company.apply(lambda s: sorted(set(s))):
    for a in cs:
        for b in cs:
            if a != b: M[cix[a], cix[b]] += 1
grp = dict(zip(comp, SpectralClustering(5, affinity="precomputed", random_state=0).fit_predict(M + 1e-3)))

def prep(kind):
    bars, hl = raw[kind]; S = np.sort(bars.session.unique())
    P = {c: np.log(bars.pivot(index="session", columns="bar_ix", values=c).reindex(S).to_numpy()) for c in ("open", "high", "low", "close")}
    hl = hl.copy(); hl["grp"] = hl.company.map(grp)
    assert hl.grp.notna().all(), "company not seen in-sample"
    h49 = hl[hl.bar_ix <= 49]
    dist = h49.groupby(["session", "grp"]).company.nunique().unstack(fill_value=0).reindex(index=S, columns=range(5), fill_value=0).to_numpy()
    cnt = h49.groupby(["session", "grp"]).size().unstack(fill_value=0).reindex(index=S, columns=range(5), fill_value=0).to_numpy()
    main = np.argmax(dist * 1000 + cnt, 1); hl["i"] = hl.session.map({s: i for i, s in enumerate(S)})
    hl = hl[hl.grp.to_numpy() == main[hl.i.to_numpy()]]
    O, H, L, C = (P[c][:, :50] for c in ("open", "high", "low", "close"))
    gk = np.sqrt(np.clip((0.5 * (H - L) ** 2 - (2 * np.log(2) - 1) * (C - O) ** 2).mean(1), 1e-10, None))
    return dict(S=S, C=P["close"], hl=hl, gk=gk)

ins, oos = prep("ins"), prep("oos")
tm = np.sort(pd.concat([ins["hl"].tmpl, oos["hl"].tmpl]).unique()); tix = {t: k for k, t in enumerate(tm)}

def design(d, maxbar):
    C = d["C"][:, :maxbar + 1]; R = np.diff(C, axis=1); R = (R - R.mean(0)) / d["gk"][:, None]; nb = R.shape[1]
    h = d["hl"]; ri = h.i.to_numpy(); b = h.bar_ix.to_numpy(); ci = h.tmpl.map(tix).to_numpy(); rows, cols, vals = [], [], []
    for k in range(11):
        j = b + k - 1; ok = (j >= 0) & (j < nb); rows.append(ri[ok] * nb + j[ok]); cols.append(ci[ok]); vals.append(np.full(ok.sum(), wk(k)))
    X = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(C) * nb, len(tm)))
    return sp.diags(1 / np.repeat(d["gk"], nb)) @ X, R.ravel()

(Xa, Ra), (Xb, Rb) = design(ins, 99), design(oos, 49)
coef = pd.Series(lsqr(sp.vstack([Xa, Xb]).tocsr(), np.concatenate([Ra, Rb]), damp=np.sqrt(LAM / np.median(ins["gk"]) ** 2))[0], index=tm)

h = oos["hl"][oos["hl"].bar_ix <= 49]
sig = (h.tmpl.map(coef) * remaining(h.bar_ix.to_numpy())).groupby(h.i).sum().reindex(range(len(oos["S"]))).fillna(0).to_numpy()
pred = sig / oos["gk"]
out = Path(__file__).resolve().parent / "sector_joint.csv"
pd.DataFrame({"headline_id": oos["S"].astype(np.int64), "pred": pred}).to_csv(out, index=False, lineterminator="\n")
sub = pd.read_csv(out); exp = set(raw["oos"][0].session.unique())
assert list(sub.columns) == ["headline_id", "pred"] and len(sub) == 10000 and sub.headline_id.is_unique
assert set(sub.headline_id) == exp and np.isfinite(sub.pred).all()
prev = pd.read_csv(Path(__file__).resolve().parent / "joint_vol.csv").set_index("headline_id").pred
print(f"wrote {out.name}: rows={len(sub)} nonzero={(sub.pred != 0).mean():.2f} corr with joint_vol={sub.set_index('headline_id').pred.corr(prev):.3f} "
      f"sha256={hashlib.sha256(out.read_bytes()).hexdigest()[:16]}")
