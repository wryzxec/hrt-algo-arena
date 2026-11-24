import sys; sys.path.insert(0,"research")
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.sparse.linalg import lsqr
from features import load
from sklearn.model_selection import KFold
bars = pd.read_parquet("data/ins_bars.parquet"); _, hl, close, ret = load("ins")
S = close.index; y = ret.to_numpy(); Cf = np.log(close.to_numpy())
R = np.diff(Cf, axis=1); R = R - R.mean(0)
grp = pd.read_pickle("research/company_group.pkl"); hl["grp"] = hl.company.map(grp)
# main sector from headlines in bars 0..49 only (what OOS has)
h49 = hl[hl.bar_ix <= 49]
dist = h49.groupby(["session","grp"]).company.nunique().unstack(fill_value=0).reindex(S, fill_value=0)
top = dist.to_numpy(); srt = np.sort(top, 1)
print("sessions with tied main sector:", (srt[:,-1]==srt[:,-2]).mean().round(3))
cnt = h49.groupby(["session","grp"]).size().unstack(fill_value=0).reindex(S, fill_value=0).to_numpy()
main = np.argmax(top*1000 + cnt, 1)               # tie-break by headline count
hl["main"] = hl.grp.to_numpy() == main[hl.session.map({s:i for i,s in enumerate(S)}).to_numpy()]
P = {c: np.log(bars[bars.bar_ix<=49].pivot(index="session",columns="bar_ix",values=c).reindex(S).to_numpy()) for c in ("open","high","low","close")}
gk = np.sqrt(np.clip((0.5*(P["high"]-P["low"])**2-(2*np.log(2)-1)*(P["close"]-P["open"])**2).mean(1),1e-10,None))
tm = np.sort(hl.tmpl.unique()); tix = {t:k for k,t in enumerate(tm)}
sidx = {s:i for i,s in enumerate(S)}
def wk(k): return 0.5 if k in (0,10) else 1.0
def joint(tr, h, lam=50.0):
    h = h[h.session.isin(S[tr])]; rowmap = {s:r for r,s in enumerate(S[tr])}
    ri = h.session.map(rowmap).to_numpy(); b = h.bar_ix.to_numpy(); ci = h.tmpl.map(tix).to_numpy(); rows, cols, vals = [], [], []
    for k in range(11):
        j = b+k-1; ok = (j>=0)&(j<99); rows.append(ri[ok]*99+j[ok]); cols.append(ci[ok]); vals.append(np.full(ok.sum(), wk(k)))
    X = sp.csr_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))), shape=(len(tr)*99, len(tm)))
    return pd.Series(lsqr(X, R[tr].ravel(), damp=np.sqrt(lam))[0], index=tm)
def remaining(b): return sum(wk(k)*((b+k>49)&(b+k<=59)) for k in range(11))
def signal(te, coef, h):
    h = h[h.session.isin(S[te]) & (h.bar_ix<=49)]
    e = h.tmpl.map(coef).fillna(0)*remaining(h.bar_ix.to_numpy())
    return e.groupby(h.session).sum().reindex(S[te]).fillna(0).to_numpy()
folds = list(KFold(5, shuffle=True, random_state=0).split(S))
sh = lambda p, i=slice(None): (p*y)[i].mean()/(p*y)[i].std()*np.sqrt(252)
res = {k: np.zeros(len(S)) for k in ["all","main_fit_all_pred","main"]}
hm = hl[hl.main]
for a, b in folds:
    ca = joint(a, hl); cm = joint(a, hm)
    res["all"][b] = signal(b, ca, hl)/gk[b]
    res["main_fit_all_pred"][b] = signal(b, ca, hm)/gk[b]
    res["main"][b] = signal(b, cm, hm)/gk[b]
for k,v in res.items(): print(f"{k:20s} 5fold={sh(v):.3f} folds={[round(sh(v,b),2) for _,b in folds]}", flush=True)
np.savez("research/sector_cv.npz", **res, main_idx=main, gk=gk)
