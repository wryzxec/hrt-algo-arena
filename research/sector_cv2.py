import sys; sys.path.insert(0,"research")
import numpy as np, pandas as pd, scipy.sparse as sp, re
from scipy.sparse.linalg import lsqr
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
SUFFIX={"Group","Co"}
grp = pd.read_pickle("research/company_group.pkl")
def prep(kind):
    bars = pd.read_parquet(f"data/{kind}_bars.parquet"); hl = pd.read_parquet(f"data/{kind}_headlines.parquet")
    S = np.sort(bars.session.unique()); P = {c: np.log(bars.pivot(index="session",columns="bar_ix",values=c).reindex(S).to_numpy()) for c in ("open","high","low","close")}
    t = hl.headline.str.split(); hl["company"] = [" ".join(x[:3]) if x[2] in SUFFIX else " ".join(x[:2]) for x in t]
    rest = [" ".join(x[3:]) if x[2] in SUFFIX else " ".join(x[2:]) for x in t]
    hl["tmpl"] = [re.sub(r"\d+(\.\d+)?%","N%",re.sub(r"\$[\d.]+[MBK]?","$X",r)) for r in rest]
    hl["grp"] = hl.company.map(grp); h49 = hl[hl.bar_ix<=49]
    dist = h49.groupby(["session","grp"]).company.nunique().unstack(fill_value=0).reindex(S, fill_value=0).to_numpy()
    cnt = h49.groupby(["session","grp"]).size().unstack(fill_value=0).reindex(S, fill_value=0).to_numpy()
    main = np.argmax(dist*1000+cnt, 1); sidx = {s:i for i,s in enumerate(S)}; hl["i"] = hl.session.map(sidx)
    hl = hl[hl.grp.to_numpy() == main[hl.i.to_numpy()]].copy()
    O,H,L,C = (P[c][:, :50] for c in ("open","high","low","close"))
    gk = np.sqrt(np.clip((0.5*(H-L)**2-(2*np.log(2)-1)*(C-O)**2).mean(1),1e-10,None))
    return dict(S=S, C=P["close"], hl=hl, gk=gk)
ins, oos = prep("ins"), prep("oos")
y = pd.read_parquet("data/ins_returns.parquet").set_index("session").returns.reindex(ins["S"]).to_numpy()
tm = np.sort(pd.concat([ins["hl"].tmpl, oos["hl"].tmpl]).unique()); tix = {t:k for k,t in enumerate(tm)}
def wk(k): return 0.5 if k in (0,10) else 1.0
def remaining(b): return sum(wk(k)*((b+k>49)&(b+k<=59)) for k in range(11))
def design(d, idx, maxbar, volnorm):
    """rows = (session, bar j) for bar returns j+1 <= maxbar"""
    C = d["C"][idx][:, :maxbar+1]; R = np.diff(C, axis=1); R = R - R.mean(0); nb = R.shape[1]
    h = d["hl"][d["hl"].i.isin(idx)]; rowmap = {i:r for r,i in enumerate(idx)}
    ri = h.i.map(rowmap).to_numpy(); b = h.bar_ix.to_numpy(); ci = h.tmpl.map(tix).to_numpy(); rows, cols, vals = [], [], []
    for k in range(11):
        j = b+k-1; ok = (j>=0)&(j<nb); rows.append(ri[ok]*nb+j[ok]); cols.append(ci[ok]); vals.append(np.full(ok.sum(), wk(k)))
    X = sp.csr_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))), shape=(len(idx)*nb, len(tm)))
    if volnorm:
        v = np.repeat(d["gk"][idx], nb); X = sp.diags(1/v) @ X; R = R / d["gk"][idx][:, None]
    return X, R.ravel()
def fit(parts, volnorm, lam):
    Xs, Rs = zip(*[design(*p, volnorm) for p in parts])
    return lsqr(sp.vstack(Xs).tocsr(), np.concatenate(Rs), damp=np.sqrt(lam))[0]
def signal(d, idx, coef):
    h = d["hl"][d["hl"].i.isin(idx) & (d["hl"].bar_ix<=49)]
    e = h.tmpl.map(dict(zip(tm, coef))).fillna(0)*remaining(h.bar_ix.to_numpy())
    return e.groupby(h.i).sum().reindex(idx).fillna(0).to_numpy()
N = len(ins["S"]); allo = np.arange(len(oos["S"]))
folds = list(KFold(5, shuffle=True, random_state=0).split(np.arange(N)))
sh = lambda p, i=slice(None): (p*y)[i].mean()/(p*y)[i].std()*np.sqrt(252)
cfg = {"base(main)": (False, False, 50), "volnorm": (True, False, 50), "volnorm+extra49": (True, True, 50),
       "volnorm+extra49 lam200": (True, True, 200), "volnorm+extra49 lam10": (True, True, 10)}
res = {k: np.zeros(N) for k in cfg}
for a, b in folds:
    for name, (vn, extra, lam) in cfg.items():
        lam_eff = lam/ (np.median(ins["gk"])**2) if vn else lam
        parts = [(ins, a, 99)] + ([(ins, b, 49), (oos, allo, 49)] if extra else [])
        coef = fit(parts, vn, lam_eff)
        res[name][b] = signal(ins, b, coef)/ins["gk"][b]
    print("fold done", flush=True)
for k,v in res.items(): print(f"{k:26s} 5fold={sh(v):.3f} folds={[round(sh(v,b),2) for _,b in folds]}", flush=True)
np.savez("research/sector_cv2.npz", **{k.replace(' ','_').replace('+','_'):v for k,v in res.items()})
