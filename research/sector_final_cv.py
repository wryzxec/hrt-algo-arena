import sys; sys.path.insert(0,"research")
exec(open("research/sector_cv2.py").read().split("N = len(ins")[0])
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
N=len(ins["S"]); allo=np.arange(len(oos["S"])); folds=list(KFold(5,shuffle=True,random_state=0).split(np.arange(N)))
sh=lambda p,i=slice(None):(p*y)[i].mean()/(p*y)[i].std()*np.sqrt(252)
def realized_weight(b, lo, hi):   # drift bars of a headline at b that fall in (lo, hi]
    return sum(wk(k)*((b+k>lo)&(b+k<=hi)) for k in range(11))
def feats(d, idx, coef):
    cm = dict(zip(tm, coef)); h = d["hl"][d["hl"].i.isin(idx) & (d["hl"].bar_ix<=49)].copy(); c = h.tmpl.map(cm).fillna(0).to_numpy(); b = h.bar_ix.to_numpy()
    g = lambda w: pd.Series(c*w).groupby(h.i.to_numpy()).sum().reindex(idx).fillna(0).to_numpy()
    sig = g(remaining(b)); gk = d["gk"][idx]; C = d["C"][idx]
    out = {"sig": sig}
    for lo in (39, 44, 46):
        exp = g(realized_weight(b, lo, 49)); real = (C[:,49]-C[:,lo])/gk
        out[f"surp{lo}"] = real - exp
    # instance-level: surprise weighted by how much of recent headlines' drift remains
    rec = (remaining(b) > 0).astype(float); out["recent_n"] = pd.Series(rec).groupby(h.i.to_numpy()).sum().reindex(idx).fillna(0).to_numpy()
    out["surp44_x_rec"] = out["surp44"] * (out["recent_n"] > 0)
    return pd.DataFrame(out)
lam = 200/(np.median(ins["gk"])**2)
P = {k: np.zeros(N) for k in ["sig","lin_sig_surp","lin_all"]}
for a, b in folds:
    coef = fit([(ins, a, 99), (ins, b, 49), (oos, allo, 49)], True, lam)
    Fa, Fb = feats(ins, a, coef), feats(ins, b, coef)
    ga, gb = ins["gk"][a], ins["gk"][b]
    P["sig"][b] = Fb.sig/gb
    for name, cols in [("lin_sig_surp", ["sig","surp44"]), ("lin_all", list(Fa.columns))]:
        r = LinearRegression().fit(Fa[cols], y[a]/ga)                   # target in vol units
        P[name][b] = (r.predict(Fb[cols]) - r.intercept_)/gb             # E[r]/vol^2 ~ (pred*vol)/vol^2
    print("fold", dict(zip(cols, np.round(r.coef_/r.coef_[0],3))), flush=True)
for k,v in P.items(): print(f"{k:14s} 5fold={sh(v):.3f} folds={[round(sh(v,b),2) for _,b in folds]}")
