import sys; sys.path.insert(0,"research")
from features import *
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
hl, close, ret = headline_table("ins")
_,hl_full,_,_ = load("ins")
C = close.to_numpy()[:, :50]; S = close.index; y = ret.to_numpy()
def sharpe(p, y): pnl=p*y; return pnl.mean()/pnl.std()*np.sqrt(252)
def stats(name, p):
    pnl=p*y; print(f"{name:45s} sharpe={sharpe(p,y):7.2f} corr={np.corrcoef(p,y)[0,1]:.3f} nonzero={(p!=0).mean():.2f}")
# price features
lr = np.diff(np.log(C),axis=1)
vol = lr[:, -20:].std(1)
mom = {k: np.log(C[:,49]/C[:,49-k]) for k in (1,3,5,10,20,49)}
results = {}
# 1. notebook keyword baseline
kw = hl.assign(s=hl.headline.str.contains("secures").astype(float)-hl.headline.str.contains("decline").astype(float)).groupby("session").s.sum().reindex(S).fillna(0).to_numpy()
stats("1 notebook keyword (secures/decline)", kw)
# CV for learned template effects
kf = KFold(5, shuffle=True, random_state=0)
preds = {k: np.zeros(len(S)) for k in ["verb_all","verb_w","tmpl_w","tmpl_w_vol","tmpl_w_shrunk","ridge"]}
for tr, te in kf.split(S):
    eff = template_effects(hl_full, close, S[tr])
    vmean = eff.groupby("verb").f.mean()
    g = eff.groupby("tmpl").f; tm, tn = g.mean(), g.size()
    # shrink template toward verb mean
    tv = eff.groupby("tmpl").verb.first().map(vmean)
    k0 = 30; tshr = (tm*tn + tv*k0)/(tn+k0)
    h = hl[hl.session.isin(S[te])].copy()
    h["v"] = h.verb.map(vmean).fillna(0); h["t"] = h.tmpl.map(tm).fillna(h.v); h["ts"] = h.tmpl.map(tshr).fillna(h.v)
    agg = lambda col, w: (h[col]*w).groupby(h.session).sum().reindex(S[te]).fillna(0).to_numpy()
    preds["verb_all"][te] = agg("v", np.sign(h.v)*0+1)
    preds["verb_w"][te] = agg("v", h.w)
    preds["tmpl_w"][te] = agg("t", h.w)
    preds["tmpl_w_shrunk"][te] = agg("ts", h.w)
    preds["tmpl_w_vol"][te] = preds["tmpl_w_shrunk"][te] / vol[te]**2
# ridge on news signal + price features, nested
X = np.column_stack([preds["tmpl_w_shrunk"]] + [mom[k] for k in mom] + [vol])
for tr, te in kf.split(S):
    m = Ridge(alpha=1e-6).fit(X[tr], y[tr]); preds["ridge"][te] = m.predict(X[te])
stats("2 verb sentiment, all headlines equal", preds["verb_all"])
stats("3 verb sentiment x remaining-drift weight", preds["verb_w"])
stats("4 template effect x remaining drift", preds["tmpl_w"])
stats("5 shrunk template x remaining drift", preds["tmpl_w_shrunk"])
stats("6 #5 / vol^2 (risk-scaled)", preds["tmpl_w_vol"])
stats("7 ridge: #5 + momentum/vol features", preds["ridge"])
for k in mom: stats(f"  momentum {k} bars", mom[k]); stats(f"  reversal {k} bars", -mom[k])
np.save("research/oof_tmpl.npy", preds["tmpl_w_shrunk"])
