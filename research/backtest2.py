import sys; sys.path.insert(0,"research")
from features import *
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
hl, close, ret = headline_table("ins")
_,hl_full,_,_ = load("ins")
C = close.to_numpy()[:, :50]; S = close.index; y = ret.to_numpy()
def stats(name, p):
    pnl=p*y; print(f"{name:50s} sharpe={pnl.mean()/pnl.std()*np.sqrt(252):6.2f} corr={np.corrcoef(p,y)[0,1]:.3f}")
lr = np.diff(np.log(C),axis=1); vol = lr[:,-20:].std(1)
# magnitude check: within template, does |amount| or pct scale f?
eff_all = template_effects(hl_full, close)
h_full = hl_full[(hl_full.bar_ix>=1)&(hl_full.bar_ix<=89)].copy()
h_full["f"]=eff_all.f.values
amt = h_full.headline.str.extract(r"\$([\d.]+)([MB])"); h_full["amt"]=amt[0].astype(float)*np.where(amt[1]=="B",1000,1)
h_full["pct"]=h_full.headline.str.extract(r"([\d.]+)%")[0].astype(float)
h_full["fd"]=h_full.f - h_full.groupby("tmpl").f.transform("mean")
for c in ["amt","pct"]:
    m=h_full[c].notna(); print(c, "n",m.sum(),"corr(within-template resid, log val)", np.corrcoef(h_full.fd[m], np.log(h_full[c][m]))[0,1].round(3),
       "corr signed", np.corrcoef(h_full.fd[m]*np.sign(h_full.groupby('tmpl').f.transform('mean')[m]), np.log(h_full[c][m]))[0,1].round(3))
kf = KFold(5, shuffle=True, random_state=0)
F = {}
oof = {k: np.zeros(len(S)) for k in ["base","ridge","gbm"]}
def build(h, sess, tm_map, vmean):
    h = h.copy(); h["v"]=h.verb.map(vmean).fillna(0); h["t"]=h.tmpl.map(tm_map).fillna(h.v)
    out = pd.DataFrame(index=sess)
    out["news"] = (h.t*h.w).groupby(h.session).sum()
    for lo,hi in [(40,44),(45,49),(30,39),(0,29)]:
        mm=(h.bar_ix>=lo)&(h.bar_ix<=hi); out[f"s{lo}"]=h.t[mm].groupby(h.session[mm]).sum()
    # elapsed-drift: expected move already realized since headlines in 40-49
    el = np.clip(49-h.bar_ix+0.5,0,10); out["exp_done"]=(h.t*el*(h.bar_ix>=39)).groupby(h.session).sum()
    out["nhl"]=h.groupby("session").size()
    return out.reindex(sess).fillna(0)
for tr, te in kf.split(S):
    eff = template_effects(hl_full, close, S[tr]); vmean=eff.groupby("verb").f.mean()
    g=eff.groupby("tmpl").f; tv=eff.groupby("tmpl").verb.first().map(vmean); tm=(g.mean()*g.size()+tv*30)/(g.size()+30)
    Xtr = build(hl[hl.session.isin(S[tr])], S[tr], tm, vmean)  # note: in-fold template effects -> slight leak on train, fine for relative
    Xte = build(hl[hl.session.isin(S[te])], S[te], tm, vmean)
    i_tr, i_te = tr, te
    def addp(X, idx):
        X=X.copy(); X["vol"]=vol[idx]
        for k in (1,3,5,10,20): X[f"m{k}"]=np.log(C[idx,49]/C[idx,49-k])
        X["real_vs_exp"]=X["m10"]-X["exp_done"]  # surprise realized vs expected drift
        return X
    Xtr, Xte = addp(Xtr,i_tr), addp(Xte,i_te)
    oof["base"][te]=Xte.news
    r = Ridge(alpha=1e-4).fit(Xtr, y[tr]); oof["ridge"][te]=r.predict(Xte)
    gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=1.0).fit(Xtr,y[tr]); oof["gbm"][te]=gb.predict(Xte)
    F["cols"]=list(Xtr.columns); F["coef"]=r.coef_
stats("5 shrunk template x remaining drift", oof["base"])
stats("8 ridge: news + age buckets + price", oof["ridge"])
stats("9 GBM: same features", oof["gbm"])
stats("10 blend 0.5 ridge + 0.5 gbm", 0.5*oof["ridge"]/oof["ridge"].std()+0.5*oof["gbm"]/oof["gbm"].std())
print(dict(zip(F["cols"], np.round(F["coef"]*1e3,4))))
