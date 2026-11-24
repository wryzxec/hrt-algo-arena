import sys; sys.path.insert(0,"research")
from features import *
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

hl, close, ret = headline_table("ins"); _, hl_full, _, _ = load("ins")
Cf = close.to_numpy(); C = Cf[:, :50]; S = close.index; y = ret.to_numpy(); N=len(S)
sidx = {s:i for i,s in enumerate(S)}
lr = np.diff(np.log(C),axis=1); vol = lr[:,-20:].std(1)
mom = np.column_stack([np.log(C[:,49]/C[:,49-k]) for k in (1,3,5,10,20)])
Rf = np.diff(np.log(Cf),axis=1); Rf = Rf - Rf.mean(0)      # market-adjusted bar returns, Rf[:,j] = bar j+1
folds = list(KFold(5, shuffle=True, random_state=0).split(S))
tmpls = np.sort(hl_full.tmpl.unique()); tix = {t:k for k,t in enumerate(tmpls)}
def sh(p): pnl=p*y; return pnl.mean()/pnl.std()*np.sqrt(252)
def fold_sh(p): return [round(((p*y)[te].mean()/(p*y)[te].std())*np.sqrt(252),2) for _,te in folds]
def report(name, p): print(f"{name:55s} sharpe={sh(p):5.2f}  corr={np.corrcoef(p,y)[0,1]:.3f}  folds={fold_sh(p)}", flush=True)

def agg(te, eff_by_tmpl):
    h = hl[hl.session.isin(S[te])]
    t = h.tmpl.map(eff_by_tmpl).fillna(0)
    return (t*h.w).groupby(h.session).sum().reindex(S[te]).fillna(0).to_numpy()

# per-headline 10-bar drift (market-adjusted log), for univariate estimators
def per_headline(tr):
    e = template_effects(hl_full, close, S[tr]); return e
def shrink_mean(e, col="f", k0=30):
    vm = e.groupby("verb")[col].mean(); g = e.groupby("tmpl")[col]
    tv = e.groupby("tmpl").verb.first().map(vm)
    return (g.mean()*g.size()+tv*k0)/(g.size()+k0)

# joint estimation: every bar's return = sum of drift of all active headlines + noise
def joint(tr, lam=50.0, L=10):
    h = hl_full[hl_full.session.isin(S[tr])]
    rowmap = {s:r for r,s in enumerate(S[tr])}
    rows, cols, vals = [], [], []
    ri = h.session.map(rowmap).to_numpy(); b = h.bar_ix.to_numpy(); ci = h.tmpl.map(tix).to_numpy()
    for k in range(L+1):
        wk = 0.5 if k in (0,L) else 1.0; j = b+k-1; ok=(j>=0)&(j<99)
        rows.append(ri[ok]*99+j[ok]); cols.append(ci[ok]); vals.append(np.full(ok.sum(), wk))
    X = sp.csr_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))), shape=(len(tr)*99, len(tmpls)))
    yy = Rf[tr].ravel()
    coef = lsqr(X, yy, damp=np.sqrt(lam))[0]
    return pd.Series(coef, index=tmpls)

oof = {k: np.zeros(N) for k in ["base","clip","median","joint","ridge7","logit_sign","hgb_sign","clip_joint"]}
for fi,(tr,te) in enumerate(folds):
    e = per_headline(tr)
    oof["base"][te] = agg(te, shrink_mean(e))
    q = e.f.quantile([0.01,0.99]).to_numpy(); e["fc"] = e.f.clip(*q)
    oof["clip"][te] = agg(te, shrink_mean(e,"fc"))
    oof["median"][te] = agg(te, e.groupby("tmpl").f.median())
    jt = joint(tr); oof["joint"][te] = agg(te, jt)
    # train-side signals for the stacked models (use in-fold effects; same as before)
    def sig(idx, eff):
        h=hl[hl.session.isin(S[idx])]; t=h.tmpl.map(eff).fillna(0)
        return (t*h.w).groupby(h.session).sum().reindex(S[idx]).fillna(0).to_numpy()
    eff = shrink_mean(e)
    Xtr = np.column_stack([sig(tr,eff), mom[tr], vol[tr]]); Xte = np.column_stack([oof["base"][te], mom[te], vol[te]])
    oof["ridge7"][te] = Ridge(alpha=1e-6).fit(Xtr, y[tr]).predict(Xte)
    lg = LogisticRegression(C=1e3, max_iter=2000).fit(Xtr/Xtr.std(0), y[tr]>0)
    oof["logit_sign"][te] = 2*lg.predict_proba(Xte/Xtr.std(0))[:,1]-1
    hg = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=200).fit(Xtr, y[tr]>0)
    oof["hgb_sign"][te] = 2*hg.predict_proba(Xte)[:,1]-1
    print("fold",fi,"done", flush=True)
np.savez("research/oof_kaggle.npz", **oof)
print()
report("5  baseline: shrunk template mean x drift left", oof["base"])
report("A  up/down target, logistic -> 2p-1", oof["logit_sign"])
report("A' up/down target, boosted trees -> 2p-1", oof["hgb_sign"])
report("B  clipped (1-99%) drifts before averaging", oof["clip"])
report("B' median drift per template", oof["median"])
report("C  joint fit of all headline types (sparse ridge)", oof["joint"])
report("7  linear: news + momentum + vol", oof["ridge7"])
z = lambda p: p/p.std()
report("D  blend 5 + 7", z(oof["base"])+z(oof["ridge7"]))
report("D' blend C + 7", z(oof["joint"])+z(oof["ridge7"]))
report("D'' blend 5 + C + 7 + A", z(oof["base"])+z(oof["joint"])+z(oof["ridge7"])+z(oof["logit_sign"]))
for a in (1,2,4): report(f"   squash: tanh({a} * z(5))  (sign-like sizing)", np.tanh(a*z(oof["base"])))
