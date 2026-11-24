import sys; sys.path.insert(0,"research")
src = open("research/sector_cv2.py").read().split("N = len(ins")[0]
# parametrise main-sector rule: 'h49' (deployable) or 'full' (oracle: uses all 100 bars of headlines)
src = src.replace('def prep(kind):', 'def prep(kind, rule="h49"):').replace(
 'h49 = hl[hl.bar_ix<=49]', 'h49 = hl[hl.bar_ix<=49] if rule=="h49" else hl')
exec(src)
import numpy as np
from sklearn.model_selection import KFold
ins_o = prep("ins", "full")
N=len(ins["S"]); allo=np.arange(len(oos["S"])); folds=list(KFold(5,shuffle=True,random_state=0).split(np.arange(N)))
sh=lambda p,i=slice(None):(p*y)[i].mean()/(p*y)[i].std()*np.sqrt(252)
# how often does <=49 rule disagree with full-session rule?
def mainvec(d):
    h=d["hl"]; return h.groupby("i").grp.first().reindex(range(N))
m49, mfull = mainvec(ins), mainvec(ins_o)
print("main-sector disagreement (<=49 vs full session):", (m49!=mfull).mean().round(3))
lam=200/(np.median(ins["gk"])**2)
def cv(dtrain, dpred, sector_specific=False):
    p=np.zeros(N)
    for a,b in folds:
        if sector_specific:
            out=np.zeros(len(b))
            global tm, tix
            base_tm=tm
            hl_s = dtrain["hl"].copy(); hl_s["tmpl"]=hl_s.tmpl+"|"+hl_s.grp.astype(str); dt=dict(dtrain, hl=hl_s)
            hp = dpred["hl"].copy(); hp["tmpl"]=hp.tmpl+"|"+hp.grp.astype(str); dp=dict(dpred, hl=hp)
            ho = oos["hl"].copy(); ho["tmpl"]=ho.tmpl+"|"+ho.grp.astype(str); do=dict(oos, hl=ho)
            tm=np.sort(pd.concat([hl_s.tmpl, ho.tmpl]).unique()); tix={t:k for k,t in enumerate(tm)}
            coef=fit([(dt,a,99),(dp,b,49),(do,allo,49)],True,lam); p[b]=signal(dp,b,coef)/dpred["gk"][b]
            tm=base_tm; tix={t:k for k,t in enumerate(tm)}
        else:
            coef=fit([(dtrain,a,99),(dpred,b,49),(oos,allo,49)],True,lam); p[b]=signal(dpred,b,coef)/dpred["gk"][b]
    return p
r={}
r["deployable (<=49 sector)"]=cv(ins, ins)
r["ORACLE sector at predict"]=cv(ins_o, ins_o)
r["oracle train, <=49 predict"]=cv(ins_o, ins)
r["sector-specific coefs"]=cv(ins, ins, sector_specific=True)
for k,v in r.items(): print(f"{k:30s} 5fold={sh(v):.3f} folds={[round(sh(v,b),2) for _,b in folds]}", flush=True)
