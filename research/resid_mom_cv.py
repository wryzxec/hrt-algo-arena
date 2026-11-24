import sys; sys.path.insert(0,"research")
exec(open("research/sector_cv2.py").read().split("N = len(ins")[0])
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge
N=len(ins["S"]); allo=np.arange(len(oos["S"])); lam=200/(np.median(ins["gk"])**2)
folds=list(KFold(5,shuffle=True,random_state=0).split(np.arange(N)))
sh=lambda p,i=slice(None):(p*y)[i].mean()/(p*y)[i].std()*np.sqrt(252)
gk=ins["gk"]; C=ins["C"]
def resid49(d, idx, coef):
    """residual bar returns (vol units) for bars 1..49 using only headlines <=49 and coefs"""
    cm=dict(zip(tm,coef)); h=d["hl"][d["hl"].i.isin(idx)&(d["hl"].bar_ix<=49)]; rm={v:k for k,v in enumerate(idx)}
    c=h.tmpl.map(cm).fillna(0).to_numpy(); b=h.bar_ix.to_numpy(); i=h.i.map(rm).to_numpy()
    R=np.diff(d["C"][idx][:,:50],axis=1); R=(R-R.mean(0))/d["gk"][idx][:,None]   # note: cross-session demean within the given set
    E=np.zeros_like(R)
    for k in range(11):
        j=b+k-1; ok=(j>=0)&(j<49); np.add.at(E,(i[ok],j[ok]),wk(k)*c[ok])
    return R-E
HL=(3,10,25,1000)
def ew(Z):
    t=np.arange(49)[::-1]; return np.column_stack([ (Z*np.exp(-t/h)).sum(1) for h in HL])
sig=np.zeros(N); F=np.zeros((N,len(HL))); Zall=np.zeros((N,49))
for a,b in folds:
    coef=fit([(ins,a,99),(ins,b,49),(oos,allo,49)],True,lam)
    sig[b]=signal(ins,b,coef)/gk[b]
    Za=resid49(ins,a,coef); Zb=resid49(ins,b,coef)
    F[a] = F[a]  # train-side features recomputed per fold below
    Zall[b]=Zb
    # store per-fold train features for nested fit
    globals().setdefault("TR",{})[tuple(b[:3])]=(ew(Za),coef)
F=ew(Zall)
print("univariate 5fold Sharpe of residual momentum features (vol-sized):")
for k,hh in enumerate(HL): print("  halflife",hh, round(sh(F[:,k]/gk),3), " corr w/ y/gk:", round(np.corrcoef(F[:,k], y/gk)[0,1],4))
print("news signal alone:", round(sh(sig),3))
P={}
P["sig+ewall (nested lin)"]=np.zeros(N); P["sig+ew25"]=np.zeros(N); P["sig+ew1000"]=np.zeros(N)
for a,b in folds:
    Fa,_=TR[tuple(b[:3])]
    sa=sig[a]*0  # need train-side news signal: compute in-sample signal for train rows with train coef
    _,coef=TR[tuple(b[:3])]; sa=signal(ins,a,coef)/gk[a]
    for name,cols in [("sig+ewall (nested lin)",list(range(len(HL)))),("sig+ew25",[2]),("sig+ew1000",[3])]:
        Xa=np.column_stack([sa*gk[a], Fa[:,cols]]); Xb=np.column_stack([sig[b]*gk[b], F[b][:,cols]])
        r=LinearRegression().fit(Xa, y[a]/gk[a]); P[name][b]=(r.predict(Xb)-r.intercept_)/gk[b]
for k,v in P.items(): print(f"{k:24s} 5fold={sh(v):.3f} folds={[round(sh(v,b),2) for _,b in folds]}")
np.savez("research/resid_mom.npz", sig=sig, F=F, **{k.replace(' ','_').replace('+','_').replace('(','').replace(')',''):v for k,v in P.items()})
