import sys; sys.path.insert(0,"research")
exec(open("research/sector_cv2.py").read().split("N = len(ins")[0])
import numpy as np
N=len(ins["S"]); allo=np.arange(len(oos["S"])); lam=200/(np.median(ins["gk"])**2)
coef=fit([(ins,np.arange(N),99),(oos,allo,49)],True,lam)          # diagnostic: full-sample coefs
cm=dict(zip(tm,coef)); h=ins["hl"]; c=h.tmpl.map(cm).fillna(0).to_numpy(); b=h.bar_ix.to_numpy(); i=h.i.to_numpy()
gk=ins["gk"]; C=ins["C"]; R=np.diff(C,axis=1); R=(R-R.mean(0))/gk[:,None]
E=np.zeros_like(R)
for k in range(11):
    j=b+k-1; ok=(j>=0)&(j<99); np.add.at(E,(i[ok],j[ok]),wk(k)*c[ok])
Z=R-E                                                        # residual in vol units
print("explained var share of bar returns: %.3f"%(1-Z.var()/R.var()))
print("residual std (vol units) bars 0-48 vs 49-98: %.3f %.3f"%(Z[:,:49].std(),Z[:,49:].std()))
ac=lambda X,l: np.corrcoef(X[:,:-l].ravel(),X[:,l:].ravel())[0,1]
print("residual autocorr lags1-5:",[round(ac(Z,l),4) for l in range(1,6)])
# does past residual path predict future 10-bar residual sum?
from sklearn.linear_model import RidgeCV; from sklearn.model_selection import cross_val_predict
Xr=Z[:,:49]; yr=Z[:,49:59].sum(1)
p=cross_val_predict(RidgeCV(alphas=np.logspace(-2,4,13)),Xr,yr,cv=5); print("CV corr past-residual->future residual: %.4f"%np.corrcoef(p,yr)[0,1])
# heteroskedasticity: future 10-bar residual var vs GK, vs residual realized vol, vs #recent headlines
fv=Z[:,49:59].std(1)
rv=Z[:,:49].std(1)
nrec=np.bincount(i[(b>=40)&(b<=49)],minlength=N)
print("future resid vol (vol units) corr with: past resid vol %.3f, #recent hl %.3f"%(np.corrcoef(fv,rv)[0,1],np.corrcoef(fv,nrec)[0,1]))
# is per-session vol constant? compare GK first half vs second half of bars 0-49 against sampling noise
H,L,O,Cc=None,None,None,None
print("vol ratio future/GK percentiles:",np.percentile(Z[:,49:59].std(1),[10,50,90]).round(2))
