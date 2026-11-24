import sys; sys.path.insert(0,"research")
exec(open("research/sector_cv2.py").read().split("N = len(ins")[0])
import numpy as np
from sklearn.model_selection import KFold
N=len(ins["S"]); folds=list(KFold(5,shuffle=True,random_state=0).split(np.arange(N)))
sh=lambda p,i=slice(None):(p*y)[i].mean()/(p*y)[i].std()*np.sqrt(252)
# 1. impulse profile, main-sector headlines only (signed by template mean drift), in vol units
C=ins["C"]; R=np.diff(C,axis=1); R=(R-R.mean(0))/ins["gk"][:,None]
h=ins["hl"]; h=h[(h.bar_ix>=1)&(h.bar_ix<=80)]
i=h.i.to_numpy(); b=h.bar_ix.to_numpy()
f=np.array([R[i,b-1+k] for k in range(10)]).sum(0); s=np.sign(pd.Series(f).groupby(h.tmpl.values).transform("mean").to_numpy())
prof=[(s*R[i,b-1+k]).mean() for k in range(0,16)]
print("impulse (vol units) k=0..15:", np.round(np.array(prof)/np.mean(prof[1:10]),2))
# 2. residual momentum + calibration, using the best signal from sector_cv2
o=np.load("research/sector_cv2.npz"); sig=o["volnorm_extra49_lam200"]
Cc=C[:,:50]; gk=ins["gk"]
# expected drift realised in bars 40..49 from main headlines: need coef per fold -> approximate with OOF-free proxy: skip; use raw momentum
mom={k:(Cc[:,49]-Cc[:,49-k])/gk for k in (1,3,5,10)}
for a in (-0.5,-1,-1.5,-2): print("sizing exponent", a, round(sh(sig*gk*gk**a),3))
z=lambda p:p/p.std()
for k,m in mom.items():
    for wgt in (0.1,0.2,0.3):
        print(f"sig + {wgt}*mom{k}/vol", round(sh(z(sig)+wgt*z(m)),3), end=" | ")
    print()
# calibrated ridge in CV
F=np.column_stack([sig]+[m for m in mom.values()]+[1/gk])
p=np.zeros(N)
for a,b in folds:
    from sklearn.linear_model import LinearRegression
    r=LinearRegression().fit(F[a]*gk[a,None], y[a]/gk[a]); p[b]=r.predict(F[b]*gk[b,None])/gk[b]
print("calibrated linear (sig, mom1..10) in vol units:", round(sh(p),3), [round(sh(p,b),2) for _,b in folds])
