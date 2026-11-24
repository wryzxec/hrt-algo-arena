import sys; sys.path.insert(0,"research")
exec(open("research/sector_cv2.py").read().split("N = len(ins")[0])
import numpy as np
from sklearn.model_selection import KFold
N=len(ins["S"]); folds=list(KFold(5,shuffle=True,random_state=0).split(np.arange(N)))
sh=lambda p,i=slice(None):(p*y)[i].mean()/(p*y)[i].std()*np.sqrt(252)
P=np.load("research/pred_ew_ridge.npy"); gk=ins["gk"]; m=P*gk           # m = E[y/gk]
print("base (m/gk):", round(sh(P),3), [round(sh(P,b),2) for _,b in folds])
for c in (0.5,1,2,4):
    cc=c*np.var(y/gk)
    p=m/(gk*(cc+m**2)); print(f"m/(gk(c*var+m^2)) c={c}:", round(sh(p),3), [round(sh(p,b),2) for _,b in folds])
# vol forecast: blend GK with close-close vol and recent GK
C=ins["C"][:,:50]; cc_vol=np.diff(C,axis=1).std(1)
bars=pd.read_parquet("data/ins_bars.parquet"); bars=bars[bars.bar_ix<=49]
H=np.log(bars.pivot(index="session",columns="bar_ix",values="high").to_numpy()); L=np.log(bars.pivot(index="session",columns="bar_ix",values="low").to_numpy())
O=np.log(bars.pivot(index="session",columns="bar_ix",values="open").to_numpy()); Cc=np.log(bars.pivot(index="session",columns="bar_ix",values="close").to_numpy())
gk_rec=np.sqrt(np.clip((0.5*(H-L)**2-(2*np.log(2)-1)*(Cc-O)**2)[:,-20:].mean(1),1e-10,None))
park=np.sqrt(((H-L)**2).mean(1)/(4*np.log(2)))
fut=np.diff(ins["C"][:,49:60],axis=1).std(1)
for nm,v in [("gk",gk),("cc",cc_vol),("gk_rec20",gk_rec),("park",park),("geo gk*cc",np.sqrt(gk*cc_vol)),("0.7gk+0.3cc",0.7*gk+0.3*cc_vol)]:
    p=m/v**1 * (gk/v)*0+ m*gk/v**2
    print(f"vol={nm:12s} corr future={np.corrcoef(v,fut)[0,1]:.3f}  sharpe={sh(p):.3f}", [round(sh(p,b),2) for _,b in folds])
