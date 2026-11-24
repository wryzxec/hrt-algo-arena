import sys; sys.path.insert(0,"research")
import numpy as np, pandas as pd
from features import load
bars=pd.read_parquet("data/ins_bars.parquet"); bars=bars[bars.bar_ix<=49]
H=bars.pivot(index="session",columns="bar_ix",values="high").to_numpy(); L=bars.pivot(index="session",columns="bar_ix",values="low").to_numpy()
O=bars.pivot(index="session",columns="bar_ix",values="open").to_numpy(); Cc=bars.pivot(index="session",columns="bar_ix",values="close").to_numpy()
_,_,close,ret=load("ins"); y=ret.to_numpy(); o=np.load("research/oof_kaggle.npz")
from sklearn.model_selection import KFold
folds=list(KFold(5,shuffle=True,random_state=0).split(y))
sh=lambda p,idx=slice(None): (p*y)[idx].mean()/(p*y)[idx].std()*np.sqrt(252)
park=np.sqrt((np.log(H/L)**2).mean(1)/(4*np.log(2)))
gk=np.sqrt((0.5*np.log(H/L)**2-(2*np.log(2)-1)*np.log(Cc/O)**2).mean(1))
cc=np.diff(np.log(Cc),axis=1).std(1)
# future realized vol (bars 50-59), check which estimator predicts it best
F=np.log(close.to_numpy()); fv=np.diff(F[:,49:60],axis=1).std(1)
for nm,v in [("close-close",cc),("parkinson",park),("garman-klass",gk)]:
    print(f"{nm:13s} corr w/ future vol={np.corrcoef(v,fv)[0,1]:.3f}  ", "joint:", " ".join(f"a={a}:{sh(o['joint']*v**a):.3f}" for a in (0,-0.5,-1,-1.5)), "| fold diffs a=-1:", [round(sh(o['joint']*v**-1,te)-sh(o['joint'],te),2) for _,te in folds])
print("future vol / park ratio spread:", np.percentile(fv/park,[10,50,90]).round(2))
