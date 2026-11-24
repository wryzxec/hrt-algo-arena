import sys; sys.path.insert(0,"research")
import numpy as np
from features import load
_,_,close,ret=load("ins"); y=ret.to_numpy()
o=np.load("research/oof_kaggle.npz")
C=np.log(close.to_numpy()[:,:50]); 
from sklearn.model_selection import KFold
folds=list(KFold(5,shuffle=True,random_state=0).split(y))
def sh(p,idx=None):
    pnl=p*y if idx is None else (p*y)[idx]; return pnl.mean()/pnl.std()*np.sqrt(252)
for n in (49,30,20):
    v=np.diff(C,axis=1)[:,-n:].std(1)
    for name in ("base","joint"):
        p=o[name]; print(f"vol{n} {name:5s}", " ".join(f"a={a}:{sh(p*v**a):.3f}" for a in (0,-0.5,-1)), "| folds a=-1 minus a=0:", [round(sh(p*v**-1,te)-sh(p,te),2) for _,te in folds])
