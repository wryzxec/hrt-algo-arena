import sys; sys.path.insert(0,"src")
import numpy as np
from arena.data import load_sessions, partitions
from arena.evaluate import score
from arena.models.text import TextModel
X,y = load_sessions(); tr,va,_ = partitions(X.index)
def gk(X):
    O,H,L,C=(np.log(np.vstack(X[c].to_numpy())) for c in ("open","high","low","close"))
    return np.sqrt((0.5*(H-L)**2-(2*np.log(2)-1)*(C-O)**2).mean(1))
p=TextModel(3000).fit(X.loc[tr],y.loc[tr]).predict(X.loc[va]); v=gk(X.loc[va])
for a in (0,-0.5,-1): print("text * gk^%s"%a, round(score(y.loc[va],p*v**a)["sharpe"],3))
for al in (1000,10000): 
    q=TextModel(al).fit(X.loc[tr],y.loc[tr]).predict(X.loc[va]); print("alpha",al,[round(score(y.loc[va],q*v**a)["sharpe"],3) for a in (0,-1)])
