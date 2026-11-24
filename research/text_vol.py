import sys; sys.path.insert(0,"src")
import numpy as np
from arena.data import load_sessions, partitions
from arena.evaluate import score
from arena.models.text import TextModel
X,y = load_sessions(); tr,va,_ = partitions(X.index)
def vol(X, n=49): 
    c=np.log(np.vstack(X.close.to_numpy())); return np.diff(c,axis=1)[:,-n:].std(1)
v=vol(X); vs=v/np.median(v); vtr=vs[X.index.get_indexer(tr)]; vva=vs[X.index.get_indexer(va)]
# A: multiply prediction by vol^a (fit unchanged)
m=TextModel(3000).fit(X.loc[tr],y.loc[tr]); p=m.predict(X.loc[va])
for a in (-1,-0.5,0,0.5,1,1.5): print("pred*vol^%s"%a, round(score(y.loc[va], p*vva**a)["sharpe"],3))
# B: fit on y/vol (so model learns sentiment in vol units), predict * vol^a
m2=TextModel(3000).fit(X.loc[tr],y.loc[tr]/vtr); p2=m2.predict(X.loc[va])
for a in (0,0.5,1): print("fit y/vol, pred*vol^%s"%a, round(score(y.loc[va], p2*vva**a)["sharpe"],3))
for al in (1000,10000):
    p3=TextModel(al).fit(X.loc[tr],y.loc[tr]/vtr).predict(X.loc[va]); print("fit y/vol alpha",al, [round(score(y.loc[va], p3*vva**a)["sharpe"],3) for a in (0,0.5,1)])
for n in (10,20):
    w=vol(X,n); w=w/np.median(w); print("vol window",n, [round(score(y.loc[va], p*w[X.index.get_indexer(va)]**a)["sharpe"],3) for a in (0,0.5,1)])
