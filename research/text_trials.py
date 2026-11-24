import sys, time, json; sys.path.insert(0,"src")
import numpy as np
from arena.data import load_sessions, partitions
from arena.evaluate import score
from arena.models.text import TextModel
X,y = load_sessions(); tr,va,_ = partitions(X.index)
out=[]
for mode in ("recent","all"):
    for a in (1,10,100):
        t=time.time(); m=TextModel(a,mode).fit(X.loc[tr],y.loc[tr]); s=score(y.loc[va], m.predict(X.loc[va]))["sharpe"]
        out.append(dict(mode=mode,alpha=a,sharpe=round(s,3),sec=round(time.time()-t,1))); print(out[-1], flush=True)
json.dump(out, open("reports/text/trials.json","w"), indent=1)
