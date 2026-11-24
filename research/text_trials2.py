import sys, json; sys.path.insert(0,"src"); sys.path.insert(0,"research")
import numpy as np, pandas as pd
from arena.data import load_sessions, partitions
from arena.evaluate import score
from arena.models.text import TextModel, remaining_drift
X,y = load_sessions(); tr,va,_ = partitions(X.index)
out=json.load(open("reports/text/trials.json"))
for a in (300,1000):
    m=TextModel(a,"recent").fit(X.loc[tr],y.loc[tr]); s=score(y.loc[va], m.predict(X.loc[va]))["sharpe"]
    out.append(dict(mode="recent",alpha=a,sharpe=round(s,3),note="beyond issue grid")); print(out[-1], flush=True)
json.dump(out, open("reports/text/trials.json","w"), indent=1)
# reference: template-mean x drift, estimated from train labels + first-half only (no future bars)
from common import SUFFIX
def tm(h):
    t=h.split(); rest=" ".join(t[3:]) if t[2] in SUFFIX else " ".join(t[2:])
    import re; rest=re.sub(r"\$[\d.]+[MBK]?","$X",rest); return re.sub(r"\d+(\.\d+)?%","N%",rest)
rows=[(s,tm(h),w) for s,hs,bs in zip(X.index,X.headlines,X.headline_bars) for h,w in zip(hs,remaining_drift(bs)) if w>0]
d=pd.DataFrame(rows,columns=["s","t","w"]); D=d.pivot_table(index="s",columns="t",values="w",aggfunc="sum").reindex(X.index).fillna(0)
from sklearn.linear_model import Ridge
for a in (1e-3,1e-2,1e-1,1):
    r=Ridge(alpha=a).fit(D.loc[tr],y.loc[tr]); print("template one-hot x drift ridge alpha",a, round(score(y.loc[va], r.predict(D.loc[va])-r.intercept_)["sharpe"],3))
