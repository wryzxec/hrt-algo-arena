import sys, json; sys.path.insert(0,"src")
from arena.data import load_sessions, partitions
from arena.evaluate import score
from arena.models.text import TextModel
X,y = load_sessions(); tr,va,_ = partitions(X.index)
out=json.load(open("reports/text/trials.json"))
for a in (3000,10000,30000):
    m=TextModel(a,"recent").fit(X.loc[tr],y.loc[tr]); s=score(y.loc[va], m.predict(X.loc[va]))["sharpe"]
    out.append(dict(mode="recent",alpha=a,sharpe=round(s,3),note="beyond issue grid")); print(out[-1], flush=True)
json.dump(out, open("reports/text/trials.json","w"), indent=1)
