import sys; sys.path.insert(0,"research")
from common import *
bars, hl0, close, ret = load()
hl = pd.read_pickle("research/hl_ins.pkl")
C = close.to_numpy(); sidx = {s:i for i,s in enumerate(close.index)}
i = hl.session.map(sidx).to_numpy(); b = hl.bar_ix.to_numpy()
for k in [0,2,5,10,20]:
    e = np.minimum(b+k, 99); s = np.maximum(b-1,0)
    hl[f"f{k}"] = C[i,e]/C[i,s]-1
print("n templates", hl.tmpl.nunique(), "n verbs", hl.verb.nunique())
g = hl.groupby("verb")[["f0","f2","f5","f10","f20"]].mean()*1e4
g["n"]=hl.groupby("verb").size()
pd.set_option("display.width",200); pd.set_option("display.max_rows",200)
print(g.sort_values("f10").round(1).to_string())
