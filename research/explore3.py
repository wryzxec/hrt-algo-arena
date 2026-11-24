import sys; sys.path.insert(0,"research")
from common import *
bars, hl, close, ret = load()
C = close.to_numpy(); sidx = {s:i for i,s in enumerate(close.index)}
# check market-adjustment: ret vs raw C[59]/C[49]-1
raw = C[:,59]/C[:,49]-1
print("corr raw vs ret", np.corrcoef(raw, ret)[0,1], "mean raw", raw.mean(), "diff std", (raw-raw.mean()-ret).std())
# companies per session, and rank
cnt = hl.groupby(["session","company"]).size().rename("n").reset_index()
cnt["rank"] = cnt.groupby("session").n.rank(ascending=False, method="first")
print(cnt.groupby("session").size().describe())
print(cnt[cnt["rank"]==1].n.describe(), cnt[cnt["rank"]==2].n.describe())
hl = hl.merge(cnt, on=["session","company"])
# bar return at headline bar (close[b]/close[b-1])
i = hl.session.map(sidx).to_numpy(); b = hl.bar_ix.to_numpy()
ok = b>=1
r0 = np.full(len(hl), np.nan); r0[ok] = C[i[ok], b[ok]]/C[i[ok], b[ok]-1]-1
hl["r0"]=r0
# baseline abs bar return
allr = np.abs(np.diff(C,axis=1)/C[:,:-1]).mean()
print("baseline abs bar ret", allr)
print(hl.groupby(hl["rank"].clip(upper=4)).r0.apply(lambda x: x.abs().mean()))
hl.to_pickle("research/hl_ins.pkl")
