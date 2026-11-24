import sys; sys.path.insert(0,"research")
from common import *
bars, hl0, close, ret = load()
hl = pd.read_pickle("research/hl_ins.pkl")
C = close.to_numpy(); sidx = {s:i for i,s in enumerate(close.index)}
i = hl.session.map(sidx).to_numpy(); b = hl.bar_ix.to_numpy()
hl["f20"] = C[i,np.minimum(b+20,99)]/C[i,np.maximum(b-1,0)]-1
# template sentiment via split-half to avoid leak: use verb mean f20 sign
vs = hl.groupby("verb").f20.mean(); hl["sent"] = np.sign(hl.verb.map(vs)) 
hl.loc[hl.verb.isin(["schedules","sees","board","Chief","CTO","CEO","CFO"]),"sent"]=0
print("signed f20 by rank (bps):"); print((hl.sent*hl.f20).groupby(hl["rank"].clip(upper=4)).mean()*1e4)
print("by n mentions:"); print((hl.sent*hl.f20).groupby(hl["n"].clip(upper=9)).mean()*1e4)
# impulse curve: mean signed return of bar b+k (k=0..40) relative to headline
R = np.diff(C,axis=1)/C[:,:-1]   # R[:,j] = return of bar j+1
R = R - R.mean(axis=0)          # demean per bar across sessions
m = hl[(hl.sent!=0)&(hl.bar_ix<50)]
ii = m.session.map(sidx).to_numpy(); bb=m.bar_ix.to_numpy(); ss=m.sent.to_numpy()
prof=[]
for k in range(0,45):
    j = bb+k-1; ok=(j>=0)&(j<99)
    prof.append((ss[ok]*R[ii[ok],j[ok]]).mean()*1e4)
print("impulse per bar (bps) k=0..44:", np.round(prof,2))
print("cumulative:", np.round(np.cumsum(prof),1))
