import sys; sys.path.insert(0,"research")
from features import *
_, hl, close, ret = load("ins")
C=close.to_numpy(); S=close.index; sidx={s:i for i,s in enumerate(S)}
R=np.diff(np.log(C),axis=1); R=R-R.mean(0)
eff=template_effects(hl,close); hl2=hl[(hl.bar_ix>=1)&(hl.bar_ix<=89)].copy(); hl2["f"]=eff.f.values
hl2["tm"]=hl2.groupby("tmpl").f.transform("mean")
i=hl2.session.map(sidx).to_numpy(); b=hl2.bar_ix.to_numpy()
# session vol measured on bars before the headline (pre) excluding news periods is messy; use whole-first-half vol
vol=R[:,:49].std(1); hl2["vol"]=vol[i]
# 1. does drift scale with stock vol?  regress f on tm and tm*vol
q=pd.qcut(hl2.vol,5,labels=False)
print("drift / template-mean by vol quintile (1.0 = no scaling):")
for k in range(5):
    m=(q==k)&(hl2.tm.abs()>hl2.tm.abs().median()); print(" q",k, "vol=%.4f"%hl2.vol[m].mean(), "slope=%.2f"%(np.sum(hl2.f[m]*hl2.tm[m])/np.sum(hl2.tm[m]**2)))
# 2. company-specific effect? residual drift by company, sign-adjusted
hl2["res"]=(hl2.f-hl2.tm)*np.sign(hl2.tm)
g=hl2.groupby("company").res.agg(["mean","count","sem"]); print("company residual t-stats: max |t| =", (g["mean"]/g["sem"]).abs().max().round(2), " n companies", len(g))
# company multiplier: slope per company
sl=hl2.groupby("company").apply(lambda d: np.sum(d.f*d.tm)/np.sum(d.tm**2)); print("per-company slope spread: ", sl.describe().round(2).to_dict())
# split-half reliability of per-company slope
h1=hl2[hl2.session%2==0]; h2=hl2[hl2.session%2==1]
s1=h1.groupby("company").apply(lambda d: np.sum(d.f*d.tm)/np.sum(d.tm**2)); s2=h2.groupby("company").apply(lambda d: np.sum(d.f*d.tm)/np.sum(d.tm**2))
print("split-half corr of per-company slope:", np.corrcoef(s1,s2.reindex(s1.index))[0,1].round(3))
# 3. per-session slope reliability (is it a session-level 'news sensitivity'?)
hl2["half"]=hl2.bar_ix<45
a=hl2[hl2.half].groupby("session").apply(lambda d: np.sum(d.f*d.tm)/np.sum(d.tm**2)+0); 
c=hl2[~hl2.half].groupby("session").apply(lambda d: np.sum(d.f*d.tm)/np.sum(d.tm**2)+0)
j=a.index.intersection(c.index); print("per-session sensitivity corr early vs late headlines:", np.corrcoef(a[j].clip(-20,20),c[j].clip(-20,20))[0,1].round(3))
# 4. same-company repeated headlines: does the 2nd headline of same company have smaller/larger effect?
hl2=hl2.sort_values(["session","bar_ix"]); hl2["nth"]=hl2.groupby(["session","company"]).cumcount()
print("slope by nth mention of the company in session:", hl2.groupby(hl2.nth.clip(upper=4)).apply(lambda d: np.sum(d.f*d.tm)/np.sum(d.tm**2)).round(2).to_dict())
# 5. stacking: do drifts overlap additively? slope when another headline active within 10 bars
cnt=hl2.groupby("session").bar_ix.apply(np.array)
# 6. drift length per verb (does it vary?)
for v in ["secures","misses","announces","faces"]:
    m=hl[(hl.verb==v)&(hl.bar_ix<=60)]; ii=m.session.map(sidx).to_numpy(); bb=m.bar_ix.to_numpy(); sgn=np.sign(hl2[hl2.verb==v].f.mean())
    prof=[ (R[ii,bb+k-1]).mean()*1e4*sgn for k in range(0,16)]
    print(v, np.round(prof,1))
