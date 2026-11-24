import sys; sys.path.insert(0,"research")
exec(open("research/sector_cv2.py").read().split("N = len(ins")[0])
import numpy as np
C=ins["C"]; R=np.diff(C,axis=1); R=(R-R.mean(0))/ins["gk"][:,None]
h=ins["hl"]; h=h[(h.bar_ix>=1)&(h.bar_ix<=80)].copy(); i=h.i.to_numpy(); b=h.bar_ix.to_numpy()
first=np.array([R[i,b-1+k] for k in range(1,5)]).mean(0); second=np.array([R[i,b-1+k] for k in range(5,10)]).mean(0)
h["d1"]=first; h["d2"]=second
h["m"]=h.groupby("tmpl").d1.transform("mean")*0.5+h.groupby("tmpl").d2.transform("mean")*0.5
r1=h.d1-h.m; r2=h.d2-h.m
print("instance residual corr (bars 1-4 vs 5-9 after headline):", np.corrcoef(r1,r2)[0,1].round(4))
# isolated headlines only (no other main headline within +-12 bars) to avoid overlap
hb=h.sort_values(["i","bar_ix"]); gap_prev=hb.groupby("i").bar_ix.diff().fillna(99); gap_next=-hb.groupby("i").bar_ix.diff(-1).fillna(-99)
iso=hb[(gap_prev>12)&(gap_next>12)]
print("isolated n=",len(iso)," corr:", np.corrcoef(iso.d1-iso.m, iso.d2-iso.m)[0,1].round(4))
# signed by template: does instance magnitude scale? regress d2 on d1 within sign
s=np.sign(iso.m); print("signed: corr(s*d1, s*d2) isolated:", np.corrcoef(s*iso.d1, s*iso.d2)[0,1].round(4))
# template coefficient distribution (drift in vol units per bar)
tmean=h.groupby("tmpl").apply(lambda d: (d.d1.mean()*4+d.d2.mean()*5)/9)
print("template drift (vol units) sorted:", np.round(np.sort(tmean.values),3)[::6])
print("n templates", len(tmean), "n headlines per template median", h.groupby("tmpl").size().median())
