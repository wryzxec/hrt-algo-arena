import sys; sys.path.insert(0,"research")
import numpy as np, pandas as pd
from features import load, template_effects
from sklearn.cluster import SpectralClustering
_, hl, close, ret = load("ins")
comp = np.sort(hl.company.unique()); cix = {c:i for i,c in enumerate(comp)}
# co-occurrence: companies appearing in same session
sc = hl.groupby("session").company.apply(lambda s: sorted(set(s)))
M = np.zeros((len(comp), len(comp)))
for cs in sc:
    for a in cs:
        for b in cs:
            if a != b: M[cix[a], cix[b]] += 1
print("co-occurrence: mean within-pair", M[M>0].mean().round(1), "fraction of zero pairs", (M==0).mean().round(2))
for k in (4,5,6):
    lab = SpectralClustering(k, affinity="precomputed", random_state=0).fit_predict(M + 1e-3)
    within = sum(M[np.ix_(lab==g, lab==g)].sum() for g in range(k)) / M.sum()
    print(f"k={k} sizes={np.bincount(lab).tolist()} share of co-occurrence within cluster={within:.2f}")
lab = SpectralClustering(5, affinity="precomputed", random_state=0).fit_predict(M + 1e-3)
grp = dict(zip(comp, lab)); hl["grp"] = hl.company.map(grp)
# how many groups per session; main group = most distinct companies
dist = hl.groupby(["session","grp"]).company.nunique().unstack(fill_value=0)
print("distinct companies per session by group (first rows):\n", dist.head())
main = dist.idxmax(1); hl["main"] = hl.grp == hl.session.map(main)
print("share of headlines in main group:", hl.main.mean().round(2))
# what companies per session: print cluster names to see sectors
for g in range(5): print(g, [c for c in comp if grp[c]==g][:8])
# drift slope in vs out of main group
eff = template_effects(hl, close); h2 = hl[(hl.bar_ix>=1)&(hl.bar_ix<=89)].copy(); h2["f"] = eff.f.values
h2["tm"] = h2.groupby("tmpl").f.transform("mean")
for flag in (True, False):
    d = h2[h2.main==flag]; print("main-sector" if flag else "other", "n=",len(d), "slope=%.2f"%(np.sum(d.f*d.tm)/np.sum(d.tm**2)))
pd.Series(grp).to_pickle("research/company_group.pkl")
