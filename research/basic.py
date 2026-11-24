"""Ultra-basic: verb sign (+1/-1/0) x remaining drift / vol.  Variants with fitted verb means."""
import sys; sys.path.insert(0,"src"); sys.path.insert(0,"research")
import numpy as np, pandas as pd
from common import load
from features import remaining_weight
from arena.data import load_sessions, partitions
from arena.models.text import gk_vol
from sklearn.model_selection import KFold
X, y = load_sessions(); S = X.index; yv = y.to_numpy(); g = gk_vol(X)
_, hl, _, _ = load("ins"); hl = hl[hl.bar_ix <= 49].copy(); hl["w"] = remaining_weight(hl.bar_ix.to_numpy())
POS = {"secures","announces","completes","wins","expands","launches","signs","achieves","beats","raises"}
NEG = {"misses","recalls","loses","lowers","faces","warns","withdraws","delays"}
hl["sgn"] = hl.verb.map(lambda v: 1.0 if v in POS else (-1.0 if v in NEG else 0.0))
def agg(col): return (hl[col]*hl.w).groupby(hl.session).sum().reindex(S).fillna(0).to_numpy()
sh = lambda p, idx: (p*yv)[idx].mean()/(p*yv)[idx].std()*np.sqrt(252)
tr, va, _ = partitions(S); itr, iva = S.get_indexer(tr), S.get_indexer(va)
folds = list(KFold(5, shuffle=True, random_state=0).split(S))
def report(name, fn):
    p_split = fn(itr, iva); cv = np.zeros(len(S))
    for a, b in folds: cv[b] = fn(a, b)
    print(f"{name:52s} split={(lambda q:(q*yv[iva]).mean()/(q*yv[iva]).std()*np.sqrt(252))(p_split):.3f}  5fold={sh(cv, slice(None)):.3f}  folds={[round(sh(cv,b),2) for _,b in folds]}")
s_sign = agg("sgn")
report("A  hand-labelled verb sign x drift (0 params)", lambda a,b: s_sign[b])
report("A' same / vol", lambda a,b: s_sign[b]/g[b])
# B: verb mean effect fitted from first-half data only: regress y on per-verb drift-weighted sums
verbs = sorted(hl.verb.unique()); V = pd.crosstab(hl.session, hl.verb, values=hl.w, aggfunc="sum").reindex(index=S, columns=verbs).fillna(0).to_numpy()
from sklearn.linear_model import Ridge
def fitB(a, b, vol=True):
    m = Ridge(alpha=10.0).fit(V[a], yv[a]); p = V[b] @ m.coef_; return p/g[b] if vol else p
report("B  per-verb ridge (37 params) x drift / vol", fitB)
# C: two numbers: separate magnitude for positive and negative verbs
P_ = agg("sgn"); pos = (hl.sgn>0)*hl.w; neg = (hl.sgn<0)*hl.w
Pp = pos.groupby(hl.session).sum().reindex(S).fillna(0).to_numpy(); Pn = neg.groupby(hl.session).sum().reindex(S).fillna(0).to_numpy()
def fitC(a,b):
    M = np.column_stack([Pp, Pn]); c = np.linalg.lstsq(M[a], yv[a], rcond=None)[0]; return (M[b] @ c)/g[b]
report("C  pos/neg magnitudes (2 params) x drift / vol", fitC)
pd.Series(s_sign/g, index=S).to_pickle("research/basic_ins.pkl")
print("unlabelled verbs (weight 0):", sorted(set(verbs)-POS-NEG))
