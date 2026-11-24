import sys; sys.path.insert(0,"src"); sys.path.insert(0,"research")
import numpy as np, pandas as pd, scipy.sparse as sp, time
from arena.data import load_sessions
from text_pr8 import TextModel, _norm, remaining_drift, gk_vol
from text_hybrid import TextModel as Hybrid
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
X, y = load_sessions(); S = X.index; yv = y.to_numpy(); g = gk_vol(X)
hlf = pd.read_parquet("data/ins_headlines.parquet"); bars = pd.read_parquet("data/ins_bars.parquet")
Cf = np.log(bars.pivot(index="session",columns="bar_ix",values="close").reindex(S).to_numpy())
folds = list(KFold(5, shuffle=True, random_state=0).split(S))
sh = lambda p, idx=slice(None): (p*yv)[idx].mean()/(p*yv)[idx].std()*np.sqrt(252)
def aug_fit_predict(tr_ids, te_X, alpha):
    h = hlf[hlf.session.isin(set(tr_ids))].copy(); h["txt"] = h.headline.map(_norm)
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=5, sublinear_tf=True).fit(h[h.bar_ix<=49].txt)
    Hm = vec.transform(h.txt); ri = h.session.map({s:i for i,s in enumerate(tr_ids)}).to_numpy(); b = h.bar_ix.to_numpy()
    blocks, ys = [], []; tri = S.get_indexer(tr_ids)
    for t in range(49, 90):
        w = remaining_drift(b-(t-49))*(b<=t)
        blocks.append(sp.csr_matrix((w,(ri,np.arange(len(ri)))), shape=(len(tr_ids),len(ri))) @ Hm)
        if t == 49: ys.append(yv[tri])
        else: fr = Cf[:,t+10]-Cf[:,t]; ys.append((fr-fr.mean())[tri])     # cross-session mean over ALL ins sessions' bars >49 (label-free prices)
    m = Ridge(alpha=alpha).fit(sp.vstack(blocks).tocsr(), np.concatenate(ys))
    tm = TextModel(); tm.vec = vec; tm.mode = "recent"; tm.vol_scale = True; tm.ridge = m
    return tm.predict(te_X)
res = {k: np.zeros(len(S)) for k in ["text_a3000","hybrid","aug_a10000","aug_a30000"]}
t0=time.time()
for k,(a,b) in enumerate(folds):
    res["text_a3000"][b] = TextModel(3000).fit(X.iloc[a], y.iloc[a]).predict(X.iloc[b]); res["hybrid"][b] = Hybrid().fit(X.iloc[a], y.iloc[a]).predict(X.iloc[b])
    for al in (10000,30000): res[f"aug_a{al}"][b] = aug_fit_predict(S[a], X.iloc[b], al)
    print("fold",k,round(time.time()-t0),"s",flush=True)
basic = pd.read_pickle("research/basic_ins.pkl").to_numpy()
res["basic_sign"] = basic
for k,v in res.items(): print(f"{k:14s} 5fold={sh(v):.3f} folds={[round(sh(v,b),2) for _,b in folds]}")
z=lambda p:p/p.std(); print("blend text_a3000+aug_a30000", round(sh(z(res['text_a3000'])+z(res['aug_a30000'])),3))
np.savez("research/cv_candidates.npz", **res)
