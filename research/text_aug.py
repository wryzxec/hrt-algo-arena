import sys; sys.path.insert(0,"src"); sys.path.insert(0,"research")
import numpy as np, pandas as pd, scipy.sparse as sp
from arena.data import load_sessions, partitions
from arena.evaluate import score
from arena.models.text import _norm, remaining_drift, gk_vol
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
X,y = load_sessions(); tr,va,_ = partitions(X.index)
hlf = pd.read_parquet("data/ins_headlines.parquet"); bars = pd.read_parquet("data/ins_bars.parquet")
S = X.index; Cf = np.log(bars.pivot(index="session",columns="bar_ix",values="close").reindex(S).to_numpy())
trs = set(tr); h = hlf[hlf.session.isin(trs)].copy(); h["txt"] = h.headline.map(_norm)
vec = TfidfVectorizer(ngram_range=(1,2), min_df=5, sublinear_tf=True).fit(h[h.bar_ix<=49].txt)
Hm = vec.transform(h.txt); rowof = {s:i for i,s in enumerate(tr)}; ri = h.session.map(rowof).to_numpy(); b = h.bar_ix.to_numpy()
blocks, ys = [], []
for t in range(49, 90):
    w = remaining_drift(b - (t-49)) * (b <= t)
    A = sp.csr_matrix((w, (ri, np.arange(len(ri)))), shape=(len(tr), len(ri)))
    blocks.append(A @ Hm)
    if t == 49: ys.append(y.loc[tr].to_numpy())
    else: fr = Cf[:, t+10]-Cf[:, t]; ys.append((fr - fr.mean())[S.get_indexer(tr)])
Xa = sp.vstack(blocks).tocsr(); Ya = np.concatenate(ys)
# validation design at t=49
hv = hlf[hlf.session.isin(set(va)) & (hlf.bar_ix<=49)]; rv = hv.session.map({s:i for i,s in enumerate(va)}).to_numpy()
Av = sp.csr_matrix((remaining_drift(hv.bar_ix.to_numpy()), (rv, np.arange(len(rv)))), shape=(len(va), len(rv)))
Xv = Av @ vec.transform(hv.headline.map(_norm)); gv = gk_vol(X.loc[va])
for al in (3000, 10000, 30000, 100000):
    m = Ridge(alpha=al).fit(Xa, Ya); p = m.predict(Xv) - m.intercept_
    print(f"augmented x41 TF-IDF ridge alpha={al}: raw={score(y.loc[va],p)['sharpe']:.3f}  /vol={score(y.loc[va],p/gv)['sharpe']:.3f}", flush=True)
np.save("research/text_aug_pred.npy", p/gv)
