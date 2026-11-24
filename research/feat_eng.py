import sys; sys.path.insert(0,"research"); sys.path.insert(0,"src")
import numpy as np, pandas as pd, scipy.sparse as sp, json
from common import load
from features import remaining_weight
from arena.data import partitions
from arena.evaluate import score
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
bars = pd.read_parquet("data/ins_bars.parquet"); bars = bars[bars.bar_ix <= 49]
_, hl, close, ret = load("ins"); hl = hl[hl.bar_ix <= 49].copy(); S = close.index; y = ret.to_numpy()
P = {c: np.log(bars.pivot(index="session", columns="bar_ix", values=c).reindex(S).to_numpy()) for c in ("open","high","low","close")}
O,H,L,C = P["open"],P["high"],P["low"],P["close"]
tr, va, _ = partitions(S); itr, iva = S.get_indexer(tr), S.get_indexer(va)
gk = np.sqrt(np.clip((0.5*(H-L)**2-(2*np.log(2)-1)*(C-O)**2).mean(1),1e-10,None))
sidx = {s:i for i,s in enumerate(S)}; hl["i"] = hl.session.map(sidx); hl["w"] = remaining_weight(hl.bar_ix.to_numpy())
tm = np.sort(hl.tmpl.unique()); tix = {t:k for k,t in enumerate(tm)}; hl["t"] = hl.tmpl.map(tix)
def onehot(w): return sp.csr_matrix((w, (hl.i, hl.t)), shape=(len(S), len(tm)))
W = onehot(hl.w.to_numpy())                                   # remaining drift
Wv = sp.diags(gk/np.median(gk)) @ W                           # remaining drift x vol
A = {f"age{lo}": onehot(((hl.bar_ix>=lo)&(hl.bar_ix<=hi)).astype(float).to_numpy()) for lo,hi in [(40,44),(45,49),(30,39)]}
el = np.clip(49-hl.bar_ix.to_numpy()+0.5, 0, 10)*(hl.bar_ix.to_numpy()>=39)
E = onehot(el)                                               # drift already elapsed (for surprise)
r = np.diff(C, axis=1)
dense = {"mom": np.column_stack([C[:,49]-C[:,49-k] for k in (1,3,5,10,20,49)]),
         "vol": np.column_stack([gk, r[:,-20:].std(1), r.std(1), (H-L)[:,-10:].mean(1)]),
         "shape": np.column_stack([C[:,49]-C.max(1), C[:,49]-C.min(1), C[:,49]-O[:,49], (H-L)[:,49]]),
         "news": np.column_stack([np.bincount(hl.i, minlength=len(S)), np.bincount(hl.i, weights=(hl.bar_ix>=40), minlength=len(S))])}
def z(M, idx): mu, sd = M[itr].mean(0), M[itr].std(0)+1e-12; return (M-mu)/sd
def ev(name, X, model, size_vol=True, target_vol=False):
    X = X.tocsr() if sp.issparse(X) else X
    yt = y[itr]/gk[itr] if target_vol else y[itr]
    m = model.fit(X[itr], yt); p = m.predict(X[iva]) - (getattr(m, "intercept_", 0) if hasattr(m,"coef_") else 0)
    if target_vol: p = p*gk[iva]
    out = {"model": name, "raw": round(score(y[iva], p)["sharpe"],3), "/vol": round(score(y[iva], p/gk[iva])["sharpe"],3)}
    print(json.dumps(out), flush=True); return p
res = {}
D_all = np.column_stack([z(v, None) for v in dense.values()])
for a in (0.3, 1, 3):
    res[f"W a{a}"] = ev(f"ridge tmpl x drift (alpha {a})", W, Ridge(alpha=a))
for a in (0.3, 1, 3):
    res[f"Wv a{a}"] = ev(f"ridge tmpl x drift x vol (alpha {a})", Wv, Ridge(alpha=a))
X1 = sp.hstack([Wv, W, E, *A.values(), sp.csr_matrix(D_all*1e-1)])
for a in (1, 3, 10):
    res[f"big a{a}"] = ev(f"ridge ALL features (alpha {a})", X1, Ridge(alpha=a))
# compact features for trees: learned drift signal (train-fit) + dense
sig = Ridge(alpha=1).fit(Wv[itr], y[itr]); s_all = (Wv @ sig.coef_)
sigE = E @ sig.coef_
Xc = np.column_stack([s_all, sigE, C[:,49]-C[:,39] - sigE, D_all])
ev("HGB on signal+surprise+dense", Xc, HistGradientBoostingRegressor(max_iter=400, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=1.0, random_state=42))
ev("ridge on signal+surprise+dense", Xc, Ridge(alpha=1e-3))
from sklearn.neural_network import MLPRegressor
ev("sklearn MLP (64,32) on signal+surprise+dense", Xc/ Xc[itr].std(0), MLPRegressor((64,32), alpha=1e-2, max_iter=300, early_stopping=True, random_state=42))
print("note: signal coef fit on train rows only; HGB/MLP stack trained on in-sample signal (mild optimism)")
