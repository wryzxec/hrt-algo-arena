"""Shared tensors for deep models. Windows of 50 bars ending at bar t, target = t->t+10 log return
minus the cross-session mean for that t. t=49 is the real task; t in 50..89 are augmentation windows
(only ever built for TRAIN sessions -- they use bars > 49 of those sessions as extra training samples)."""
import sys; sys.path.insert(0,"research"); sys.path.insert(0,"src")
import numpy as np, pandas as pd
from common import load
from arena.data import partitions

def build():
    bars = pd.read_parquet("data/ins_bars.parquet"); _, hl, close, ret = load("ins")
    S = close.index
    piv = {c: np.log(bars.pivot(index="session", columns="bar_ix", values=c).reindex(S).to_numpy()) for c in ("open","high","low","close")}
    tmpls = np.sort(hl.tmpl.unique()); tix = {t:k+1 for k,t in enumerate(tmpls)}   # 0 = padding
    # headline template ids per (session, bar): up to 4 per bar
    MAXH = 4
    T = np.zeros((len(S), 100, MAXH), dtype=np.int64)
    sidx = {s:i for i,s in enumerate(S)}; slot = {}
    for s, b, t in zip(hl.session.map(sidx), hl.bar_ix, hl.tmpl.map(tix)):
        k = slot.get((s,b), 0)
        if k < MAXH: T[s,b,k] = t; slot[(s,b)] = k+1
    tr, va, au = partitions(S)
    return dict(S=S, O=piv["open"], H=piv["high"], L=piv["low"], C=piv["close"], T=T, ntmpl=len(tmpls)+1,
                y=ret.to_numpy(), tr=S.get_indexer(tr), va=S.get_indexer(va), au=S.get_indexer(au))

def windows(d, idx, ts):
    """price channels (n,50,4), template ids (n,50,MAXH), target (n,), vol (n,)"""
    Xp, Xt, Y, V = [], [], [], []
    C, O, H, L, T = d["C"], d["O"], d["H"], d["L"], d["T"]
    for t in ts:
        sl = slice(t-49, t+1)
        c = C[idx, sl]; prev = np.concatenate([O[idx, t-49:t-48], C[idx, t-49:t]], 1)
        r = c - prev
        ch = np.stack([r, H[idx, sl]-L[idx, sl], c-O[idx, sl], c - c[:, -1:]], -1)
        Xp.append(ch); Xt.append(T[idx, sl])
        if t == 49: y = d["y"][idx]
        else:
            fr = C[:, t+10] - C[:, t]; y = (fr - fr.mean())[idx]          # market-adjust across ALL sessions at that t
        Y.append(y)
        hh, ll, oo = H[idx, sl], L[idx, sl], O[idx, sl]
        V.append(np.sqrt(np.clip((0.5*(hh-ll)**2 - (2*np.log(2)-1)*(c-oo)**2).mean(1), 1e-10, None)))
    return (np.concatenate(Xp).astype(np.float32), np.concatenate(Xt), np.concatenate(Y).astype(np.float32), np.concatenate(V).astype(np.float32))
