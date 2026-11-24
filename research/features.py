import sys; sys.path.insert(0,"research")
from common import *
NEUTRAL = {"schedules","sees","board"}
def remaining_weight(b, horizon_start=49, horizon_end=59, L=10):
    # headline at bar b drifts on bars b+1..b+L-1 (approx, plus half at b and b+L)
    w = np.zeros_like(b, dtype=float)
    for k in range(0, L+1):
        bar = b + k
        wk = 0.5 if k in (0, L) else 1.0
        w += wk * ((bar > horizon_start) & (bar <= horizon_end))
    return w
def headline_table(kind):
    bars, hl, close, ret = load(kind)
    hl = hl[hl.bar_ix <= 49].copy()
    hl["w"] = remaining_weight(hl.bar_ix.to_numpy())
    # magnitude parse
    amt = hl.headline.str.extract(r"\$([\d.]+)([MB])")
    hl["amt"] = amt[0].astype(float) * np.where(amt[1]=="B", 1000, 1)
    hl["pct"] = hl.headline.str.extract(r"([\d.]+)%")[0].astype(float)
    return hl, close, ret
def template_effects(hl_full, close, sessions_mask=None, L=10):
    """per-template mean per-bar drift (bps) estimated from full-session data"""
    C = close.to_numpy(); sidx = {s:i for i,s in enumerate(close.index)}
    h = hl_full if sessions_mask is None else hl_full[hl_full.session.isin(sessions_mask)]
    h = h[(h.bar_ix>=1)&(h.bar_ix<=99-L)]
    i = h.session.map(sidx).to_numpy(); b = h.bar_ix.to_numpy()
    raw = C[:,1:]/C[:,:-1]-1; mkt = raw.mean(0)
    cum = np.concatenate([np.zeros((len(C),1)), np.cumsum(np.log(C[:,1:]/C[:,:-1]) - np.log1p(mkt), 1)],1)
    f = (cum[i, b+L-1] - cum[i, b-1]) / L
    df = pd.DataFrame({"tmpl":h.tmpl.values, "verb":h.verb.values, "f":f})
    return df
