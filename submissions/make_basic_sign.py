"""Ultra-basic fallback (5-fold CV 2.37, zero fitted parameters): hand-labelled verb sign x drift left / GK vol."""
from pathlib import Path
import numpy as np, pandas as pd
D = Path(__file__).resolve().parent.parent / "data"
POS = {"secures","announces","completes","wins","expands","launches","signs","achieves","beats","raises"}
NEG = {"misses","recalls","loses","lowers","faces","warns","withdraws","delays"}
def verb(h):
    t = h.split(); return t[3] if t[2] in {"Group","Co"} else t[2]
bars = pd.read_parquet(D/"oos_bars.parquet"); bars = bars[bars.bar_ix <= 49]; hl = pd.read_parquet(D/"oos_headlines.parquet"); hl = hl[hl.bar_ix <= 49]
ss = np.sort(bars.session.unique()); P = {c: np.log(bars.pivot(index="session", columns="bar_ix", values=c).reindex(ss).to_numpy()) for c in ("open","high","low","close")}
gk = np.sqrt(np.clip((0.5*(P["high"]-P["low"])**2-(2*np.log(2)-1)*(P["close"]-P["open"])**2).mean(1), 1e-10, None))
b = hl.bar_ix.to_numpy(); w = sum((0.5 if k in (0,10) else 1.0)*((b+k>49)&(b+k<=59)) for k in range(11))
s = hl.headline.map(lambda h: 1.0 if verb(h) in POS else (-1.0 if verb(h) in NEG else 0.0)) * w
p = s.groupby(hl.session).sum().reindex(ss).fillna(0).to_numpy() / gk
out = Path(__file__).resolve().parent/"basic_sign.csv"
pd.DataFrame({"headline_id": ss.astype(np.int64), "pred": p}).to_csv(out, index=False, lineterminator="\n")
sub = pd.read_csv(out); assert len(sub)==10000 and sub.headline_id.is_unique and np.isfinite(sub.pred).all(); print("wrote", out.name)
