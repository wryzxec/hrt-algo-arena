"""Extremely basic model: main-sector headlines x drift still to come / volatility. No momentum, no regression.

pred = sum over main-sector headlines (bars <= 49) of effect[template] x remaining_drift(bar) / GK vol

effect[template] = plain average, over in-sample main-sector headlines at bars 1..89, of the stock's
next-10-bar drift per bar in volatility units (market-adjusted). One number per template, nothing tuned.
Variant "sign": effect = +1 / -1 / 0 from a hand-labelled verb list (zero fitted parameters).

    .venv/bin/python submissions/basic_model.py          # CV both variants + write basic_template.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sector_model import DATA, HALF, load_all, remaining_drift, sharpe  # noqa: E402

POS = {"secures", "announces", "completes", "wins", "expands", "launches", "signs", "achieves", "beats", "raises"}
NEG = {"misses", "recalls", "loses", "lowers", "faces", "warns", "withdraws", "delays"}


def template_effects(d, idx):
    """Average next-10-bar drift per bar (vol units) of each template, from sessions idx (bars 1..89)."""
    C, gk = d["C"], d["gk"]
    h = d["hl"][d["hl"].i.isin(idx) & (d["hl"].bar_ix >= 1) & (d["hl"].bar_ix <= 89)]
    R = np.diff(C, axis=1)
    cum = np.concatenate([np.zeros((len(C), 1)), np.cumsum(R - R.mean(0), axis=1)], axis=1)
    i, b = h.i.to_numpy(), h.bar_ix.to_numpy()
    drift = (cum[i, b + 9] - cum[i, b - 1]) / 10 / gk[i]
    return pd.Series(drift).groupby(h.tmpl.to_numpy()).mean()


def sign_effects(d):
    tm = d["hl"].tmpl.unique()
    verb = pd.Series(tm, index=tm).str.split().str[0]
    return verb.map(lambda v: 1.0 if v in POS else (-1.0 if v in NEG else 0.0))


def predict(d, idx, eff):
    h = d["hl"][d["hl"].i.isin(idx) & (d["hl"].bar_ix <= HALF)]
    s = (h.tmpl.map(eff).fillna(0) * remaining_drift(h.bar_ix.to_numpy())).groupby(h.i).sum()
    return s.reindex(idx).fillna(0).to_numpy() / d["gk"][idx]


if __name__ == "__main__":
    ins, oos, y = load_all()
    N = len(ins["S"])
    sgn = sign_effects(ins)
    for seed in (0, 1):
        folds = list(KFold(5, shuffle=True, random_state=seed).split(np.arange(N)))
        pt, ps = np.zeros(N), np.zeros(N)
        for a, b in folds:
            pt[b] = predict(ins, b, template_effects(ins, a))
            ps[b] = predict(ins, b, sgn)
        for name, p in (("template averages", pt), ("verb sign (0 params)", ps)):
            print(f"seed {seed} {name:22s} 5-fold Sharpe {sharpe(p, y):.3f}  folds {[round(sharpe(p[b], y[b]), 2) for _, b in folds]}")

    allo = np.arange(len(oos["S"]))
    pred = predict(oos, allo, template_effects(ins, np.arange(N)))
    out = Path(__file__).resolve().parent / "basic_template.csv"
    pd.DataFrame({"headline_id": oos["S"].astype(np.int64), "pred": pred}).to_csv(out, index=False, lineterminator="\n")
    sub = pd.read_csv(out)
    assert list(sub.columns) == ["headline_id", "pred"] and len(sub) == 10000 and sub.headline_id.is_unique
    assert set(sub.headline_id) == set(pd.read_parquet(DATA / "oos_bars.parquet").session) and np.isfinite(sub.pred).all()
    full = pd.read_csv(Path(__file__).resolve().parent / "sector_resid_sized.csv").pred
    print(f"wrote {out.name}: rows={len(sub)} corr with sector_resid_sized={np.corrcoef(sub.pred, full)[0, 1]:.3f}")
