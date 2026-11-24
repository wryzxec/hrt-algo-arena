"""Candidate (5-fold in-sample CV 3.53-3.55 over 3 shuffles; repo audit split 3.64):

1. Main-sector headlines only (5 sectors x 16 companies; off-sector headlines have zero price effect).
2. Per-template drift (vol units) fitted jointly by sparse ridge on all in-sample bars 1-99 + OOS bars 1-49
   (label-free); drift profile 0.5,1x9,0.5 over bars b..b+10.  news signal = sum coef x drift bars left after 49.
3. Residual momentum: unexplained bar returns (bars 1-49, vol units) summed with exponential decay,
   half-lives 1,2,3,5,10,25,1000 (short-term continuation, long-term slight reversal).
4. Ridge (alpha 50, standardised) of y/vol on [news signal, 7 residual sums], fitted on in-sample labels.
5. Sharpe-optimal sizing: pred = m / (vol * (Var(y/vol) + m^2)), i.e. E[r]/E[r^2].
Run from repo root: .venv/bin/python submissions/make_sector_resid_sized.py
"""
import hashlib, sys
from pathlib import Path
root = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(root / "research"))
import os; os.chdir(root)
exec(open(root / "research/final_pipeline.py").read())
N = len(ins["S"]); allo = np.arange(len(oos["S"]))
pred = train_predict(np.arange(N), oos, allo, [(oos, allo, 49)])
out = root / "submissions/sector_resid_sized.csv"
pd.DataFrame({"headline_id": oos["S"].astype(np.int64), "pred": pred}).to_csv(out, index=False, lineterminator="\n")
sub = pd.read_csv(out); exp = set(pd.read_parquet(root / "data/oos_bars.parquet").session.unique())
assert list(sub.columns) == ["headline_id", "pred"] and len(sub) == 10000 and sub.headline_id.is_unique
assert set(sub.headline_id) == exp and np.isfinite(sub.pred).all()
prev = pd.read_csv(root / "submissions/sector_joint.csv").set_index("headline_id").pred
print(f"wrote {out.name}: rows={len(sub)} nonzero={(sub.pred != 0).mean():.2f} corr with sector_joint={sub.set_index('headline_id').pred.corr(prev):.3f} "
      f"max|pred|/median|pred|={sub.pred.abs().max()/sub.pred.abs().median():.1f} sha256={hashlib.sha256(out.read_bytes()).hexdigest()[:16]}")
