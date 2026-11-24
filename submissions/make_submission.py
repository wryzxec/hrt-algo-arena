"""Write the submission CSV from sector_model.py, fitted on all 10,000 in-sample sessions.

    .venv/bin/python submissions/make_submission.py [output.csv]

Default output: submissions/sector_resid_sized.csv (columns headline_id,pred; one row per OOS session).
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sector_model import DATA, HALF, load_all, train_predict  # noqa: E402


def main(out: Path):
    ins, oos, y = load_all()
    N, allo = len(ins["S"]), np.arange(len(oos["S"]))
    pred = train_predict(ins, oos, y, np.arange(N), oos, allo, [(oos, allo, HALF)])

    pd.DataFrame({"headline_id": oos["S"].astype(np.int64), "pred": pred}).to_csv(out, index=False, lineterminator="\n")

    sub = pd.read_csv(out)
    expected = set(pd.read_parquet(DATA / "oos_bars.parquet").session.unique())
    assert list(sub.columns) == ["headline_id", "pred"], "wrong columns"
    assert len(sub) == 10000 and sub.headline_id.is_unique, "wrong row count or duplicate ids"
    assert set(sub.headline_id) == expected, "ids do not match OOS sessions"
    assert np.isfinite(sub.pred).all(), "non-finite predictions"
    print(f"wrote {out}: rows={len(sub)} sha256={hashlib.sha256(out.read_bytes()).hexdigest()[:16]}")


if __name__ == "__main__":
    default = Path(__file__).resolve().parent / "sector_resid_sized.csv"
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else default)
