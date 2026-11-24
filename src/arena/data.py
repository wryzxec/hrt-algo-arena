from pathlib import Path
import numpy as np
import pandas as pd

def load_sessions(split="ins", root="."):
    """Only bars/headlines 0..49 enter model inputs. Targets are separate."""
    if split not in {"ins", "oos"}: raise ValueError(split)
    root = Path(root)
    bars = pd.read_parquet(root / f"{split}_bars.parquet")
    news = pd.read_parquet(root / f"{split}_headlines.parquet")
    bars = bars[bars.bar_ix.between(0,49)].sort_values(["session","bar_ix"])
    news = news[news.bar_ix.between(0,49)].sort_values(["session","bar_ix"], kind="stable")
    grouped = bars.groupby("session", sort=True)
    if not grouped.bar_ix.apply(lambda x: np.array_equal(x.to_numpy(), np.arange(50))).all():
        raise ValueError("Expected exactly bars 0..49 per session")
    X = grouped[["open","high","low","close"]].agg(list)
    ng = news.groupby("session")
    for column, source in [("headlines","headline"),("headline_bars","bar_ix")]:
        X[column] = ng[source].agg(list).reindex(X.index).apply(lambda x: x if isinstance(x,list) else [])
    y = None
    if split == "ins":
        y = pd.read_parquet(root / "ins_returns.parquet").set_index("session")["returns"]
        if not y.index.is_unique or set(y.index) != set(X.index): raise ValueError("Target mismatch")
        y = y.reindex(X.index)
    return X,y

def partitions(index):
    ids = np.random.default_rng(42).permutation(np.sort(np.asarray(index)))
    n = len(ids)
    return ids[:int(.7*n)], ids[int(.7*n):int(.85*n)], ids[int(.85*n):]
