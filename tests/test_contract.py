import numpy as np
import pytest
from arena.data import load_sessions,partitions
from arena.evaluate import score

def test_boundary():
    X,y=load_sessions()
    assert len(X)==len(y)==10000 and X.index.equals(y.index)
    assert "returns" not in X
    assert all(len(x)==50 for x in X.close)
    assert all(all(0<=b<=49 for b in bars) for bars in X.headline_bars)
    a,b,c=partitions(X.index)
    assert len(set(a)|set(b)|set(c))==10000
    assert not (set(a)&set(b) or set(a)&set(c) or set(b)&set(c))
    Z,target=load_sessions("oos")
    assert len(Z)==10000 and target is None
    assert not set(Z.index)&set(X.index)

def test_score():
    assert score([1,2,3],[1,1,1])["sharpe"]==pytest.approx(np.sqrt(252)*2/np.std([1,2,3]))
    with pytest.raises(ValueError): score([1,2],[np.nan,1])
