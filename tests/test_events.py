import numpy as np
import pandas as pd
import pytest

from arena.models.events import _features, build


def test_timing_and_observable_price_response():
    closes = [100.] * 49 + [110.]
    frame = pd.DataFrame({
        "close": [closes], "headline_bars": [[49, 44]],
        "headlines": [["Firm sees 5% margin improvement", "Firm secures $2B contract"]],
    })
    features, = _features(frame)
    assert features["latest/margin improvement"] == 1
    assert features["recent5/secures"] == .5
    assert features["response/secures"] == pytest.approx(.1)
    assert features["magnitude/secures/million"] == pytest.approx(2000 * 2 ** (-5 / 15))
    assert features["signed5"] == 1.5
    frame.at[0, "headlines"][0] = "Firm sees no margin improvement"
    assert _features(frame)[0]["latest sign"] == -1


def test_future_inputs_labels_and_ids_cannot_affect_predictions():
    frame = pd.DataFrame({
        "close": [[100.] * 50] * 3, "headline_bars": [[49], [], [44]],
        "headlines": [["Firm secures contract"], [], ["Firm misses quarterly estimates"]],
    })
    model = build().fit(frame, [.01, 0, -.01])
    expected = model.predict(frame)
    assert np.isfinite(expected).all() and np.ptp(expected) > 0
    changed = frame.iloc[::-1].reset_index(drop=True).assign(returns=999)
    changed.index = [10, 30, 20]
    changed["close"] = [bars + [np.nan] for bars in changed.close]
    changed["headline_bars"] = [bars + [50] for bars in changed.headline_bars]
    changed["headlines"] = [news + [None] for news in changed.headlines]
    np.testing.assert_array_equal(model.predict(changed), expected[::-1])
