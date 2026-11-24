import numpy as np
import pandas as pd
import pytest

from arena.evaluate import score
from arena.validate import validate_predictions


def test_aligns_by_session_not_row_position():
    frame = pd.DataFrame({"headline_id": [12, 10, 11], "pred": [.2, .3, .1]})
    np.testing.assert_array_equal(validate_predictions(frame, [10, 11, 12]), [.3, .1, .2])


@pytest.mark.parametrize("frame", [
    pd.DataFrame({"headline_id": [10, 10], "pred": [1., 2.]}),
    pd.DataFrame({"headline_id": [10, 12], "pred": [1., 2.]}),
    pd.DataFrame({"headline_id": [10], "pred": [1.]}),
    pd.DataFrame({"headline_id": [10.5, 11.], "pred": [1., 2.]}),
    pd.DataFrame({"headline_id": [10, 11], "pred": [np.inf, 2.]}),
    pd.DataFrame({"headline_id": [10, 11], "pred": [np.nan, 2.]}),
    pd.DataFrame({"headline_id": [10, 11], "pred": ["oops", "2"]}),
    pd.DataFrame({"headline_id": [10, 11], "pred": [1., 2.], "index": [0, 1]}),
    pd.DataFrame({"pred": [1., 2.], "headline_id": [10, 11]}),
])
def test_rejects_invalid_submission(frame):
    with pytest.raises(ValueError):
        validate_predictions(frame, [10, 11])


def test_score_rejects_empty_and_overflow():
    with pytest.raises(ValueError):
        score([], [])
    with pytest.raises(ValueError):
        score([1e300, 1.], [1e300, 1.])


def test_evaluation_preserves_existing_report(tmp_path, monkeypatch):
    from arena import evaluate

    previous = tmp_path / "metrics.json"
    previous.write_text('{"sharpe": 3.12}\n')
    monkeypatch.setattr("sys.argv", ["evaluate", "--model", "arena.models.events",
                                     "--out", str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        evaluate.main()
    assert error.value.code == 2
    assert previous.read_text() == '{"sharpe": 3.12}\n'
    assert list(tmp_path.iterdir()) == [previous]
