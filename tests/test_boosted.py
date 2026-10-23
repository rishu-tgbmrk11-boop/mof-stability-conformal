import numpy as np
import pytest

from src.models import boosted


@pytest.mark.parametrize("task", ["thermal", "solvent", "water"])
def test_shapes_and_feature_selection(task):
    rng = np.random.RandomState(0)
    X = rng.randn(300, 12); y = (X[:, 0] + 0.1 * rng.randn(300))
    y = y * 100 + 300 if task == "thermal" else (y > 0).astype(int) if task == "solvent" else np.digitize(y, [-1, 0, 1])
    for c in (dict(kind="xgb", depth=3, mcw=1, fs=4, **({"obj": "reg:squarederror"} if task == "thermal" else {"balanced": True})),
              dict(kind="et", leaf=3, fs=None, balanced=False)):
        o = boosted.fit_predict(task, c, X[:200], y[:200], X[200:], 0, n_jobs=2)
        assert len(o["score"]) == 100 and np.isfinite(o["score"]).all()
        if task == "water":
            assert np.allclose(sum(o[f"p{i}"] for i in range(4)), 1)
    assert boosted.select_features(task, X, y, 3, 0, 2).tolist().count(0) == 1   # the informative column is selected


def test_average_and_grid():
    a, b = {"score": np.array([0., 2.])}, {"score": np.array([2., 2.])}
    assert boosted.average([a, b])["score"].tolist() == [1., 2.]
    assert len({boosted.cfg_name(c) for c in boosted.config_grid("water")}) == len(boosted.config_grid("water"))
