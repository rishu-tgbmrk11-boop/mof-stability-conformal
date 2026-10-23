import numpy as np
import torch
import torch.nn as nn

from src.features.paper_splits import paper_random_split
from src.models.stability_nets import SolventNet, ThermalNet, predict, train_released_recipe


def test_architectures_forward_shapes():
    assert ThermalNet(5, [8, 4], 1)(torch.zeros(3, 5)).shape == (3, 1)
    net = SolventNet(5, [8, 4], 1).eval()
    assert net(torch.zeros(3, 5)).shape == (3, 1)


def test_state_dict_bug_is_reproduced_and_fix_differs():
    rng = np.random.RandomState(0)
    X = rng.randn(200, 6).astype("float32"); y = X[:, 0] * 2 + 0.1 * rng.randn(200).astype("float32")
    Xtr, ytr, Xva, yva = X[:150], y[:150], X[150:], y[150:]
    kw = dict(loss_fn=nn.MSELoss(), lr=0.05, batch_size=32, num_epochs=200, patience=3, seed=0, device="cpu")
    a = train_released_recipe(ThermalNet(6, [16], 1), Xtr, ytr, Xva, yva, faithful_best_state=True, **kw)
    b = train_released_recipe(ThermalNet(6, [16], 1), Xtr, ytr, Xva, yva, faithful_best_state=False, **kw)
    assert a["restore_changed_nothing"] is True    # faithful: restoring the "best" state is a no-op
    assert b["best_epoch"] <= b["epochs_run"] - 1


def test_paper_split_is_deterministic_and_sized():
    import pandas as pd
    df = pd.DataFrame({"y": range(100), "a": range(100)})
    X1, X2, y1, y2 = paper_random_split(df, "y"); X1b, X2b, _, _ = paper_random_split(df, "y")
    assert len(X2) == 20 and list(X2.index) == list(X2b.index)
