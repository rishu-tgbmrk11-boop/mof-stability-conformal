import numpy as np
import torch
import torch.nn as nn

from src.eval.metrics import cluster_bootstrap, ece
from src.models.stability_nets import ThermalNet, train_released_recipe


def test_ece_zero_for_calibrated_and_large_for_overconfident():
    rng = np.random.RandomState(0)
    p = rng.rand(20000); y = (rng.rand(20000) < p).astype(int)
    assert ece(p, y) < 0.02
    assert ece(np.where(y == 1, 0.99, 0.01) * 0 + 0.99, y) > 0.3   # always 0.99 but only ~half positive


def test_cluster_bootstrap_interval_covers_mean_and_widens_with_clustering():
    rng = np.random.RandomState(1)
    g = np.repeat(np.arange(50), 10)
    x = rng.randn(50)[g] * 2 + rng.randn(500) * 0.1      # strong within-group correlation
    pt, lo, hi = cluster_bootstrap(lambda i: x[i].mean(), g, 500, seed=0)
    pt2, lo2, hi2 = cluster_bootstrap(lambda i: x[i].mean(), np.arange(500), 500, seed=0)   # ignoring groups
    assert lo <= pt <= hi and (hi - lo) > (hi2 - lo2)


def test_fixed_restore_really_restores_best_epoch():
    rng = np.random.RandomState(0)
    X = rng.randn(240, 6).astype("float32"); y = (X[:, 0] * 2 + rng.randn(240)).astype("float32")
    net = ThermalNet(6, [64, 64], 1)
    h = train_released_recipe(net, X[:160], y[:160], X[160:], y[160:], nn.MSELoss(), 0.02, 32, 300, 5, seed=0, device="cpu", faithful_best_state=False)
    net.eval()
    with torch.no_grad():
        val = sum(nn.MSELoss()(net(torch.tensor(X[160:][i:i + 32])), torch.tensor(y[160:][i:i + 32]).unsqueeze(1)).item() for i in range(0, 80, 32)) / 80
    assert abs(val - h["best_val"]) < 1e-5 and h["best_epoch"] < h["epochs_run"] - 1 and not h["restore_changed_nothing"]
