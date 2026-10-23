"""Tests for the restricted unpickler: it must load scientific objects and refuse everything else BEFORE running any code."""
import os
import pickle

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier

from src.models import safe_pickle as sp


class _Evil:
    def __reduce__(self):  # would run `touch <marker>` if it were ever executed
        return (os.system, (f"touch {MARKER}",))


MARKER = "/tmp/_pickle_pwned_marker"


def test_malicious_os_system_is_blocked_and_never_runs(tmp_path):
    if os.path.exists(MARKER):
        os.remove(MARKER)
    data = pickle.dumps(_Evil())
    with pytest.raises(sp.BlockedGlobalError):
        sp.safe_loads(data)
    assert not os.path.exists(MARKER), "payload was executed!"


def test_builtins_eval_is_blocked():
    class E:
        def __reduce__(self):
            return (eval, ("1+1",))

    with pytest.raises(sp.BlockedGlobalError):
        sp.safe_loads(pickle.dumps(E()))


def test_joblib_sklearn_model_roundtrip(tmp_path):
    rng = np.random.RandomState(0)
    X, y = rng.randn(60, 4), rng.randint(0, 2, 60)
    rf = RandomForestClassifier(10, random_state=0).fit(X, y)
    p = tmp_path / "rf.pkl"
    joblib.dump(rf, p)
    seen = []
    rf2 = sp.safe_joblib_load(str(p), seen=seen)
    assert np.array_equal(rf.predict(X), rf2.predict(X))
    assert any(m.startswith("sklearn") for m, _ in seen)


def test_joblib_unpickler_is_restored_after_load(tmp_path):
    import joblib.numpy_pickle as npp

    original = npp.NumpyUnpickler
    p = tmp_path / "a.pkl"
    joblib.dump(np.arange(3), p)
    sp.safe_joblib_load(str(p))
    assert npp.NumpyUnpickler is original


def test_dataframe_plain_pickle_roundtrip(tmp_path):
    df = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    p = tmp_path / "df.pkl"
    df.to_pickle(p)
    out = sp.safe_read_pickle(str(p))
    assert out.equals(df)


def test_joblib_malicious_is_blocked(tmp_path):
    if os.path.exists(MARKER):
        os.remove(MARKER)
    p = tmp_path / "evil.pkl"
    joblib.dump(_Evil(), p)
    with pytest.raises(sp.BlockedGlobalError):
        sp.safe_joblib_load(str(p))
    assert not os.path.exists(MARKER)


class _RepoNet(nn.Module):  # stands in for a class defined in the audited repository
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(3, 2)])


def test_torch_module_needs_stub_then_loads(tmp_path):
    net = _RepoNet()
    p = tmp_path / "net.pt"
    torch.save(net, p)
    globs = sp.scan_torch_zip_globals(str(p))
    assert ("torch.nn.modules.linear", "Linear") in globs
    with pytest.raises(sp.BlockedGlobalError):  # repo class is not allow-listed
        sp.safe_torch_load(str(p))
    stubs = {(_RepoNet.__module__, "_RepoNet"): _RepoNet}
    net2 = sp.safe_torch_load(str(p), stubs=stubs)
    assert torch.equal(net.layers[0].weight, net2.layers[0].weight)
