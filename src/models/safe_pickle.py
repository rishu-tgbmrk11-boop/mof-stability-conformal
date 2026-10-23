"""Restricted unpickling for UNTRUSTED pickle / joblib / torch files.

A pickle can execute arbitrary code when loaded. These helpers only ever resolve names that are on an explicit allow-list
(numpy, pandas, scikit-learn, scipy, xgboost, collections, and a small set of torch tensor/module classes). Any other global
(os.system, subprocess, builtins.eval, ...) raises `BlockedGlobalError` BEFORE anything is called. Classes that belong to the
repository being audited (e.g. `__main__.ThermalModel`) are mapped to inert stand-ins via `stubs`.

Always run Python as `python -I` when loading untrusted files, so the working directory is not on the import path.
"""
from __future__ import annotations

import io
import pickle
import zipfile
from typing import Any, Dict, Iterable, List, Optional

# module-prefix allow-list for scientific libraries (matched on the module name only)
ALLOWED_MODULE_PREFIXES = (
    "numpy", "pandas", "sklearn", "scipy", "xgboost", "joblib.numpy_pickle", "collections", "copyreg", "_codecs",
)
# builtins that are harmless data constructors (NOT eval/exec/getattr/__import__/open ...)
ALLOWED_BUILTINS = {"slice", "set", "frozenset", "list", "dict", "tuple", "int", "float", "str", "bytes", "bytearray", "complex", "bool", "range", "object"}
# exact torch names needed to rebuild tensors and plain nn modules (no torch.load / storage._load_from_bytes)
ALLOWED_TORCH = {
    ("torch._utils", "_rebuild_tensor_v2"), ("torch._utils", "_rebuild_parameter"), ("torch._utils", "_rebuild_tensor"),
    ("torch", "FloatStorage"), ("torch", "DoubleStorage"), ("torch", "LongStorage"), ("torch", "IntStorage"), ("torch", "BoolStorage"),
    ("torch", "float32"), ("torch", "float64"), ("torch", "Size"),
    ("torch.nn.parameter", "Parameter"), ("torch.nn.modules.container", "ModuleList"), ("torch.nn.modules.linear", "Linear"),
    ("torch.nn.modules.dropout", "Dropout"), ("torch.nn.modules.activation", "ReLU"), ("torch.nn.modules.activation", "LeakyReLU"),
    ("collections", "OrderedDict"),
}
# exact pyarrow helpers that pandas>=3 string columns pickle through (they only rebuild an array from raw buffers)
ALLOWED_EXACT = {("pyarrow.lib", "_restore_array"), ("pyarrow.lib", "type_for_alias"), ("pyarrow.lib", "py_buffer")}  # pure data constructors (no code execution)


class BlockedGlobalError(pickle.UnpicklingError):
    """Raised when an untrusted pickle asks for a global that is not on the allow-list."""


class Stub:
    """Inert stand-in for a repository class: accepts any state, runs no code."""

    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["_state"] = state


def is_allowed(module: str, name: str) -> bool:
    if module in ("builtins", "__builtin__"):  # "__builtin__" is the Python-2 spelling used by old pickle protocols
        return name in ALLOWED_BUILTINS
    if (module, name) in ALLOWED_TORCH or (module, name) in ALLOWED_EXACT:
        return True
    return any(module == p or module.startswith(p + ".") for p in ALLOWED_MODULE_PREFIXES)


def _make_find_class(base, stubs: Optional[Dict[tuple, Any]], seen: Optional[List[tuple]]):
    stubs = stubs or {}

    def find_class(self, module, name):
        if seen is not None:
            seen.append((module, name))
        if (module, name) in stubs:
            return stubs[(module, name)]
        if not is_allowed(module, name):
            raise BlockedGlobalError(f"blocked global {module}.{name}")
        return base.find_class(self, module, name)  # explicit base: type(self) may be a wrapper subclass (torch.load wraps the unpickler)

    return find_class


def safe_unpickler_class(base=pickle.Unpickler, stubs=None, seen=None):
    """Build an Unpickler subclass (of `base`) whose find_class enforces the allow-list."""
    return type("SafeUnpickler", (base,), {"find_class": _make_find_class(base, stubs, seen)})


def safe_loads(data: bytes, stubs=None, seen=None):
    return safe_unpickler_class(stubs=stubs, seen=seen)(io.BytesIO(data)).load()


def safe_read_pickle(path: str, stubs=None, seen=None):
    """Plain pickle file (e.g. written by DataFrame.to_pickle)."""
    with open(path, "rb") as f:
        return safe_unpickler_class(stubs=stubs, seen=seen)(f).load()


def safe_joblib_load(path: str, stubs=None, seen=None):
    """joblib.dump() file: numpy arrays are stored inline, so joblib's own NumpyUnpickler is subclassed with the same allow-list."""
    import joblib
    import joblib.numpy_pickle as npp

    original = npp.NumpyUnpickler
    npp.NumpyUnpickler = safe_unpickler_class(base=original, stubs=stubs, seen=seen)  # joblib.load looks this name up at call time
    try:
        return joblib.load(path)
    finally:
        npp.NumpyUnpickler = original


def safe_torch_load(path: str, stubs=None, seen=None, map_location="cpu"):
    """torch.save() of a whole module: torch.load with weights_only=False but a restricted pickle module."""
    import types

    import torch

    mod = types.ModuleType("safe_pickle_for_torch")
    mod.Unpickler = safe_unpickler_class(stubs=stubs, seen=seen)
    mod.load = pickle.load
    mod.HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL
    mod.UnpicklingError = pickle.UnpicklingError
    return torch.load(path, map_location=map_location, weights_only=False, pickle_module=mod)


def scan_torch_zip_globals(path: str) -> List[tuple]:
    """STATIC listing (no execution) of the globals referenced by a torch zip checkpoint's pickle stream."""
    import pickletools

    out: List[tuple] = []
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.endswith("data.pkl")]
        for n in names:
            data = z.read(n)
            strings: List[str] = []
            for op, arg, _ in pickletools.genops(data):
                if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE"):
                    strings.append(arg)
                elif op.name == "GLOBAL":
                    m, nm = arg.split(" ", 1)
                    out.append((m, nm))
                elif op.name == "STACK_GLOBAL" and len(strings) >= 2:
                    out.append((strings[-2], strings[-1]))
    return sorted(set(out))
