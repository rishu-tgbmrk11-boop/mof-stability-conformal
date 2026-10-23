"""Load the three stability tasks from the released CSVs, using each task's OWN CSV rows (not the merged all_in_one table).

The merge in the released `data_selection.py` replaces some water/solvent features with thermal-table values (see notes/stability_inventory.md);
here every task keeps the features computed for its own table. The feature COLUMNS are the released ones (170 / 170 / 159)."""
from __future__ import annotations

import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

RAW = "data/raw/optimof/data/"
FEATURE_COLUMNS_JSON = "data/processed/phase2/feature_columns.json"
TASKS = ("thermal", "solvent", "water")
THERMAL_STABLE_C = 350.0            # SI Section 4: thermal screening threshold (>= 350 degC)
WATER_STABLE_MIN_CLASS = 2           # SI Section 4: water categories 2 and 3 (0-based) count as stable


def _csv(task: str) -> Tuple[pd.DataFrame, str, str]:
    if task == "thermal":
        return pd.read_csv(RAW + "thermal/thermal_all_data.csv"), "refcode", "T"
    if task == "solvent":
        return pd.read_csv(RAW + "solvent/solvent_all_data.csv"), "refcode", "flag"
    if task == "water":
        return pd.read_csv(RAW + "water_and_haz/data.csv"), "MOF_name", "water_label"
    raise ValueError(task)


def feature_columns() -> Dict[str, list]:
    return json.load(open(FEATURE_COLUMNS_JSON))


def load_task(task: str) -> pd.DataFrame:
    """Index = MOF name. Columns: the released feature columns, `label_raw`, `y` (target: thermal degC, solvent 0/1, water class 0-3) and `stable` (0/1)."""
    df, idc, lab = _csv(task)
    cols = feature_columns()[task]
    out = df.set_index(idc)[cols].astype(float).copy()
    raw = df.set_index(idc)[lab]
    out["label_raw"] = raw.values
    if task == "thermal":
        out["y"] = raw.values.astype(float)
        out["stable"] = (raw.values >= THERMAL_STABLE_C).astype(int)
    elif task == "solvent":
        out["y"] = (raw.values > 0).astype(int)
        out["stable"] = out["y"].values
    else:
        out["y"] = (raw.values - 1).astype(int)
        out["stable"] = (out["y"].values >= WATER_STABLE_MIN_CLASS).astype(int)
    out.index.name = "name"
    return out


def doi_by_name() -> Dict[str, str]:
    s = pd.read_csv(RAW + "solvent/solvent_all_data.csv").set_index("refcode")["doi"]
    return s.to_dict()


def feature_matrix(df: pd.DataFrame, task: str) -> np.ndarray:
    return df[feature_columns()[task]].values.astype(np.float64)
