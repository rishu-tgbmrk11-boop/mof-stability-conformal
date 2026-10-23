"""Reproduce the (random, NOT group-aware) train/test splits used by the released OptiMOF training scripts."""
from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def paper_random_split(df: pd.DataFrame, label: str, test_size: float = 0.2, random_state: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Same call as the released scripts: train_test_split(features, labels, test_size=0.2, random_state=0)."""
    X = df.loc[:, df.columns != label]
    y = df.loc[:, label]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def paper_val_split(X: pd.DataFrame, y: pd.Series, val_size: float = 0.1, random_state: int = 0):
    """The 10 % validation split the NN scripts carve out of the training part (used for early stopping)."""
    return train_test_split(X, y, test_size=val_size, random_state=random_state)
