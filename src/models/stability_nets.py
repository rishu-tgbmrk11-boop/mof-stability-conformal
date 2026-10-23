"""Architectures of the two released OptiMOF neural networks, written to match the classes pickled in model/*.pkl
(`__main__.ThermalModel`, `__main__.SolventModel`), plus a training loop that reproduces the released recipe."""
from __future__ import annotations

import copy
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ThermalNet(nn.Module):
    """Fully connected regressor, leaky-ReLU between layers, no activation on the output (as in thermal_model.py)."""

    def __init__(self, input_size: int = 0, hidden_layer_sizes: Optional[List[int]] = None, output_size: int = 1):
        super().__init__()
        sizes = [input_size] + list(hidden_layer_sizes or []) + [output_size]
        self.graph_depth = len(sizes)
        self.layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i + 1]) for i in range(self.graph_depth - 1)])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.leaky_relu(x)
        return x


class SolventNet(nn.Module):
    """Fully connected binary classifier (logit output), ReLU + dropout between layers (as in solvent_model.py)."""

    def __init__(self, input_size: int = 0, hidden_layer_sizes: Optional[List[int]] = None, output_size: int = 1, dropout_prob: float = 0.2):
        super().__init__()
        sizes = [input_size] + list(hidden_layer_sizes or []) + [output_size]
        self.graph_depth = len(sizes)
        self.layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i + 1]) for i in range(self.graph_depth - 1)])
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.dropout(F.relu(x))
        return x


@torch.no_grad()
def predict(net: nn.Module, X: np.ndarray, device: str = "cpu", batch: int = 2048) -> np.ndarray:
    """Raw network output (regression value, or logit for the classifier) in eval mode."""
    net = net.to(device).eval()
    out = []
    for i in range(0, len(X), batch):
        out.append(net(torch.tensor(X[i:i + batch], dtype=torch.float32, device=device)).cpu().numpy().ravel())
    return np.concatenate(out)


def train_released_recipe(net: nn.Module, Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray, loss_fn: nn.Module, lr: float, batch_size: int,
                          num_epochs: int, patience: int, seed: int, device: str = "cuda", faithful_best_state: bool = True) -> Dict:
    """Training loop of the released scripts: Adam, mini-batch, early stopping on validation loss.

    faithful_best_state=True reproduces `best_model = self.model.state_dict()` (a dict of REFERENCES to the live parameters), so the
    later `load_state_dict(best_model)` changes nothing. False stores a real copy (`copy.deepcopy`), i.e. genuinely restores the best epoch.
    The validation loss is computed as the released code does: sum of batch-mean losses divided by the number of validation samples.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    net.to(device)
    tX, ty = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32).unsqueeze(1)
    vX, vy = torch.tensor(Xva, dtype=torch.float32, device=device), torch.tensor(yva, dtype=torch.float32, device=device).unsqueeze(1)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    best_val, best_state, no_improve, hist, best_epoch = float("inf"), None, 0, [], -1
    for epoch in range(num_epochs):
        net.train()
        perm = torch.randperm(len(tX), generator=gen)
        for i in range(0, len(tX), batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = tX[idx].to(device), ty[idx].to(device)
            loss = loss_fn(net(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val = 0.0
            for i in range(0, len(vX), batch_size):
                val += loss_fn(net(vX[i:i + batch_size]), vy[i:i + batch_size]).item()
            val /= len(vX)
        hist.append(val)
        if val < best_val:
            best_val, no_improve, best_epoch = val, 0, epoch
            best_state = net.state_dict() if faithful_best_state else copy.deepcopy(net.state_dict())
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    final_state_before = {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    restored_equals_final = all(torch.equal(final_state_before[k], v) for k, v in net.state_dict().items())
    return {"epochs_run": len(hist), "best_epoch": best_epoch, "best_val": best_val, "final_val": hist[-1], "restore_changed_nothing": restored_equals_final, "val_history": hist}
