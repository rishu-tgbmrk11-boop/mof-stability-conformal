"""Phase 6 (1b, 1c): the released thermal recipe on the CSV random_state=0 split (SI test set, 627 MOFs), repeated with 50 different RECORDED torch seeds
(torch.manual_seed(seed) before the network is built; seeds 0..49), everything else as released (leaky-ReLU 170-2048-...-32-1 net, Adam lr 1e-4, batch 512, patience 100,
max 1000 epochs, MSE on UNSCALED temperatures, scaler fitted on the 80% training part, 10% validation split with random_state=0, as-coded early-stopping restore = no-op).
Variants for 1c (all fixed in advance, all reported, none selected on the test set; 50 seeds each):
  baseline            the released recipe
  target_scaled       temperatures standardised with training mean/sd for training; predictions mapped back
  clip5               standardised features clipped to [-5, 5] (train, validation and test)
  drop_extreme_rows   training rows with any |z| > 10 removed (scaler refitted on the kept rows)
  quantile            QuantileTransformer(normal, 200 quantiles) fitted on the training part instead of StandardScaler
  target_scaled+clip5 both of the first two
  S5, S5+target_scaled, S5+quantile+target_scaled   (ADDED POST HOC) SI Table S5 architecture / learning rate / epochs with the same preprocessing options
  quantile+target_scaled   (ADDED POST HOC, after seeing that quantile and target scaling each close part of the gap; diagnostic only, not a model selection)
usage: python -I scripts/p6_01b_thermal_seeds.py VARIANT [VARIANT ...]   (repo root)"""
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from src.features import stability_data as sd
from src.models.stability_nets import SolventNet, ThermalNet, predict, train_released_recipe

OUT = "data/processed/phase6/"
SEEDS = list(range(50))
dev = "cuda" if torch.cuda.is_available() else "cpu"
d = sd.load_task("thermal"); cols = sd.feature_columns()["thermal"]
tr, te = train_test_split(np.arange(len(d)), test_size=0.2, random_state=0)
Xtr_raw, Xte_raw = d.iloc[tr][cols].values.astype(float), d.iloc[te][cols].values.astype(float)
ytr_raw, yte = d.iloc[tr]["label_raw"].values.astype(float), d.iloc[te]["label_raw"].values.astype(float)
assert len(te) == 627


def prepare(variant):
    keep = np.ones(len(Xtr_raw), bool)
    if "drop_extreme_rows" in variant:
        z = StandardScaler().fit(Xtr_raw); keep = (np.abs(z.transform(Xtr_raw)) <= 10).all(1)
    if "quantile" in variant:
        sc = QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=0).fit(Xtr_raw[keep])
    else:
        sc = StandardScaler().fit(Xtr_raw[keep])
    A, B = sc.transform(Xtr_raw[keep]), sc.transform(Xte_raw)
    if "clip5" in variant:
        A, B = np.clip(A, -5, 5), np.clip(B, -5, 5)
    y = ytr_raw[keep]
    mu, sdv = (y.mean(), y.std()) if "target_scaled" in variant else (0.0, 1.0)
    Xa, Xv, ya, yv = train_test_split(A, (y - mu) / sdv, test_size=0.1, random_state=0)
    return Xa, ya, Xv, yv, B, mu, sdv, int((~keep).sum())


for variant in sys.argv[1:]:
    Xa, ya, Xv, yv, B, mu, sdv, n_drop = prepare(variant); rows = []
    print(f"\n### {variant}: fit {len(Xa)} / val {len(Xv)} / test {len(B)}; training rows dropped {n_drop}; target mean/sd used {mu:.1f}/{sdv:.1f}", flush=True)
    for s in SEEDS:
        t0 = time.time(); torch.manual_seed(s); np.random.seed(s)
        if "S5" in variant:                                  # SI Table S5 thermal settings: ReLU + dropout 0.124, hidden 128-512-2048-256-1024, lr 0.002, 100 epochs (batch size / patience not given: released values)
            net = SolventNet(170, [128, 512, 2048, 256, 1024], 1, dropout_prob=0.124); lr, ep = 2e-3, 100
        else:
            net = ThermalNet(170, [2048, 1024, 512, 256, 128, 64, 32], 1); lr, ep = 1e-4, 1000
        h = train_released_recipe(net, Xa, ya, Xv, yv, nn.MSELoss(), lr, 512, ep, 100, seed=s, device=dev, faithful_best_state=True)
        p = predict(net, B, dev) * sdv + mu
        rows.append(dict(seed=s, r2=r2_score(yte, p), rmse=mean_squared_error(yte, p) ** .5, mae=mean_absolute_error(yte, p), epochs_run=h["epochs_run"], best_epoch=h["best_epoch"], restore_noop=bool(h["restore_changed_nothing"]), secs=round(time.time() - t0, 1)))
    df = pd.DataFrame(rows)
    print(f"  R2 mean {df.r2.mean():.3f} sd {df.r2.std():.3f} min {df.r2.min():.3f} p50 {df.r2.quantile(.5):.3f} p90 {df.r2.quantile(.9):.3f} p99 {df.r2.quantile(.99):.3f} max {df.r2.max():.3f}; RMSE mean {df.rmse.mean():.1f}; MAE mean {df.mae.mean():.1f}; share R2>=0.41: {(df.r2 >= .41).mean():.2f}; restore no-op in {int(df.restore_noop.sum())}/{len(df)}", flush=True)
    json.dump(dict(variant=variant, test_sd_ddof0=float(yte.std()), n_dropped_train=n_drop, runs=rows), open(OUT + f"thermal_seeds_{variant}.json", "w"), indent=1, default=float)
print("DONE")
