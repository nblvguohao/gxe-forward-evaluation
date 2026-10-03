"""Adapter for LSTM-GNN (Morshedian & Domaratzki 2026, PLoS Comput Biol, doi:10.1371/journal.pcbi.1013729;
github.com/amma/maize-yield-gxe-gnn, commit 3af3c88). The repository has no LICENSE file, so its code is
imported unmodified at run time from a local checkout (LSTMGNN_CODE) and never redistributed here.

Pipeline as in the public code, architecture C (the paper's main architecture):
  * markers: their encoding is 1/2/3 with -1 missing, so data_v2 calls (0/1/2, -1 missing) are shifted by +1;
    then their preprocess_data.py defaults: random imputation by observed frequency (impute_markers), drop
    markers whose most frequent value has share >= 0.95, prune markers identical in >= 95 % of rows within
    windows of 100 (prune_similar_markers); then RobustScaler and PCA(548) (Arch_C_train.prepare_inputs).
    Their code fits all of these on every row of the genotype file it is given; `fit_ids` chooses the rows.
  * weather: the 5 NASA POWER daily variables of evn_vector_LSTM.WEATHER_FEATURES, median fill and
    StandardScaler, an LSTM autoencoder (21 dims, 200 epochs, best reconstruction kept); its encoder gives the
    environment vectors. `fit_envs` chooses the environments the scaler and the autoencoder are fitted on;
    the others are encoded with the fitted model (their --model-in path).
  * environment vectors: RobustScaler (their prepare_inputs, fitted on the env table it is given).
  * target: RobustScaler on the training rows; loss 0.8 MSE + 0.2 MAE; AdamW(3e-4, wd 1e-4);
    ReduceLROnPlateau on validation RMSE; checkpoint and early stopping (patience 15) on validation PCC;
    batch 32; 30 GATv2 rounds, hidden 128, 8 heads, dropout 0.25, top-k 10; mixed precision on CUDA.
"""
from __future__ import annotations

import importlib
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd


def import_lstmgnn():
    code = os.environ.get("LSTMGNN_CODE", "<workstation>/dart-gxe/third_party/maize-yield-gxe-gnn")
    if code not in sys.path:
        sys.path.insert(0, code)
    return (importlib.import_module("preprocess_data"), importlib.import_module("evn_vector_LSTM"),
            importlib.import_module("Arch_C_train"))


def marker_table(geno, hybrids) -> pd.DataFrame:
    X = geno.X[[geno.index[h] for h in hybrids]].astype(np.int16)
    X = np.where(X < 0, -1, X + 1)  # their 1/2/3 coding, -1 = missing
    return pd.DataFrame(X, index=list(hybrids), columns=[f"m{i}" for i in range(X.shape[1])])


def preprocess_markers(ppd, table: pd.DataFrame, fit_ids, seed: int) -> pd.DataFrame:
    """Their three filters with the decisions (kept columns) taken on `fit_ids` rows. Imputation draws from
    each column's observed frequencies in the rows it is applied to, as their impute_column does."""
    fit = ppd.impute_markers(table.loc[list(fit_ids)], seed, -1)
    keep = ppd.prune_similar_markers(ppd.drop_low_variance_markers(fit, 0.95), 0.95, 100).columns
    rest = [h for h in table.index if h not in set(fit_ids)]
    parts = [fit[keep]]
    if rest:
        parts.append(ppd.impute_markers(table.loc[rest, keep], seed + 1, -1))
    return pd.concat(parts).loc[table.index]


def geno_pcs(markers: pd.DataFrame, fit_ids, n_pc: int = 548, seed: int = 42):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import RobustScaler
    X = markers.to_numpy(np.float32)
    rows = [markers.index.get_loc(h) for h in fit_ids]
    sc = RobustScaler().fit(X[rows])
    Z = sc.transform(X).astype(np.float32)
    n = min(n_pc, len(rows), Z.shape[1])
    solver = "randomized" if n < min(len(rows), Z.shape[1]) else "full"
    pca = PCA(n_components=n, svd_solver=solver, random_state=seed).fit(Z[rows])
    return dict(zip(markers.index, pca.transform(Z).astype(np.float32))), float(pca.explained_variance_ratio_.sum())


def env_vectors(evl, daily: pd.DataFrame, fit_envs, all_envs, seed: int, device: str, epochs: int = 200):
    """Their LSTM autoencoder fitted on `fit_envs`; every env in `all_envs` encoded with it."""
    import torch
    cols = ["env", "date"] + list(evl.WEATHER_FEATURES)
    w = daily[daily["env"].isin(set(all_envs))][cols].rename(columns={"env": "Env", "date": "Date"})
    with tempfile.TemporaryDirectory() as td:
        f_fit, f_all = os.path.join(td, "fit.csv"), os.path.join(td, "all.csv")
        w[w["Env"].isin(set(fit_envs))].to_csv(f_fit, index=False)
        w.to_csv(f_all, index=False)
        evl.set_seed(seed)
        envs_fit, x, lengths, scaler, fill = evl.load_weather(f_fit, "Env", "Date")
        model = evl.EnvironmentLSTM(len(evl.WEATHER_FEATURES), 21).to(device)
        best = evl.train(model, x.to(device), lengths.to(device),
                         SimpleNamespace(lr=1e-3, weight_decay=1e-4, epochs=epochs))
        envs, xa, la, _, _ = evl.load_weather(f_all, "Env", "Date", scaler, fill.to_dict())
        model.eval()
        with torch.no_grad():
            V = model.encode(xa.to(device), la.to(device)).cpu().numpy()
    return dict(zip(envs, V.astype(np.float32))), float(best)


def robust(vectors: dict, fit_keys):
    from sklearn.preprocessing import RobustScaler
    keys = list(vectors)
    M = np.stack([vectors[k] for k in keys])
    sc = RobustScaler().fit(np.stack([vectors[k] for k in fit_keys]))
    return dict(zip(keys, sc.transform(M).astype(np.float32)))


def trait_frame(rows: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"Env": rows["env"].astype(str).to_numpy(), "Hybrid": rows["genotype"].astype(str).to_numpy(),
                         "Yield_Mg_ha": rows["y"].to_numpy(float)})
