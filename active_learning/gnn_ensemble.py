#!/usr/bin/env python
# coding: utf-8
"""
GC-MPNN Deep Ensemble + Predictive Entropy + Per-Pair CSV Export
=================================================================
Trains N_ENSEMBLE independent models per LOGO fold.
Epistemic uncertainty = variance across ensemble members.

Key variable
------------
  N_ENSEMBLE : int   number of ensemble members (default 5, try 10)

CSV output: entropy_per_pair.csv
---------------------------------
One row per (polymer, eval_gas, fold_held_out) triplet.
Columns:
    smiles          — polymer repeat unit SMILES
    polymer_idx     — integer index into all_graphs for that gas
    eval_gas        — gas being evaluated (He/H2/N2/O2/CH4/CO2)
    fold_held_out   — which gas was excluded from training in this fold
    is_ood          — True when eval_gas == fold_held_out
    y_true          — experimental log permeability (original scale)
    y_pred_mean     — ensemble mean prediction (original scale)
    epistemic_std   — std across ensemble members (original scale)
    entropy_nats    — H = 0.5 * log(2πe σ²)
    mse             — (y_pred_mean - y_true)²

Other outputs per LOGO fold
----------------------------
  kde_entropy_mse_ensemble_holdout_{gas}.png
  violin_entropy_ensemble_holdout_{gas}.png
  logo_entropy_heatmap_ensemble.png
  logo_entropy_matrix_ensemble.csv
  logo_ensemble_results.pt
"""

import math
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, spearmanr

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import (MessagePassing, global_mean_pool,
                                global_add_pool, GlobalAttention)
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler
from rdkit import Chem

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
#   KEY VARIABLE
# ─────────────────────────────────────────────────────────────
N_ENSEMBLE = 20

# ─────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'Using CUDA: {torch.cuda.get_device_name(0)}')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using MPS (Apple Silicon)')
else:
    device = torch.device('cpu')
    print('Using CPU')

NUM_WORKERS = 2 if device.type == 'cuda' else 0
PIN_MEMORY  = device.type == 'cuda'

# ─────────────────────────────────────────────────────────────
# Best Hyperparameters
# ─────────────────────────────────────────────────────────────
BEST_PARAMS = {
    'learning_rate' : 0.0004989483263215937,
    'l2_lambda'     : 0.0045419658909029055,
    'hidden_dim'    : 256,
    'num_mp_layers' : 3,
    'fusion_dim'    : 16,
    'dropout'       : 0.013135488457537477,
    'pooling'       : 'mean',
    'batch_size'    : 128,
}

FINAL_EPOCHS   = 500
FINAL_PATIENCE = 100

# ─────────────────────────────────────────────────────────────
# Gas Properties & Features
# ─────────────────────────────────────────────────────────────
GAS_PROPERTIES = {
    'He':  {'sigma': 2.551, 'epsilon':  10.2, 'omega': -0.383, 'Tc':   5.2, 'Pc':  2.28, 'd': 2.6,  'Vd':  2.67},
    'H2':  {'sigma': 2.827, 'epsilon':  59.7, 'omega': -0.265, 'Tc':  33.2, 'Pc': 13.00, 'd': 2.89, 'Vd':  6.12},
    'N2':  {'sigma': 3.798, 'epsilon':  71.4, 'omega':  0.037, 'Tc': 126.2, 'Pc': 63.14, 'd': 3.64, 'Vd': 18.5 },
    'O2':  {'sigma': 3.467, 'epsilon': 106.7, 'omega':  0.022, 'Tc': 154.6, 'Pc': 50.43, 'd': 3.46, 'Vd': 16.3 },
    'CH4': {'sigma': 3.758, 'epsilon': 148.6, 'omega':  0.011, 'Tc': 190.6, 'Pc': 46.1,  'd': 3.8,  'Vd': 24.42},
    'CO2': {'sigma': 3.941, 'epsilon': 195.2, 'omega':  0.253, 'Tc': 304.1, 'Pc': 73.80, 'd': 3.3,  'Vd': 26.9 },
}

GASES      = ['He', 'H2', 'N2', 'O2', 'CH4', 'CO2']
GAS_COLORS = {'He':'#e41a1c', 'H2':'#377eb8', 'N2':'#4daf4a',
              'O2':'#984ea3', 'CH4':'#ff7f00', 'CO2':'#a65628'}
FEATURE_DIM = 7

def get_gas_features(gas_name: str) -> np.ndarray:
    p = GAS_PROPERTIES[gas_name]
    return np.array([p['sigma'], p['epsilon'], p['omega'],
                     p['Tc'],    p['Pc'],      p['d'],  p['Vd']],
                    dtype=np.float32)

# ─────────────────────────────────────────────────────────────
# Seed Helper
# ─────────────────────────────────────────────────────────────
def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ─────────────────────────────────────────────────────────────
# Graph Construction  (g.smiles stored for CSV)
# ─────────────────────────────────────────────────────────────
def smiles_to_graph(smiles_str: str):
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        return None
    node_features = []
    for atom in mol.GetAtoms():
        node_features.append([
            atom.GetAtomicNum(), atom.GetDegree(), atom.GetFormalCharge(),
            atom.GetHybridization().real, int(atom.GetIsAromatic()),
            atom.GetTotalNumHs(), atom.GetNumImplicitHs(),
        ])
    x = torch.tensor(node_features, dtype=torch.float)
    edge_indices, edge_features = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices.extend([[i, j], [j, i]])
        ef = [
            int(bond.GetBondType() == Chem.BondType.SINGLE),
            int(bond.GetBondType() == Chem.BondType.DOUBLE),
            int(bond.GetBondType() == Chem.BondType.TRIPLE),
            int(bond.GetBondType() == Chem.BondType.AROMATIC),
            int(bond.GetIsConjugated()), int(bond.IsInRing()),
            int(bond.GetBondType() == Chem.BondType.SINGLE and
                not bond.IsInRing() and
                bond.GetBeginAtom().GetDegree() > 1 and
                bond.GetEndAtom().GetDegree() > 1),
        ]
        edge_features.extend([ef, ef])
    if len(edge_indices) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, 7), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(edge_features, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def build_dataset(pol_sd: pd.DataFrame) -> list:
    cols = {'CH4':'p_exp_CH4', 'CO2':'p_exp_CO2', 'H2':'p_exp_H2',
            'N2':'p_exp_N2',  'O2':'p_exp_O2',   'He':'p_exp_He'}
    dataset = []
    for idx in range(len(pol_sd)):
        smi   = pol_sd['smiles_string'].iloc[idx]
        graph = smiles_to_graph(smi)
        if graph is None:
            continue
        for gas, col in cols.items():
            perm = pol_sd[col].iloc[idx]
            if np.isnan(perm):
                continue
            g              = graph.clone()
            g.gas_features = torch.tensor(get_gas_features(gas),
                                          dtype=torch.float).unsqueeze(0)
            g.y            = torch.tensor([perm], dtype=torch.float)
            g.gas_name     = gas
            g.smiles       = smi          # ← stored for CSV export
            dataset.append(g)
    return dataset

# ─────────────────────────────────────────────────────────────
# GC-MPNN Architecture
# ─────────────────────────────────────────────────────────────
class MPNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim):
        super().__init__(aggr='add')
        self.message_mlp = nn.Sequential(
            nn.Linear(in_channels * 2 + edge_dim, out_channels), nn.ReLU(),
            nn.Linear(out_channels, out_channels))
        self.update_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels), nn.ReLU(),
            nn.Linear(out_channels, out_channels))

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        return self.message_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))

    def update(self, aggr_out, x):
        return self.update_mlp(torch.cat([x, aggr_out], dim=-1))


class GasConditionedMPNN(nn.Module):
    def __init__(self, node_features=7, edge_features=7, gas_features=FEATURE_DIM,
                 hidden_dim=256, num_mp_layers=3, fusion_dim=16,
                 l2_lambda=0.0045, dropout=0.013, pooling='mean'):
        super().__init__()
        self.l2_lambda        = l2_lambda
        self.expected_gas_dim = gas_features
        self.pooling_type     = pooling

        self.node_embedding = nn.Linear(node_features, hidden_dim)
        self.mp_layers   = nn.ModuleList([
            MPNNLayer(hidden_dim, hidden_dim, edge_features)
            for _ in range(num_mp_layers)])
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_mp_layers)])
        self.dropout = nn.Dropout(dropout)

        if pooling == 'attention':
            gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                 nn.Tanh(), nn.Linear(hidden_dim, 1))
            self.attention_pool = GlobalAttention(gate)

        self.gas_encoder = nn.Sequential(
            nn.Linear(gas_features, 64), nn.ReLU(), nn.LayerNorm(64),
            nn.Dropout(dropout),
            nn.Linear(64, fusion_dim), nn.ReLU(), nn.LayerNorm(fusion_dim))

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + fusion_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),                     nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),                      nn.ReLU(),
            nn.Linear(64, 1))

    def forward(self, data):
        x, edge_index, edge_attr, batch = (data.x, data.edge_index,
                                           data.edge_attr, data.batch)
        gas_features = data.gas_features
        batch_size   = batch.max().item() + 1
        if gas_features.shape != (batch_size, self.expected_gas_dim):
            raise ValueError(f'gas_features shape mismatch: '
                             f'{gas_features.shape} vs [{batch_size},{self.expected_gas_dim}]')
        x = F.relu(self.node_embedding(x))
        for mp_layer, bn in zip(self.mp_layers, self.batch_norms):
            x_new = mp_layer(x, edge_index, edge_attr)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = self.dropout(x_new)
            x     = x + x_new
        if self.pooling_type == 'mean':
            polymer_emb = global_mean_pool(x, batch)
        elif self.pooling_type == 'sum':
            polymer_emb = global_add_pool(x, batch)
        elif self.pooling_type == 'attention':
            polymer_emb = self.attention_pool(x, batch)
        else:
            polymer_emb = global_mean_pool(x, batch)
        gas_emb  = self.gas_encoder(gas_features)
        combined = torch.cat([polymer_emb, gas_emb], dim=-1)
        return self.fusion(combined).squeeze(-1)

    def l2_regularization(self):
        l2 = torch.tensor(0., device=next(self.parameters()).device)
        for p in self.parameters():
            l2 += torch.norm(p, 2)
        return self.l2_lambda * l2

# ─────────────────────────────────────────────────────────────
# Training Utilities
# ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, scheduler=None):
    model.train()
    total = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out  = model(data)
        loss = criterion(out, data.y) + model.l2_regularization()
        loss.backward()
        optimizer.step()
        total += loss.item()
    if scheduler:
        scheduler.step()
    return total / len(loader)


def evaluate(model, loader, criterion):
    model.eval()
    total = 0.0
    preds, targets = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out  = model(data)
            loss = criterion(out, data.y)
            total   += loss.item()
            preds   .extend(out.cpu().numpy().tolist())
            targets .extend(data.y.cpu().numpy().tolist())
    return total / len(loader), np.array(preds), np.array(targets)


def metrics(y_true, y_pred):
    mae    = np.mean(np.abs(y_true - y_pred))
    mse    = np.mean((y_true - y_pred) ** 2)
    rmse   = np.sqrt(mse)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return dict(mae=mae, mse=mse, rmse=rmse, r2=r2)


def standardize_features(train_graphs, val_graphs):
    node_sc = StandardScaler()
    edge_sc = StandardScaler()
    gas_sc  = StandardScaler()
    node_sc.fit(np.vstack([g.x.numpy() for g in train_graphs]))
    gas_sc .fit(np.array([g.gas_features.squeeze(0).numpy() for g in train_graphs]))
    all_edges = [g.edge_attr.numpy() for g in train_graphs if g.edge_attr.shape[0] > 0]
    if all_edges:
        edge_sc.fit(np.vstack(all_edges))

    def scale(graphs):
        out = []
        for g in graphs:
            gc = g.clone()
            gc.x = torch.tensor(node_sc.transform(g.x.numpy()), dtype=torch.float)
            gc.gas_features = torch.tensor(
                gas_sc.transform(g.gas_features.squeeze(0).numpy().reshape(1, -1)),
                dtype=torch.float)
            if g.edge_attr.shape[0] > 0:
                gc.edge_attr = torch.tensor(
                    edge_sc.transform(g.edge_attr.numpy()), dtype=torch.float)
            out.append(gc)
        return out

    return scale(train_graphs), scale(val_graphs)


def precompute_logo_splits(all_graphs):
    splits = {}
    for test_gas in GASES:
        train_g = [g for g in all_graphs if g.gas_name != test_gas]
        test_g  = [g for g in all_graphs if g.gas_name == test_gas]
        if not test_g:
            continue
        y_tr_raw = np.array([g.y.item() for g in train_g])
        y_te_raw = np.array([g.y.item() for g in test_g])
        y_sc     = StandardScaler()
        y_tr_sc  = y_sc.fit_transform(y_tr_raw.reshape(-1, 1)).flatten()
        y_te_sc  = y_sc.transform(y_te_raw.reshape(-1, 1)).flatten()

        def apply_y(graphs, y_vals):
            out = []
            for i, g in enumerate(graphs):
                gc   = g.clone()
                gc.y = torch.tensor([y_vals[i]], dtype=torch.float)
                out.append(gc)
            return out

        tr_sc, te_sc = standardize_features(
            apply_y(train_g, y_tr_sc),
            apply_y(test_g,  y_te_sc))

        splits[test_gas] = dict(
            train_graphs = tr_sc,
            test_graphs  = te_sc,
            y_scaler     = y_sc,
            n_train      = len(train_g),
            n_test       = len(test_g),
        )
    return splits

# ─────────────────────────────────────────────────────────────
# Single Member Training
# ─────────────────────────────────────────────────────────────
def train_single_member(train_graphs, seed: int) -> nn.Module:
    set_seeds(seed)
    loader = DataLoader(
        train_graphs, batch_size=BEST_PARAMS['batch_size'],
        shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model = GasConditionedMPNN(
        node_features = 7, edge_features = 7, gas_features = FEATURE_DIM,
        hidden_dim    = BEST_PARAMS['hidden_dim'],
        num_mp_layers = BEST_PARAMS['num_mp_layers'],
        fusion_dim    = BEST_PARAMS['fusion_dim'],
        l2_lambda     = BEST_PARAMS['l2_lambda'],
        dropout       = BEST_PARAMS['dropout'],
        pooling       = BEST_PARAMS['pooling'],
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=BEST_PARAMS['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINAL_EPOCHS)

    best_loss    = float('inf')
    patience_ctr = 0
    best_state   = None

    for epoch in range(FINAL_EPOCHS):
        loss = train_epoch(model, loader, criterion, optimizer, scheduler)
        if loss < best_loss:
            best_loss    = loss
            patience_ctr = 0
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
        if (epoch + 1) % 100 == 0:
            print(f'      seed={seed}  epoch={epoch+1:4d}  loss={loss:.4f}')
        if patience_ctr >= FINAL_PATIENCE:
            print(f'      seed={seed}  early stop @ epoch {epoch+1}')
            break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    return model

# ─────────────────────────────────────────────────────────────
# Ensemble Prediction
# ── y_pred_mean returned directly (not reconstructed from MSE)
# ─────────────────────────────────────────────────────────────
def ensemble_predict(models: list, loader, y_scaler):
    """
    Returns per-polymer arrays on original permeability scale:
        mu            — ensemble mean prediction
        entropy       — H = 0.5 * log(2πe σ²)  in nats
        epistemic_std — std across ensemble members
        y_true        — ground-truth values
    """
    all_preds           = []
    y_true_sc_collected = None

    for model in models:
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                out  = model(data)
                preds  .append(out.cpu().numpy())
                targets.append(data.y.cpu().numpy())
        all_preds.append(np.concatenate(preds))
        if y_true_sc_collected is None:
            y_true_sc_collected = np.concatenate(targets)

    all_preds  = np.stack(all_preds)                         # (N_ensemble, N_polymers)
    preds_orig = y_scaler.inverse_transform(all_preds.T).T   # (N_ensemble, N_polymers)
    y_true     = y_scaler.inverse_transform(
                     y_true_sc_collected.reshape(-1, 1)).flatten()

    mu            = preds_orig.mean(axis=0)
    var           = preds_orig.var(axis=0)
    var           = np.clip(var, 1e-12, None)
    epistemic_std = np.sqrt(var)
    entropy       = 0.5 * np.log(2 * math.pi * math.e * var)

    return mu, entropy, epistemic_std, y_true


def ensemble_test_metrics(models, test_loader, y_scaler, criterion):
    all_preds, y_true_list = [], []
    for model in models:
        model.eval()
        _, y_pred_sc, y_true_sc = evaluate(model, test_loader, criterion)
        all_preds  .append(y_pred_sc)
        y_true_list.append(y_true_sc)
    mean_pred_sc = np.stack(all_preds).mean(axis=0)
    y_pred = y_scaler.inverse_transform(mean_pred_sc.reshape(-1, 1)).flatten()
    y_true = y_scaler.inverse_transform(y_true_list[0].reshape(-1, 1)).flatten()
    return metrics(y_true, y_pred), y_true, y_pred

# ─────────────────────────────────────────────────────────────
# CSV Export
# ─────────────────────────────────────────────────────────────
def save_entropy_csv(logo_results: dict, all_graphs: list,
                     output_path: str = 'entropy_per_pair.csv') -> pd.DataFrame:
    """
    Write one row per (polymer, eval_gas, fold_held_out) triplet.

    Columns
    -------
    smiles          : polymer repeat-unit SMILES string
    polymer_idx     : position of polymer in the eval_gas subset of all_graphs
    eval_gas        : which gas was evaluated
    fold_held_out   : which gas was excluded from training in this fold
    is_ood          : True when eval_gas == fold_held_out
    y_true          : experimental log permeability (original scale)
    y_pred_mean     : ensemble mean prediction (original scale)
    epistemic_std   : std across ensemble members (original scale)
    entropy_nats    : H = 0.5 * log(2πe σ²)
    mse             : (y_pred_mean - y_true)²
    """
    rows = []

    for held_out_gas, result in logo_results.items():

        entropy_dict = result['entropy']         # {gas → (N_polymers,)}
        std_dict     = result['epistemic_std']
        mu_dict      = result['y_pred_mean']     # ← direct prediction, not from MSE
        ytrue_dict   = result['y_true_per_gas']

        for eval_gas in GASES:
            if eval_gas not in entropy_dict:
                continue

            eval_graphs = [g for g in all_graphs if g.gas_name == eval_gas]

            ent    = entropy_dict[eval_gas]
            std    = std_dict[eval_gas]
            mu     = mu_dict[eval_gas]
            y_true = ytrue_dict[eval_gas]
            mse    = (mu - y_true) ** 2

            for i, g in enumerate(eval_graphs):
                rows.append({
                    'smiles'        : g.smiles,
                    'polymer_idx'   : i,
                    'eval_gas'      : eval_gas,
                    'fold_held_out' : held_out_gas,
                    'is_ood'        : (eval_gas == held_out_gas),
                    'y_true'        : float(y_true[i]),
                    'y_pred_mean'   : float(mu[i]),
                    'epistemic_std' : float(std[i]),
                    'entropy_nats'  : float(ent[i]),
                    'mse'           : float(mse[i]),
                })

    df = pd.DataFrame(rows)
    df = df[['smiles', 'polymer_idx', 'eval_gas', 'fold_held_out', 'is_ood',
             'y_true', 'y_pred_mean', 'epistemic_std', 'entropy_nats', 'mse']]
    df.to_csv(output_path, index=False)

    print(f'\nSaved: {output_path}')
    print(f'  Total rows : {len(df):,}')
    print(f'  OOD rows (is_ood=True) : {df["is_ood"].sum():,}')
    print(f'\n  Preview:')
    print(df.head(6).to_string(index=False))
    return df

# ─────────────────────────────────────────────────────────────
# Calibration Check
# ─────────────────────────────────────────────────────────────
def spearman_calibration(entropy_dict, mse_dict):
    print(f"\n{'Gas':<6}  {'Spearman ρ':>12}  {'p-value':>12}  {'Calibrated?':>12}")
    print('-' * 50)
    results = {}
    for gas in GASES:
        if gas not in entropy_dict:
            continue
        rho, pval = spearmanr(entropy_dict[gas], mse_dict[gas])
        flag      = '✓' if rho > 0.4 and pval < 0.05 else '✗'
        print(f'{gas:<6}  {rho:>12.4f}  {pval:>12.4e}  {flag:>12}')
        results[gas] = dict(rho=rho, pval=pval)
    return results

# ─────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────
def plot_kde_entropy_mse(entropy_dict, mse_dict, title_suffix=''):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    panels = [
        (entropy_dict, 'Predictive Entropy (nats)', f'Entropy Distribution{title_suffix}'),
        (mse_dict,     'MSE (log-perm units²)',      f'MSE Distribution{title_suffix}'),
    ]
    for ax, (data_dict, xlabel, panel_title) in zip(axes, panels):
        for gas in GASES:
            if gas not in data_dict:
                continue
            vals = data_dict[gas]
            vals = vals[np.isfinite(vals)]
            kde  = gaussian_kde(vals, bw_method='scott')
            x    = np.linspace(vals.min(), vals.max(), 300)
            ax.plot(x, kde(x), color=GAS_COLORS[gas], lw=2, label=gas)
            ax.fill_between(x, kde(x), alpha=0.08, color=GAS_COLORS[gas])
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.set_title(panel_title, fontsize=13, fontweight='bold')
        ax.legend(title='Gas', fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    return fig


def plot_entropy_heatmap(entropy_matrix: pd.DataFrame, n_ensemble: int):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(entropy_matrix, annot=True, fmt='.3f', cmap='viridis',
                linewidths=0.5, ax=ax)
    ax.set_xlabel('Evaluated Gas',  fontsize=12)
    ax.set_ylabel('Held-Out Gas',   fontsize=12)
    ax.set_title(f'Mean Predictive Entropy — LOGO Folds  (ensemble n={n_ensemble})\n'
                 'Diagonal = held-out gas = OOD signal', fontsize=12)
    plt.tight_layout()
    return fig


def plot_std_per_gas(entropy_dict, test_gas, n_ensemble):
    data   = [entropy_dict[g] for g in GASES if g in entropy_dict]
    labels = [g for g in GASES if g in entropy_dict]
    colors = [GAS_COLORS[g] for g in labels]
    fig, ax = plt.subplots(figsize=(10, 5))
    parts   = ax.violinplot(data, showmedians=True, showextrema=True)
    for pc, col in zip(parts['bodies'], colors):
        pc.set_facecolor(col)
        pc.set_alpha(0.55)
    for part in ['cmedians', 'cmins', 'cmaxes', 'cbars']:
        parts[part].set_color('grey')
        parts[part].set_linewidth(1)
    if test_gas in labels:
        idx = labels.index(test_gas)
        ax.axvspan(idx + 0.6, idx + 1.4, color='gold', alpha=0.25,
                   label=f'held-out: {test_gas}')
        ax.legend(fontsize=10)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Predictive Entropy (nats)', fontsize=12)
    ax.set_title(f'Epistemic uncertainty per gas  '
                 f'[held-out: {test_gas}, ensemble n={n_ensemble}]',
                 fontsize=13, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    print(f'{"="*60}')
    print(f'GC-MPNN Deep Ensemble  (N_ENSEMBLE={N_ENSEMBLE})')
    print(f'{"="*60}')

    print('\nLoading data...')
    pol_sd     = pd.read_csv('Gas_permeability_solubility_diffusivity_wide.csv')
    all_graphs = build_dataset(pol_sd)
    print(f'  {len(all_graphs)} polymer-gas pairs loaded')

    print('Precomputing LOGO splits...')
    splits = precompute_logo_splits(all_graphs)

    entropy_matrix = {ho: {} for ho in GASES}
    logo_results   = {}

    for test_gas in GASES:
        if test_gas not in splits:
            continue

        print(f'\n{"="*60}')
        print(f'LOGO fold — held-out: {test_gas}')
        split = splits[test_gas]
        print(f'  Train: {split["n_train"]}  |  Test: {split["n_test"]}')

        # ── Train ensemble ────────────────────────────────
        print(f'\n  Training {N_ENSEMBLE} ensemble members...')
        ensemble = []
        for seed in range(N_ENSEMBLE):
            print(f'    [member {seed+1}/{N_ENSEMBLE}]')
            ensemble.append(train_single_member(split['train_graphs'], seed=seed))

        # ── Deterministic test metrics ────────────────────
        criterion   = nn.MSELoss()
        test_loader = DataLoader(
            split['test_graphs'], batch_size=BEST_PARAMS['batch_size'],
            shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

        m, y_true_test, y_pred_test = ensemble_test_metrics(
            ensemble, test_loader, split['y_scaler'], criterion)
        print(f'\n Test  R²={m["r2"]:.4f}  RMSE={m["rmse"]:.4f}  MAE={m["mae"]:.4f}')

        # ── Entropy for all 6 gases ───────────────────────
        print(f'  Computing ensemble entropy for all 6 gases...')

        fold_entropy = {}
        fold_mse     = {}
        fold_std     = {}
        fold_mu      = {}     # y_pred_mean per gas — stored for CSV
        fold_ytrue   = {}     # y_true per gas      — stored for CSV

        for eval_gas in GASES:
            eval_graphs = [g for g in all_graphs if g.gas_name == eval_gas]
            if not eval_graphs:
                continue
            _, eval_sc = standardize_features(split['train_graphs'], eval_graphs)
            ev_loader  = DataLoader(eval_sc, batch_size=BEST_PARAMS['batch_size'],
                                    shuffle=False, num_workers=NUM_WORKERS,
                                    pin_memory=PIN_MEMORY)
            mu, ent, std, yt = ensemble_predict(
                ensemble, ev_loader, split['y_scaler'])

            fold_entropy[eval_gas] = ent
            fold_mse[eval_gas]     = (mu - yt) ** 2
            fold_std[eval_gas]     = std
            fold_mu[eval_gas]      = mu      #  direct prediction
            fold_ytrue[eval_gas]   = yt      #  ground truth
            entropy_matrix[test_gas][eval_gas] = float(ent.mean())

        # ── Plots ─────────────────────────────────────────
        fig_kde = plot_kde_entropy_mse(
            fold_entropy, fold_mse,
            title_suffix=f'  [n={N_ENSEMBLE}, held-out {test_gas}]')
        fig_kde.savefig(f'kde_entropy_mse_ensemble_holdout_{test_gas}.png',
                        dpi=150, bbox_inches='tight')
        plt.close(fig_kde)

        fig_vio = plot_std_per_gas(fold_entropy, test_gas, N_ENSEMBLE)
        fig_vio.savefig(f'violin_entropy_ensemble_holdout_{test_gas}.png',
                        dpi=150, bbox_inches='tight')
        plt.close(fig_vio)
        print(f'  Saved plots for fold {test_gas}')

        # ── Spearman calibration ──────────────────────────
        print(f'\n  Spearman calibration — held-out {test_gas}:')
        spearman_results = spearman_calibration(fold_entropy, fold_mse)

        # ── Store for CSV ─────────────────────────────────
        logo_results[test_gas] = dict(
            test_metrics   = m,
            y_true         = y_true_test,
            y_pred         = y_pred_test,
            entropy        = fold_entropy,
            mse_vals       = fold_mse,
            epistemic_std  = fold_std,
            y_pred_mean    = fold_mu,      # ← new
            y_true_per_gas = fold_ytrue,   # ← new
            spearman       = spearman_results,
        )

    # ─────────────────────────────────────────────────────────
    # 6×6 Entropy Heatmap
    # ─────────────────────────────────────────────────────────
    ent_df = pd.DataFrame(entropy_matrix).T
    ent_df.index.name   = 'Held-Out Gas'
    ent_df.columns.name = 'Evaluated Gas'

    fig_hm = plot_entropy_heatmap(ent_df, N_ENSEMBLE)
    fig_hm.savefig('logo_entropy_heatmap_ensemble.png', dpi=150, bbox_inches='tight')
    plt.close(fig_hm)
    ent_df.to_csv('logo_entropy_matrix_ensemble.csv')
    print('\nSaved: logo_entropy_heatmap_ensemble.png')
    print('Saved: logo_entropy_matrix_ensemble.csv')

    # ─────────────────────────────────────────────────────────
    # Per-Pair CSV
    # ─────────────────────────────────────────────────────────
    df_pairs = save_entropy_csv(logo_results, all_graphs,
                                output_path='entropy_per_pair.csv')

    # ── OOD sanity check directly from the CSV ────────────
    print('\n  OOD vs in-distribution entropy check:')
    print(f'  {"Gas":<6}  {"OOD H":>10}  {"InDist H":>10}  {"Δ":>8}  {"Signal?":>8}')
    print('  ' + '-' * 46)
    for gas in GASES:
        ood = df_pairs[(df_pairs['eval_gas'] == gas) &
                       (df_pairs['is_ood'])]['entropy_nats'].mean()
        ind = df_pairs[(df_pairs['eval_gas'] == gas) &
                       (~df_pairs['is_ood'])]['entropy_nats'].mean()
        flag = '✓' if ood > ind else '✗'
        print(f'  {gas:<6}  {ood:>10.4f}  {ind:>10.4f}  {ood-ind:>8.4f}  {flag:>8}')

    # ─────────────────────────────────────────────────────────
    # Final Summary
    # ─────────────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'LOGO SUMMARY  (ensemble n={N_ENSEMBLE})')
    print(f'{"="*60}')
    r2_list = []
    for gas in GASES:
        if gas not in logo_results:
            continue
        m        = logo_results[gas]['test_metrics']
        diag_ent = entropy_matrix[gas].get(gas, float('nan'))
        rho      = logo_results[gas]['spearman'].get(gas, {}).get('rho', float('nan'))
        r2_list.append(m['r2'])
        print(f'  {gas:<5}  R²={m["r2"]:6.4f}  RMSE={m["rmse"]:.4f}  '
              f'OOD_H={diag_ent:.4f}  Spearman_ρ={rho:.4f}')

    print(f'\n  Mean R² : {np.mean(r2_list):.4f} ± {np.std(r2_list):.4f}')

    torch.save(logo_results, 'logo_ensemble_results.pt')
    print('Saved: logo_ensemble_results.pt')


if __name__ == '__main__':
    main()
