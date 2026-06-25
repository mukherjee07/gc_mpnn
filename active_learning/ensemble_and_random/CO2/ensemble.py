#!/usr/bin/env python
# coding: utf-8
"""
GC-MPNN - Deep Ensemble Inference (Bootstrap + Seed Diversity)
===============================================================

Loads the best hyperparameters from a JSON file produced by the nested
HPO script (e.g. Kinetic_best_params.json) and trains N_ENSEMBLE members
with two independent sources of diversity:

  1. Bootstrap resampling - each member sees a different random draw
     (with replacement) of the training pool, same size as the pool.
  2. Random seed         - each member uses a unique seed for weight
     initialisation and training-time stochasticity (dropout, etc.).

After all members are trained, per-sample statistics are computed:

  y_mean  = mean  of member predictions  (point estimate)
  y_std   = std   of member predictions  (epistemic + data uncertainty)

Outputs
-------
  {experiment}_ensemble_test_{TEST_GAS}_predictions.csv
      One row per test polymer:
        y_true, y_mean, y_std, y_pred_m0 ... y_pred_m{N-1},
        ci_lower_95, ci_upper_95, within_ci

  {experiment}_ensemble_train_pool_predictions.csv
      Same columns for the full (unbootstrapped) pool - for parity plots.

  {experiment}_ensemble_summary.json
      Aggregate metrics + calibration statistics.

  {experiment}_ensemble_member_metrics.csv
      Per-member R2, RMSE, MAE for diagnostic use.
"""

import pandas as pd
import numpy as np
import os
import json
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
import warnings

warnings.filterwarnings('ignore')

# KEY SETTINGS  (edit these)
EXPERIMENT_NAME    = 'Kinetic'         # must match the JSON prefix
PARAMS_JSON        = f'{EXPERIMENT_NAME}_best_params.json'
TEST_GAS           = 'CO2'             # locked-away gas - prediction target
N_ENSEMBLE         = 20                # number of ensemble members
FINAL_EPOCHS       = 500
FINAL_PATIENCE     = 100
BASE_SEED          = 1000              # member i uses seed BASE_SEED + i
DATA_CSV           = '../../../data/Gas_permeability_solubility_diffusivity_wide.csv'

ALL_GASES     = ['He', 'H2', 'N2', 'O2', 'CH4', 'CO2']
TRAINING_POOL = [g for g in ALL_GASES if g != TEST_GAS]

# DEVICE
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


def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# LOAD HYPERPARAMETERS
def load_hyperparameters(json_path: str) -> dict:
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Hyperparameter file not found: {json_path}\n"
            f"Expected output from nested HPO script.")
    with open(json_path, 'r') as f:
        data = json.load(f)
    hp = data['params']
    # Ensure integer types for architectural params
    hp['hidden_dim']    = int(hp['hidden_dim'])
    hp['num_mp_layers'] = int(hp['num_mp_layers'])
    hp['fusion_dim']    = int(hp['fusion_dim'])
    hp['batch_size']    = int(hp['batch_size'])
    print(f"\nLoaded hyperparameters from: {json_path}")
    print(f"  HPO best MSE (scaled): {data.get('best_hpo_mse', 'N/A')}")
    for k, v in hp.items():
        print(f"  {k:<20} : {v}")
    return hp

# GAS PROPERTIES & FEATURE FUNCTIONS
GAS_PROPERTIES = {
    'He':  {'sigma': 2.551, 'epsilon':  10.2,  'omega': -0.383, 'Tc':   5.2, 'Pc':  2.28, 'd': 2.6,  'Vd':  2.67, 'q_pos': 0.0,   'q_neg': 0.0  },
    'H2':  {'sigma': 2.827, 'epsilon':  59.7,  'omega': -0.265, 'Tc':  33.2, 'Pc': 13.00, 'd': 2.89, 'Vd':  6.12, 'q_pos': 0.0,   'q_neg': 0.0  },
    'N2':  {'sigma': 3.798, 'epsilon':  71.4,  'omega':  0.037, 'Tc': 126.2, 'Pc': 63.14, 'd': 3.64, 'Vd': 18.5,  'q_pos': 0.482, 'q_neg': 0.482},
    'O2':  {'sigma': 3.467, 'epsilon': 106.7,  'omega':  0.022, 'Tc': 154.6, 'Pc': 50.43, 'd': 3.46, 'Vd': 16.3,  'q_pos': 0.226, 'q_neg': 0.113},
    'CH4': {'sigma': 3.758, 'epsilon': 148.6,  'omega':  0.011, 'Tc': 190.6, 'Pc': 46.1,  'd': 3.8,  'Vd': 24.42, 'q_pos': 0.0,   'q_neg': 0.0  },
    'CO2': {'sigma': 3.941, 'epsilon': 195.2,  'omega':  0.253, 'Tc': 304.1, 'Pc': 73.80, 'd': 3.3,  'Vd': 26.9,  'q_pos': 0.70,  'q_neg': 0.35 },
}

EXPERIMENT_CONFIGS = {
    'Thermodynamic':      {'feature_func': lambda g: np.array([GAS_PROPERTIES[g]['sigma'], GAS_PROPERTIES[g]['epsilon'], GAS_PROPERTIES[g]['omega'], GAS_PROPERTIES[g]['Tc'], GAS_PROPERTIES[g]['Pc']], dtype=np.float32), 'feature_dim': 5},
    'Kinetic':            {'feature_func': lambda g: np.array([GAS_PROPERTIES[g]['d'], GAS_PROPERTIES[g]['Vd']], dtype=np.float32), 'feature_dim': 2},
    'Electrostatics':     {'feature_func': lambda g: np.array([GAS_PROPERTIES[g]['q_pos'], GAS_PROPERTIES[g]['q_neg']], dtype=np.float32), 'feature_dim': 2},
    'Thermo_and_Kinetic': {'feature_func': lambda g: np.array([GAS_PROPERTIES[g]['sigma'], GAS_PROPERTIES[g]['epsilon'], GAS_PROPERTIES[g]['omega'], GAS_PROPERTIES[g]['Tc'], GAS_PROPERTIES[g]['Pc'], GAS_PROPERTIES[g]['d'], GAS_PROPERTIES[g]['Vd']], dtype=np.float32), 'feature_dim': 7},
    'Full':               {'feature_func': lambda g: np.array([GAS_PROPERTIES[g]['sigma'], GAS_PROPERTIES[g]['epsilon'], GAS_PROPERTIES[g]['omega'], GAS_PROPERTIES[g]['Tc'], GAS_PROPERTIES[g]['Pc'], GAS_PROPERTIES[g]['d'], GAS_PROPERTIES[g]['Vd'], GAS_PROPERTIES[g]['q_pos'], GAS_PROPERTIES[g]['q_neg']], dtype=np.float32), 'feature_dim': 9},
    'OneHot':             {'feature_func': lambda g: np.eye(6, dtype=np.float32)[{'He':0,'H2':1,'N2':2,'O2':3,'CH4':4,'CO2':5}[g]], 'feature_dim': 6},
}

# DATA LOADING & GRAPH CONSTRUCTION
def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    p_exp_map = {
        'CH4': df['p_exp_CH4'], 'CO2': df['p_exp_CO2'],
        'H2':  df['p_exp_H2'],  'N2':  df['p_exp_N2'],
        'O2':  df['p_exp_O2'],  'He':  df['p_exp_He'],
    }
    return df['smiles_string'], p_exp_map


def create_dataset(smiles, p_exp_map, experiment_name, gas_subset):
    feat_fn = EXPERIMENT_CONFIGS[experiment_name]['feature_func']
    records = []
    for idx in range(len(smiles)):
        smi = smiles.iloc[idx]
        for gas in gas_subset:
            perm = p_exp_map[gas].iloc[idx]
            if not np.isnan(perm):
                records.append({
                    'smiles':      smi,
                    'gas':         gas,
                    'permeability': perm,
                    'gas_features': feat_fn(gas),
                })
    return records


def smiles_to_graph(smi: str):
    mol = Chem.MolFromSmiles(smi)
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


def build_pyg_dataset(records):
    dataset = []
    for rec in records:
        g = smiles_to_graph(rec['smiles'])
        if g is None:
            continue
        g.gas_features = torch.tensor(
            rec['gas_features'], dtype=torch.float).unsqueeze(0)
        g.y        = torch.tensor([rec['permeability']], dtype=torch.float)
        g.gas_name = rec['gas']
        g.smiles    = rec['smiles']
        dataset.append(g)
    return dataset

# FEATURE & TARGET STANDARDISATION
def standardize_features(train_graphs, eval_graphs):
    node_sc = StandardScaler()
    edge_sc = StandardScaler()
    gas_sc  = StandardScaler()

    node_sc.fit(np.vstack([g.x.numpy() for g in train_graphs]))
    gas_sc.fit(np.array([g.gas_features.squeeze(0).numpy() for g in train_graphs]))
    edge_data = [g.edge_attr.numpy() for g in train_graphs if g.edge_attr.shape[0] > 0]
    if edge_data:
        edge_sc.fit(np.vstack(edge_data))

    def scale(graphs):
        out = []
        for g in graphs:
            gc = g.clone()
            gc.x = torch.tensor(node_sc.transform(g.x.numpy()), dtype=torch.float)
            gc.gas_features = torch.tensor(
                gas_sc.transform(g.gas_features.squeeze(0).numpy().reshape(1, -1)),
                dtype=torch.float)
            if g.edge_attr.shape[0] > 0 and edge_data:
                gc.edge_attr = torch.tensor(
                    edge_sc.transform(g.edge_attr.numpy()), dtype=torch.float)
            out.append(gc)
        return out

    return scale(train_graphs), scale(eval_graphs)


def scale_targets(train_graphs, eval_graphs):
    y_train = np.array([g.y.item() for g in train_graphs])
    y_eval  = np.array([g.y.item() for g in eval_graphs])
    y_sc    = StandardScaler()
    y_tr_sc = y_sc.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_ev_sc = y_sc.transform(y_eval.reshape(-1, 1)).flatten()

    def apply(graphs, y_vals):
        return [
            _set_y(g.clone(), y_vals[i]) for i, g in enumerate(graphs)
        ]

    return apply(train_graphs, y_tr_sc), apply(eval_graphs, y_ev_sc), y_sc


def _set_y(g, val):
    g.y = torch.tensor([val], dtype=torch.float)
    return g

# MODEL
class MPNNLayer(MessagePassing):
    def __init__(self, in_ch, out_ch, edge_dim):
        super().__init__(aggr='add')
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_ch * 2 + edge_dim, out_ch), nn.ReLU(),
            nn.Linear(out_ch, out_ch))
        self.upd_mlp = nn.Sequential(
            nn.Linear(in_ch + out_ch, out_ch), nn.ReLU(),
            nn.Linear(out_ch, out_ch))

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        return self.msg_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))

    def update(self, aggr_out, x):
        return self.upd_mlp(torch.cat([x, aggr_out], dim=-1))


class GasConditionedMPNN(nn.Module):
    def __init__(self, node_features=7, edge_features=7, gas_features=7,
                 hidden_dim=64, num_mp_layers=3, fusion_dim=128,
                 l2_lambda=0.001, dropout=0.3, pooling='mean'):
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
            gate = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, 1))
            self.attention_pool = GlobalAttention(gate)

        self.gas_encoder = nn.Sequential(
            nn.Linear(gas_features, 64),  nn.ReLU(), nn.LayerNorm(64),
            nn.Dropout(dropout),
            nn.Linear(64, fusion_dim),    nn.ReLU(), nn.LayerNorm(fusion_dim))

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + fusion_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),                     nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),                      nn.ReLU(),
            nn.Linear(64,  1))

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch)
        gas_features = data.gas_features
        batch_size   = batch.max().item() + 1

        if gas_features.shape != (batch_size, self.expected_gas_dim):
            raise ValueError(
                f"gas_features shape {gas_features.shape} vs "
                f"[{batch_size}, {self.expected_gas_dim}]")

        x = F.relu(self.node_embedding(x))
        for mp, bn in zip(self.mp_layers, self.batch_norms):
            x = x + self.dropout(F.relu(bn(mp(x, edge_index, edge_attr))))

        if self.pooling_type == 'mean':
            p_emb = global_mean_pool(x, batch)
        elif self.pooling_type == 'sum':
            p_emb = global_add_pool(x, batch)
        elif self.pooling_type == 'attention':
            p_emb = self.attention_pool(x, batch)
        else:
            p_emb = global_mean_pool(x, batch)

        g_emb    = self.gas_encoder(gas_features)
        combined = torch.cat([p_emb, g_emb], dim=-1)
        return self.fusion(combined).squeeze(-1)

    def l2_regularization(self):
        l2 = torch.tensor(0., device=next(self.parameters()).device)
        for p in self.parameters():
            l2 += torch.norm(p, 2)
        return self.l2_lambda * l2

# TRAINING & EVALUATION
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
    if scheduler is not None:
        scheduler.step()
    return total / len(loader)


def evaluate(model, loader):
    """Returns predictions in current (scaled) space."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            preds  .extend(model(data).cpu().numpy().tolist())
            targets.extend(data.y.cpu().numpy().tolist())
    return np.array(preds), np.array(targets)


def bootstrap_sample(graphs, seed: int):
    """Draw len(graphs) samples with replacement."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(graphs), size=len(graphs))
    return [graphs[i] for i in idx]


def train_single_member(member_id: int, train_pool_raw, test_raw,
                        hp: dict, feature_dim: int):
    """
    Train one ensemble member:
      - bootstrap resample the training pool
      - set a unique seed for weight init + stochastic ops
      - return test predictions (original log10-Barrer scale)
             and train-pool predictions (for in-distribution diagnostics)
    """
    seed = BASE_SEED + member_id
    set_seeds(seed)

    # Bootstrap resample
    boot_train_raw = bootstrap_sample(train_pool_raw, seed=seed)

    # Scale targets using bootstrap train set
    train_y, test_y, y_sc = scale_targets(boot_train_raw, test_raw)

    # Standardise features using bootstrap train set
    train_sc, test_sc = standardize_features(train_y, test_y)

    # Also scale the FULL pool for in-distribution parity plots
    # (fit scaler on bootstrap sample, apply to full pool)
    _, full_pool_sc = standardize_features(train_y, scale_targets(
        boot_train_raw, train_pool_raw)[1])
    full_pool_sc_graphs = full_pool_sc   # already scaled

    train_loader      = DataLoader(train_sc,       batch_size=hp['batch_size'],
                                   shuffle=True,  num_workers=NUM_WORKERS,
                                   pin_memory=PIN_MEMORY)
    test_loader       = DataLoader(test_sc,        batch_size=hp['batch_size'],
                                   shuffle=False, num_workers=NUM_WORKERS,
                                   pin_memory=PIN_MEMORY)
    full_pool_loader  = DataLoader(full_pool_sc_graphs,
                                   batch_size=hp['batch_size'],
                                   shuffle=False, num_workers=NUM_WORKERS,
                                   pin_memory=PIN_MEMORY)

    # Model (seed already set above -> deterministic init)
    model = GasConditionedMPNN(
        node_features=7, edge_features=7, gas_features=feature_dim,
        hidden_dim    = hp['hidden_dim'],
        num_mp_layers = hp['num_mp_layers'],
        fusion_dim    = hp['fusion_dim'],
        l2_lambda     = hp['l2_lambda'],
        dropout       = hp['dropout'],
        pooling       = hp['pooling'],
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=hp['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINAL_EPOCHS)

    best_tr_loss = float('inf')
    best_state   = None
    no_improve   = 0

    for epoch in range(FINAL_EPOCHS):
        tr_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler)
        if tr_loss < best_tr_loss:
            best_tr_loss = tr_loss
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1
        if no_improve >= FINAL_PATIENCE:
            break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Predictions in original log10-Barrer scale
    tp, tt   = evaluate(model, test_loader)
    y_pred_test = y_sc.inverse_transform(tp.reshape(-1, 1)).flatten()
    y_true_test = y_sc.inverse_transform(tt.reshape(-1, 1)).flatten()

    # For full pool: we need a separate target scaler fitted on the full pool
    _, full_pool_raw_y, y_sc_pool = scale_targets(boot_train_raw, train_pool_raw)
    full_pool_sc2, _ = standardize_features(train_y, full_pool_raw_y)
    full_pool_loader2 = DataLoader(full_pool_sc2, batch_size=hp['batch_size'],
                                   shuffle=False, num_workers=NUM_WORKERS,
                                   pin_memory=PIN_MEMORY)
    fp, ft = evaluate(model, full_pool_loader2)
    y_pred_pool = y_sc_pool.inverse_transform(fp.reshape(-1, 1)).flatten()
    y_true_pool = y_sc_pool.inverse_transform(ft.reshape(-1, 1)).flatten()

    return y_pred_test, y_true_test, y_pred_pool, y_true_pool

# METRICS & CALIBRATION
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return {
        'r2':   float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0,
        'rmse': float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        'mae':  float(np.mean(np.abs(y_pred - y_true))),
    }


def calibration_stats(y_true: np.ndarray,
                      y_mean: np.ndarray,
                      y_std:  np.ndarray) -> dict:
    """
    95 % prediction interval: y_mean +/- 1.96 * y_std
    PICP (Prediction Interval Coverage Probability) - fraction of true
    values that fall within the interval. Well-calibrated -> ~0.95.
    MPIW (Mean Prediction Interval Width) - smaller is sharper.
    """
    ci_lower = y_mean - 1.96 * y_std
    ci_upper = y_mean + 1.96 * y_std
    within   = (y_true >= ci_lower) & (y_true <= ci_upper)
    picp     = float(np.mean(within))
    mpiw     = float(np.mean(ci_upper - ci_lower))
    # Normalised MPIW (relative to target range)
    y_range  = float(np.max(y_true) - np.min(y_true))
    nmpiw    = mpiw / y_range if y_range > 0 else np.nan
    return {
        'picp_95':  picp,
        'mpiw_95':  mpiw,
        'nmpiw_95': nmpiw,
        'ci_sharpness_note': (
            'Under-dispersed (overconfident)' if picp < 0.90
            else 'Over-dispersed (conservative)'  if picp > 0.99
            else 'Well-calibrated'),
    }

# MAIN
def main():
    print('\n' + '='*80)
    print('GC-MPNN  DEEP ENSEMBLE  -  Bootstrap + Seed Diversity')
    print(f'Experiment      : {EXPERIMENT_NAME}')
    print(f'Test gas        : {TEST_GAS}')
    print(f'Training pool   : {TRAINING_POOL}')
    print(f'Ensemble size   : {N_ENSEMBLE}')
    print(f'Epochs / Patience: {FINAL_EPOCHS} / {FINAL_PATIENCE}')
    print('='*80)

    # Load hyperparameters
    hp = load_hyperparameters(PARAMS_JSON)
    feature_dim = EXPERIMENT_CONFIGS[EXPERIMENT_NAME]['feature_dim']
    print(f'\nGas feature dimension: {feature_dim}')

    # Build datasets
    print('\nBuilding datasets...')
    smiles, p_exp_map = load_data(DATA_CSV)

    pool_records = create_dataset(smiles, p_exp_map, EXPERIMENT_NAME, TRAINING_POOL)
    test_records = create_dataset(smiles, p_exp_map, EXPERIMENT_NAME, [TEST_GAS])

    pool_graphs = build_pyg_dataset(pool_records)
    test_graphs = build_pyg_dataset(test_records)

    print(f'  Pool ({"+".join(TRAINING_POOL)}): {len(pool_graphs)} graphs')
    print(f'  Test ({TEST_GAS})            : {len(test_graphs)} graphs')

    # Run ensemble
    # Collect per-member predictions as arrays (n_test,) and (n_pool,)
    all_test_preds = []   # shape: (N_ENSEMBLE, n_test)
    all_pool_preds = []   # shape: (N_ENSEMBLE, n_pool)
    member_metrics = []

    for m in range(N_ENSEMBLE):
        print(f'\n Member {m+1:02d}/{N_ENSEMBLE}  '
              f'(seed={BASE_SEED+m}, bootstrap resample #{m+1}) ')

        y_pred_test, y_true_test, y_pred_pool, y_true_pool = train_single_member(
            member_id      = m,
            train_pool_raw = pool_graphs,
            test_raw       = test_graphs,
            hp             = hp,
            feature_dim    = feature_dim,
        )

        all_test_preds.append(y_pred_test)
        all_pool_preds.append(y_pred_pool)

        m_met = compute_metrics(y_true_test, y_pred_test)
        member_metrics.append({'member': m, 'seed': BASE_SEED + m, **m_met})
        print(f'   Test R2={m_met["r2"]:.4f}  '
              f'RMSE={m_met["rmse"]:.4f}  MAE={m_met["mae"]:.4f}')

    # Aggregate
    all_test_preds = np.array(all_test_preds)   # (N_ENSEMBLE, n_test)
    all_pool_preds = np.array(all_pool_preds)   # (N_ENSEMBLE, n_pool)

    y_mean_test = all_test_preds.mean(axis=0)
    y_std_test  = all_test_preds.std(axis=0, ddof=1)

    y_mean_pool = all_pool_preds.mean(axis=0)
    y_std_pool  = all_pool_preds.std(axis=0, ddof=1)

    # Metrics on ensemble mean
    print('\n' + '='*80)
    print('ENSEMBLE RESULTS')
    print('='*80)

    ens_test_met  = compute_metrics(y_true_test, y_mean_test)
    ens_pool_met  = compute_metrics(y_true_pool, y_mean_pool)
    ens_calib     = calibration_stats(y_true_test, y_mean_test, y_std_test)

    print(f'\nTest  ({TEST_GAS}) - Ensemble mean predictions:')
    print(f'  R2   = {ens_test_met["r2"]:.4f}')
    print(f'  RMSE = {ens_test_met["rmse"]:.4f}  (log10 Barrer)')
    print(f'  MAE  = {ens_test_met["mae"]:.4f}  (log10 Barrer)')
    print(f'\nCalibration (95% PI):')
    print(f'  PICP  = {ens_calib["picp_95"]:.4f}  '
          f'(target ~ 0.95 for well-calibrated)')
    print(f'  MPIW  = {ens_calib["mpiw_95"]:.4f}  (log10 Barrer)')
    print(f'  NMPIW = {ens_calib["nmpiw_95"]:.4f}')
    print(f'  Note  : {ens_calib["ci_sharpness_note"]}')

    print(f'\nTrain pool - Ensemble mean predictions:')
    print(f'  R2   = {ens_pool_met["r2"]:.4f}')
    print(f'  RMSE = {ens_pool_met["rmse"]:.4f}')

    # Save CSVs
    # Test predictions CSV
    ci_lower = y_mean_test - 1.96 * y_std_test
    ci_upper = y_mean_test + 1.96 * y_std_test
    within   = (y_true_test >= ci_lower) & (y_true_test <= ci_upper)

    test_smiles = [g.smiles for g in test_graphs]

    test_df = pd.DataFrame({
        'smiles':     test_smiles,
        'y_true':     y_true_test,
        'y_mean':     y_mean_test,
        'y_std':      y_std_test,
        'ci_lower_95': ci_lower,
        'ci_upper_95': ci_upper,
        'within_ci':  within.astype(int),
        'residual':   y_mean_test - y_true_test,
        'gas':        TEST_GAS,
        'experiment': EXPERIMENT_NAME,
    })
    # Append per-member columns for traceability
    for m in range(N_ENSEMBLE):
        test_df[f'y_pred_m{m:02d}'] = all_test_preds[m]

    csv_test = f'{EXPERIMENT_NAME}_ensemble_test_{TEST_GAS}_predictions.csv'
    test_df.to_csv(csv_test, index=False)
    print(f'\nSaved: {csv_test}')

    # Pool predictions CSV
    pool_smiles = [g.smiles for g in pool_graphs]

    pool_df = pd.DataFrame({
        'smiles':     pool_smiles,
        'y_true':     y_true_pool,
        'y_mean':     y_mean_pool,
        'y_std':      y_std_pool,
        'ci_lower_95': y_mean_pool - 1.96 * y_std_pool,
        'ci_upper_95': y_mean_pool + 1.96 * y_std_pool,
        'residual':   y_mean_pool - y_true_pool,
        'split':      'train_pool',
        'experiment': EXPERIMENT_NAME,
    })
    for m in range(N_ENSEMBLE):
        pool_df[f'y_pred_m{m:02d}'] = all_pool_preds[m]

    csv_pool = f'{EXPERIMENT_NAME}_ensemble_train_pool_predictions.csv'
    pool_df.to_csv(csv_pool, index=False)
    print(f'Saved: {csv_pool}')

    # Member-level metrics CSV
    mem_df = pd.DataFrame(member_metrics)
    csv_mem = f'{EXPERIMENT_NAME}_ensemble_member_metrics.csv'
    mem_df.to_csv(csv_mem, index=False)
    print(f'Saved: {csv_mem}')

    # Summary JSON
    summary = {
        'experiment':    EXPERIMENT_NAME,
        'test_gas':      TEST_GAS,
        'n_ensemble':    N_ENSEMBLE,
        'base_seed':     BASE_SEED,
        'diversity':     'bootstrap_resampling + unique_seed_per_member',
        'hyperparameters': hp,
        'ensemble_test_metrics':  ens_test_met,
        'ensemble_pool_metrics':  ens_pool_met,
        'calibration_95pct_PI':   ens_calib,
        'member_test_r2': {
            'mean': float(np.mean([m['r2']   for m in member_metrics])),
            'std':  float(np.std( [m['r2']   for m in member_metrics])),
            'min':  float(np.min( [m['r2']   for m in member_metrics])),
            'max':  float(np.max( [m['r2']   for m in member_metrics])),
        },
    }
    json_out = f'{EXPERIMENT_NAME}_ensemble_summary.json'
    with open(json_out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved: {json_out}')

    print('\n' + '='*80)
    print('COMPLETE')
    print('='*80)


if __name__ == '__main__':
    main()
