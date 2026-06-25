#!/usr/bin/env python
# coding: utf-8
"""
GC-MPNN - Active Learning via Ensemble Uncertainty
====================================================

Workflow
--------
1. Load the ensemble predictions CSV for the target gas (e.g. CO2).
   This CSV contains y_true and y_std for every test polymer.

2. Select N_ACTIVE samples from the test set using one of two strategies:
     'uncertainty' - top-N by y_std  (highest uncertainty first)
     'random'      - uniformly random baseline (fixed seed for fairness)

3. Match selected samples back to Gas_permeability_solubility_diffusivity_wide.csv
   via exact float match on y_true -> p_exp_{TEST_GAS} to recover SMILES.

4. Augment the training pool:
     new_train = original pool (all gases except TEST_GAS)
               + N_ACTIVE selected TEST_GAS samples

5. Train a SINGLE model (no ensemble) with the optimised hyperparameters
   loaded from {EXPERIMENT_NAME}_best_params.json.

6. Evaluate on the REMAINING test samples (TEST_GAS total - N_ACTIVE).

7. Save predictions and a summary CSV so you can sweep N_ACTIVE externally
   and plot R2 / RMSE vs N_ACTIVE for both strategies.

Key settings
------------
  EXPERIMENT_NAME   : must match the JSON and CSV prefix
  TEST_GAS          : gas whose samples are being queried
  N_ACTIVE          : how many test samples to move into training
  SELECTION_STRATEGY: 'uncertainty' or 'random'
  RANDOM_SEED       : seed for the random baseline (kept fixed for fair comparison)

Outputs
-------
  {experiment}_AL_{strategy}_N{N_active}_test_{gas}_predictions.csv
  {experiment}_AL_{strategy}_N{N_active}_summary.json
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
EXPERIMENT_NAME    = 'Kinetic'
TEST_GAS           = 'CO2'
N_ACTIVE           = 10             # number of test samples to move into training
SELECTION_STRATEGY = 'uncertainty'  # 'uncertainty' or 'random'
RANDOM_SEED        = 42             # used only when SELECTION_STRATEGY='random'

ENSEMBLE_CSV  = f'{EXPERIMENT_NAME}_ensemble_test_{TEST_GAS}_predictions.csv'
PARAMS_JSON   = f'{EXPERIMENT_NAME}_best_params.json'
DATA_CSV      = '../../../data/Gas_permeability_solubility_diffusivity_wide.csv'

FINAL_EPOCHS  = 500
FINAL_PATIENCE = 100

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

set_seeds(RANDOM_SEED)

# LOAD HYPERPARAMETERS
def load_hyperparameters(json_path: str) -> dict:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f'Hyperparameter file not found: {json_path}')
    with open(json_path, 'r') as f:
        data = json.load(f)
    hp = data['params']
    hp['hidden_dim']    = int(hp['hidden_dim'])
    hp['num_mp_layers'] = int(hp['num_mp_layers'])
    hp['fusion_dim']    = int(hp['fusion_dim'])
    hp['batch_size']    = int(hp['batch_size'])
    print(f'\nLoaded hyperparameters from: {json_path}')
    for k, v in hp.items():
        print(f'  {k:<20} : {v}')
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
    'Thermodynamic':      {'feature_func': lambda g: np.array([GAS_PROPERTIES[g]['sigma'], GAS_PROPERTIES[g]['epsilon'], GAS_PROPERTIES[g]['omega'], GAS_PROPERTIES[g]['Tc'], GAS_PROPERTIES[g]['Pc']], dtype=np.float32),                                                                                                                                                                   'feature_dim': 5},
    'Kinetic':            {'feature_func': lambda g: np.array([GAS_PROPERTIES[g]['d'], GAS_PROPERTIES[g]['Vd']], dtype=np.float32),                                                                                                                                                                                                                                                          'feature_dim': 2},
    'Electrostatics':     {'feature_func': lambda g: np.array([GAS_PROPERTIES[g]['q_pos'], GAS_PROPERTIES[g]['q_neg']], dtype=np.float32),                                                                                                                                                                                                                                                   'feature_dim': 2},
    'Thermo_and_Kinetic': {'feature_func': lambda g: np.array([GAS_PROPERTIES[g]['sigma'], GAS_PROPERTIES[g]['epsilon'], GAS_PROPERTIES[g]['omega'], GAS_PROPERTIES[g]['Tc'], GAS_PROPERTIES[g]['Pc'], GAS_PROPERTIES[g]['d'], GAS_PROPERTIES[g]['Vd']], dtype=np.float32),                                                                                                                 'feature_dim': 7},
    'Full':               {'feature_func': lambda g: np.array([GAS_PROPERTIES[g]['sigma'], GAS_PROPERTIES[g]['epsilon'], GAS_PROPERTIES[g]['omega'], GAS_PROPERTIES[g]['Tc'], GAS_PROPERTIES[g]['Pc'], GAS_PROPERTIES[g]['d'], GAS_PROPERTIES[g]['Vd'], GAS_PROPERTIES[g]['q_pos'], GAS_PROPERTIES[g]['q_neg']], dtype=np.float32),                                                         'feature_dim': 9},
    'OneHot':             {'feature_func': lambda g: np.eye(6, dtype=np.float32)[{'He':0,'H2':1,'N2':2,'O2':3,'CH4':4,'CO2':5}[g]],                                                                                                                                                                                                                                                         'feature_dim': 6},
}

# DATA LOADING & GRAPH CONSTRUCTION
def load_wide_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    p_exp_map = {
        'CH4': df['p_exp_CH4'], 'CO2': df['p_exp_CO2'],
        'H2':  df['p_exp_H2'],  'N2':  df['p_exp_N2'],
        'O2':  df['p_exp_O2'],  'He':  df['p_exp_He'],
    }
    return df, df['smiles_string'], p_exp_map


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


def record_to_graph(smi: str, gas: str, perm: float, experiment_name: str):
    """Build a single PyG graph for one (polymer, gas) record."""
    g = smiles_to_graph(smi)
    if g is None:
        return None
    feat_fn = EXPERIMENT_CONFIGS[experiment_name]['feature_func']
    g.gas_features = torch.tensor(feat_fn(gas), dtype=torch.float).unsqueeze(0)
    g.y            = torch.tensor([perm], dtype=torch.float)
    g.gas_name     = gas
    g.smiles       = smi
    return g


def build_pool_graphs(df, smiles_col, p_exp_map, experiment_name, gas_subset):
    """Build PyG graphs for all (polymer, gas) pairs in gas_subset."""
    graphs = []
    for idx in range(len(smiles_col)):
        smi = smiles_col.iloc[idx]
        for gas in gas_subset:
            perm = p_exp_map[gas].iloc[idx]
            if not np.isnan(perm):
                g = record_to_graph(smi, gas, perm, experiment_name)
                if g is not None:
                    graphs.append(g)
    return graphs

# ACTIVE SAMPLE SELECTION
def select_active_samples(ensemble_csv_path: str,
                           n_active: int,
                           strategy: str,
                           random_seed: int = 42):
    """
    Load the ensemble predictions CSV and select N_ACTIVE samples.

    Returns
    -------
    selected_df  : DataFrame of the N_ACTIVE chosen rows (with y_true, y_std)
    remaining_df : DataFrame of the un-chosen rows (new test set)
    """
    df = pd.read_csv(ensemble_csv_path)
    n_total = len(df)

    if n_active >= n_total:
        raise ValueError(
            f'N_ACTIVE={n_active} >= total test samples={n_total}. '
            f'Must leave at least one sample for evaluation.')

    if strategy == 'uncertainty':
        # Sort descending by y_std; top N_ACTIVE are most uncertain
        sorted_df   = df.sort_values('y_std', ascending=False).reset_index(drop=True)
        selected_df = sorted_df.iloc[:n_active].copy()
        remaining_df = sorted_df.iloc[n_active:].copy()

    elif strategy == 'random':
        rng = np.random.default_rng(random_seed)
        chosen_idx   = rng.choice(n_total, size=n_active, replace=False)
        mask         = np.zeros(n_total, dtype=bool)
        mask[chosen_idx] = True
        selected_df  = df[mask].copy().reset_index(drop=True)
        remaining_df = df[~mask].copy().reset_index(drop=True)

    else:
        raise ValueError(f"strategy must be 'uncertainty' or 'random', got '{strategy}'")

    print(f'\nSelection strategy : {strategy}')
    print(f'Total test samples : {n_total}')
    print(f'Selected (-> train) : {n_active}')
    print(f'Remaining (-> test) : {len(remaining_df)}')
    if strategy == 'uncertainty':
        print(f'  y_std range of selected : '
              f'[{selected_df["y_std"].min():.4f}, {selected_df["y_std"].max():.4f}]')
        print(f'  y_std range of remaining: '
              f'[{remaining_df["y_std"].min():.4f}, {remaining_df["y_std"].max():.4f}]')

    return selected_df, remaining_df


def df_rows_to_graphs(df: pd.DataFrame,
                      test_gas: str,
                      experiment_name: str,
                      label: str = '') -> list:
    """
    Build PyG graphs directly from the 'smiles' and 'y_true' columns of
    a DataFrame slice (selected or remaining).  No float matching needed -
    the smiles column in the ensemble CSV is the ground truth identifier.

    Skips rows where SMILES is missing or RDKit-unparseable and warns.
    """
    graphs   = []
    n_skip   = 0

    for _, row in df.iterrows():
        smi     = str(row['smiles']).strip()
        y_target = float(row['y_true'])

        if not smi or smi.lower() == 'nan':
            n_skip += 1
            continue

        g = record_to_graph(smi, test_gas, y_target, experiment_name)
        if g is None:
            n_skip += 1
            continue
        graphs.append(g)

    tag = f' ({label})' if label else ''
    print(f'  Built {len(graphs)} graphs{tag}'
          + (f'  [{n_skip} skipped - bad SMILES]' if n_skip else ''))
    return graphs

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
        out = []
        for i, g in enumerate(graphs):
            gc = g.clone()
            gc.y = torch.tensor([y_vals[i]], dtype=torch.float)
            out.append(gc)
        return out

    return apply(train_graphs, y_tr_sc), apply(eval_graphs, y_ev_sc), y_sc

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
            nn.Linear(64, 1))

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch)
        gas_features = data.gas_features
        batch_size   = batch.max().item() + 1

        if gas_features.shape != (batch_size, self.expected_gas_dim):
            raise ValueError(
                f'gas_features shape {gas_features.shape} vs '
                f'[{batch_size}, {self.expected_gas_dim}]')

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
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            preds  .extend(model(data).cpu().numpy().tolist())
            targets.extend(data.y.cpu().numpy().tolist())
    return np.array(preds), np.array(targets)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return {
        'r2':   float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0,
        'rmse': float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        'mae':  float(np.mean(np.abs(y_pred - y_true))),
    }


def preflight_smiles_audit(ensemble_csv_path: str):
    """
    Verify that the ensemble CSV has a 'smiles' column and that every
    entry is a non-null, non-empty string that RDKit can parse.

    Prints a summary and raises RuntimeError on critical failures.
    """
    ens_df  = pd.read_csv(ensemble_csv_path)
    n_total = len(ens_df)

    if 'smiles' not in ens_df.columns:
        raise RuntimeError(
            f"No 'smiles' column found in {ensemble_csv_path}.\n"
            f"Re-run the ensemble script (updated version) which writes "
            f"the smiles column into the CSV.")

    n_null    = ens_df['smiles'].isna().sum()
    n_empty   = (ens_df['smiles'].fillna('').str.strip() == '').sum()
    n_invalid = 0
    invalid_examples = []

    for smi in ens_df['smiles'].dropna():
        if Chem.MolFromSmiles(str(smi)) is None:
            n_invalid += 1
            if len(invalid_examples) < 5:
                invalid_examples.append(smi)

    n_ok = n_total - n_null - n_empty - n_invalid

    print('\n' + ''*60)
    print('PRE-FLIGHT SMILES AUDIT')
    print(f'  Ensemble CSV      : {ensemble_csv_path}')
    print(f'  Total rows        : {n_total}')
    print(f'   Valid SMILES    : {n_ok}  ({100*n_ok/n_total:.1f}%)')
    print(f'   Null/missing    : {n_null}')
    print(f'   Empty strings   : {n_empty}')
    print(f'   RDKit-invalid   : {n_invalid}')

    if invalid_examples:
        print('  Invalid examples  :')
        for s in invalid_examples:
            print(f'    {s}')

    if n_null + n_empty + n_invalid == 0:
        print('  Status : PERFECT - all SMILES present and RDKit-parseable.')
    elif n_ok == n_total:
        print('  Status : OK')
    else:
        msg = (f'{n_null + n_empty + n_invalid} rows have missing or unparseable ')
        print(f'  Status : WARNING - {msg}SMILES. Those rows will be skipped.')

    print(''*60 + '\n')
    return {'n_total': n_total, 'n_ok': n_ok,
            'n_null': n_null, 'n_empty': n_empty, 'n_invalid': n_invalid}


def train_and_evaluate(train_graphs, test_graphs, hp, feature_dim,
                        label=''):
    """
    Train a single model on train_graphs, evaluate on test_graphs.
    Returns predictions in original log10-Barrer scale.
    """
    train_y, test_y, y_sc = scale_targets(train_graphs, test_graphs)
    train_sc, test_sc     = standardize_features(train_y, test_y)

    train_loader = DataLoader(train_sc, batch_size=hp['batch_size'],
                              shuffle=True,  num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY)
    test_loader  = DataLoader(test_sc,  batch_size=hp['batch_size'],
                              shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY)

    set_seeds(RANDOM_SEED)   # fixed seed -> single model is reproducible
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

    print(f'\n  Training {label}:  '
          f'{len(train_graphs)} train samples | '
          f'{len(test_graphs)} test samples')

    for epoch in range(FINAL_EPOCHS):
        tr_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler)
        if tr_loss < best_tr_loss:
            best_tr_loss = tr_loss
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve  += 1
        if (epoch + 1) % 100 == 0:
            print(f'    Epoch {epoch+1:>4}  train_loss={tr_loss:.4f}')
        if no_improve >= FINAL_PATIENCE:
            print(f'    Early stop at epoch {epoch+1}')
            break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    tp, tt  = evaluate(model, test_loader)
    y_pred  = y_sc.inverse_transform(tp.reshape(-1, 1)).flatten()
    y_true  = y_sc.inverse_transform(tt.reshape(-1, 1)).flatten()

    return y_true, y_pred

# MAIN
def main():
    print('\n' + '='*80)
    print('GC-MPNN  ACTIVE LEARNING')
    print(f'Experiment       : {EXPERIMENT_NAME}')
    print(f'Test gas         : {TEST_GAS}')
    print(f'N_ACTIVE         : {N_ACTIVE}')
    print(f'Strategy         : {SELECTION_STRATEGY}')
    print(f'Ensemble CSV     : {ENSEMBLE_CSV}')
    print('='*80)

    # Load hyperparameters
    hp          = load_hyperparameters(PARAMS_JSON)
    feature_dim = EXPERIMENT_CONFIGS[EXPERIMENT_NAME]['feature_dim']

    # Load wide CSV
    print(f'\nLoading wide CSV: {DATA_CSV}')
    df_wide, smiles_col, p_exp_map = load_wide_csv(DATA_CSV)
    print(f'  Rows in wide CSV: {len(df_wide)}')

    # Pre-flight SMILES audit
    preflight_smiles_audit(ENSEMBLE_CSV)

    # Build original training pool (no TEST_GAS)
    print('\nBuilding training pool graphs...')
    pool_graphs = build_pool_graphs(
        df_wide, smiles_col, p_exp_map, EXPERIMENT_NAME, TRAINING_POOL)
    print(f'  Pool ({"+".join(TRAINING_POOL)}): {len(pool_graphs)} graphs')
    # (wide CSV still needed above for the pool; no longer used for test matching)

    # Select active samples from ensemble predictions CSV
    print(f'\nLoading ensemble predictions: {ENSEMBLE_CSV}')
    selected_df, remaining_df = select_active_samples(
        ensemble_csv_path = ENSEMBLE_CSV,
        n_active          = N_ACTIVE,
        strategy          = SELECTION_STRATEGY,
        random_seed       = RANDOM_SEED,
    )

    # Build graphs directly from smiles column in ensemble CSV
    print('\nBuilding graphs from ensemble CSV smiles column...')
    active_graphs    = df_rows_to_graphs(selected_df,  TEST_GAS, EXPERIMENT_NAME,
                                         label='selected -> train')
    remaining_graphs = df_rows_to_graphs(remaining_df, TEST_GAS, EXPERIMENT_NAME,
                                         label='remaining -> test')

    # Augment training set
    augmented_train = pool_graphs + active_graphs
    print(f'\nAugmented training set: {len(pool_graphs)} pool '
          f'+ {len(active_graphs)} active = {len(augmented_train)} total')

    # Train & evaluate
    tag = f'{EXPERIMENT_NAME}_AL_{SELECTION_STRATEGY}_N{N_ACTIVE}'

    y_true, y_pred = train_and_evaluate(
        train_graphs = augmented_train,
        test_graphs  = remaining_graphs,
        hp           = hp,
        feature_dim  = feature_dim,
        label        = tag,
    )

    met = compute_metrics(y_true, y_pred)

    print(f'\n{""*60}')
    print(f'RESULTS  [{SELECTION_STRATEGY}  N_active={N_ACTIVE}]')
    print(f'  Test gas    : {TEST_GAS}')
    print(f'  Test samples: {len(remaining_graphs)}')
    print(f'  R2          : {met["r2"]:.4f}')
    print(f'  RMSE        : {met["rmse"]:.4f}  (log10 Barrer)')
    print(f'  MAE         : {met["mae"]:.4f}  (log10 Barrer)')
    print(f'{""*60}')

    # Save predictions CSV
    pred_df = pd.DataFrame({
        'smiles':     [g.smiles for g in remaining_graphs],
        'y_true':     y_true,
        'y_pred':     y_pred,
        'residual':   y_pred - y_true,
        'gas':        TEST_GAS,
        'experiment': EXPERIMENT_NAME,
        'strategy':   SELECTION_STRATEGY,
        'n_active':   N_ACTIVE,
    })
    # Carry over the y_std of remaining samples for downstream analysis
    # (useful to check if uncertainty-selected samples were indeed hard ones)
    if 'y_std' in remaining_df.columns:
        # align by y_true - safe because we already matched this way
        std_lookup = dict(zip(remaining_df['y_true'].values,
                               remaining_df['y_std'].values))
        pred_df['y_std_ensemble'] = [std_lookup.get(v, np.nan) for v in y_true]

    csv_out = f'{tag}_test_{TEST_GAS}_predictions.csv'
    pred_df.to_csv(csv_out, index=False)
    print(f'\nSaved predictions : {csv_out}')

    # Save summary JSON
    summary = {
        'experiment':         EXPERIMENT_NAME,
        'test_gas':           TEST_GAS,
        'selection_strategy': SELECTION_STRATEGY,
        'n_active':           N_ACTIVE,
        'n_remaining_test':   len(remaining_graphs),
        'n_augmented_train':  len(augmented_train),
        'metrics':            met,
        'hyperparameters':    hp,
    }
    json_out = f'{tag}_summary.json'
    with open(json_out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved summary     : {json_out}')

    print('\n' + '='*80)
    print('COMPLETE')
    print('='*80)


if __name__ == '__main__':
    main()
