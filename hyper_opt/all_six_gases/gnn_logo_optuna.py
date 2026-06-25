#!/usr/bin/env python
# coding: utf-8
"""
Gas-Conditioned MPNN - HPO on ALL 6 Gases + Save Pretrained Model
==================================================================

Goal
----
Train a GC-MPNN on all six gases (He, H2, N2, O2, CH4, CO2) so that
the resulting model can be used as a pretrained backbone for predicting
H2S permeability (which has no labelled data for validation).

Stage 1 - Optuna HPO
    Training pool = ALL 6 gases
    For each Optuna trial:
      For each gas V in {He, H2, N2, O2, CH4, CO2}:
        val_graphs   = graphs for gas V
        train_graphs = graphs for the other 5 gases
        Train, evaluate inner val MSE
      Mean of inner val MSEs  ->  Optuna objective (minimise)

Stage 2 - Final Training
    Train on ALL 6 gases combined with the best hyperparameters.
    Save:
      - Model weights        : gc_mpnn_pretrained.pt
      - Full checkpoint      : gc_mpnn_pretrained_checkpoint.pt
        (includes weights, scalers, hyperparams, gas feature config)

The checkpoint is self-contained for H2S inference - load it, pass a
polymer graph + H2S gas features, and get a log10-Barrer prediction.

Outputs
-------
    gc_mpnn_pretrained.pt              - model state_dict only
    gc_mpnn_pretrained_checkpoint.pt   - full inference checkpoint
    all6_optuna_trials.csv             - Optuna trial history
    all6_best_params.json              - best hyperparameters
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
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# DEVICE
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon)")
else:
    device = torch.device("cpu")
    print("Using CPU")

NUM_WORKERS = 2 if device.type == 'cuda' else 0
PIN_MEMORY  = device.type == 'cuda'

def set_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seeds(42)

# KEY SETTINGS
N_OPTUNA_TRIALS = 75
HPO_EPOCHS      = 100
HPO_PATIENCE    = 20
FINAL_EPOCHS    = 500
FINAL_PATIENCE  = 100

# All 6 gases used for both HPO and final training - no gas is locked away
ALL_GASES = ['He', 'H2', 'N2', 'O2', 'CH4', 'CO2']

# Gas descriptor experiment to use for the pretrained model
# Change this if you want a different descriptor set (see EXPERIMENT_CONFIGS below)
EXPERIMENT_NAME = 'Kinetic'

print(f"\nAll gases used for HPO and training: {ALL_GASES}")
print(f"Gas descriptor experiment           : {EXPERIMENT_NAME}")

# GAS PROPERTIES
# Electrostatic descriptors updated vs. original code:
# q_pos / q_neg  ->  q, alpha
# q     = |q|, magnitude of partial charge on positive site (dimensionless)
# (TraPPE / force-field parameterisation)
# alpha = static isotropic dipole polarizability (A3)
# Source: NIST CCCBDB (Olney et al. 1997)
# He=0.208, H2=0.787, O2=1.562, N2=1.710, CH4=2.448, CO2=2.507
# H2S=3.631 (NIST CCCBDB)
# Rationale: q_pos and q_neg were perfectly correlated (r~1.0) - one was
# redundant. alpha independently captures induced-dipole / dispersion
# interactions, which are especially important for CO2 and H2S.
GAS_PROPERTIES = {
    'He':  {'sigma': 2.551, 'epsilon':  10.2,  'omega': -0.383, 'Tc':   5.2, 'Pc':  2.28, 'd': 2.6,  'Vd':  2.67, 'q': 0.0,   'alpha': 0.208},
    'H2':  {'sigma': 2.827, 'epsilon':  59.7,  'omega': -0.265, 'Tc':  33.2, 'Pc': 13.00, 'd': 2.89, 'Vd':  6.12, 'q': 0.0,   'alpha': 0.787},
    'N2':  {'sigma': 3.798, 'epsilon':  71.4,  'omega':  0.037, 'Tc': 126.2, 'Pc': 63.14, 'd': 3.64, 'Vd': 18.5,  'q': 0.964, 'alpha': 1.710},
    'O2':  {'sigma': 3.467, 'epsilon': 106.7,  'omega':  0.022, 'Tc': 154.6, 'Pc': 50.43, 'd': 3.46, 'Vd': 16.3,  'q': 0.226, 'alpha': 1.562},
    'CH4': {'sigma': 3.758, 'epsilon': 148.6,  'omega':  0.011, 'Tc': 190.6, 'Pc': 46.1,  'd': 3.8,  'Vd': 24.42, 'q': 0.0,   'alpha': 2.448},
    'CO2': {'sigma': 3.941, 'epsilon': 195.2,  'omega':  0.253, 'Tc': 304.1, 'Pc': 73.80, 'd': 3.3,  'Vd': 26.9,  'q': 0.70,  'alpha': 2.507},
    'H2S': {'sigma': 3.623, 'epsilon': 301.1,  'omega':  0.100, 'Tc': 373.3, 'Pc': 89.63, 'd': 3.6,  'Vd': 32.9,  'q': 0.42,  'alpha': 3.631},
}

# GAS FEATURE FUNCTIONS
def get_gas_features_thermodynamic(g):
    p = GAS_PROPERTIES[g]
    return np.array([p['sigma'], p['epsilon'], p['omega'], p['Tc'], p['Pc']], dtype=np.float32)

def get_gas_features_kinetic(g):
    p = GAS_PROPERTIES[g]
    return np.array([p['d'], p['Vd']], dtype=np.float32)

def get_gas_features_electrostatics(g):
    p = GAS_PROPERTIES[g]
    return np.array([p['q'], p['alpha']], dtype=np.float32)

def get_gas_features_thermo_and_kinetic(g):
    p = GAS_PROPERTIES[g]
    return np.array([p['sigma'], p['epsilon'], p['omega'],
                     p['Tc'], p['Pc'], p['d'], p['Vd']], dtype=np.float32)

def get_gas_features_full(g):
    p = GAS_PROPERTIES[g]
    return np.array([p['sigma'], p['epsilon'], p['omega'],
                     p['Tc'], p['Pc'], p['d'], p['Vd'],
                     p['q'], p['alpha']], dtype=np.float32)

def get_gas_features_onehot(g):
    # NOTE: OneHot is defined over the 6 training gases only.
    # H2S will get an all-zero vector at inference (unseen gas).
    # If you want H2S to be inferable with OneHot, switch to a
    # physical descriptor experiment (Thermodynamic, Full, etc.).
    idx = {'He': 0, 'H2': 1, 'N2': 2, 'O2': 3, 'CH4': 4, 'CO2': 5}
    v = np.zeros(6, dtype=np.float32)
    if g in idx:
        v[idx[g]] = 1.0
    return v

EXPERIMENT_CONFIGS = {
    'Thermodynamic':      {'feature_func': get_gas_features_thermodynamic,     'feature_dim': 5, 'description': 'sigma, epsilon, omega, Tc, Pc'},
    'Kinetic':            {'feature_func': get_gas_features_kinetic,           'feature_dim': 2, 'description': 'd, Vd'},
    'Electrostatics':     {'feature_func': get_gas_features_electrostatics,    'feature_dim': 2, 'description': '|q|, alpha'},
    'Thermo_and_Kinetic': {'feature_func': get_gas_features_thermo_and_kinetic,'feature_dim': 7, 'description': 'sigma, epsilon, omega, Tc, Pc, d, Vd'},
    'Full':               {'feature_func': get_gas_features_full,              'feature_dim': 9, 'description': 'sigma, epsilon, omega, Tc, Pc, d, Vd, |q|, alpha'},
    'OneHot':             {'feature_func': get_gas_features_onehot,            'feature_dim': 6, 'description': 'Categorical (1-of-6)'},
}

# DATA LOADING
print("\nLoading data...")
pol_sd = pd.read_csv('../../data/Gas_permeability_solubility_diffusivity_wide.csv')

smiles    = pol_sd['smiles_string']
p_exp_map = {
    'CH4': pol_sd['p_exp_CH4'], 'CO2': pol_sd['p_exp_CO2'],
    'H2':  pol_sd['p_exp_H2'],  'N2':  pol_sd['p_exp_N2'],
    'O2':  pol_sd['p_exp_O2'],  'He':  pol_sd['p_exp_He'],
}


def create_dataset(experiment_name, gas_subset=None):
    if gas_subset is None:
        gas_subset = ALL_GASES
    feat_fn = EXPERIMENT_CONFIGS[experiment_name]['feature_func']
    records = []
    for idx in range(len(smiles)):
        smi = smiles.iloc[idx]
        for gas in gas_subset:
            perm = p_exp_map[gas].iloc[idx]
            if not np.isnan(perm):
                records.append({
                    'smiles': smi, 'gas': gas,
                    'permeability': perm + 10.0,
                    'gas_features': feat_fn(gas),
                })
    return records

# GRAPH CONSTRUCTION
def smiles_to_graph(smi):
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
            rec['gas_features'], dtype=torch.float).unsqueeze(0)  # [1, gas_dim]
        g.y        = torch.tensor([rec['permeability']], dtype=torch.float)
        g.gas_name = rec['gas']
        dataset.append(g)
    return dataset

# FEATURE + TARGET STANDARDISATION
# Returns scalers so they can be saved with the model checkpoint
def standardize_features(train_graphs, val_graphs,
                          node_sc=None, edge_sc=None, gas_sc=None,
                          fit=True):
    """
    If fit=True  : fit scalers on train_graphs, transform both sets.
    If fit=False : use provided scalers to transform only (inference mode).
    """
    if fit:
        node_sc = StandardScaler()
        edge_sc = StandardScaler()
        gas_sc  = StandardScaler()
        node_sc.fit(np.vstack([g.x.numpy() for g in train_graphs]))
        gas_sc.fit(np.array([g.gas_features.squeeze(0).numpy()
                              for g in train_graphs]))
        edge_data = [g.edge_attr.numpy()
                     for g in train_graphs if g.edge_attr.shape[0] > 0]
        if edge_data:
            edge_sc.fit(np.vstack(edge_data))

    def scale(graphs):
        out = []
        for g in graphs:
            gc = g.clone()
            gc.x = torch.tensor(node_sc.transform(g.x.numpy()),
                                 dtype=torch.float)
            gc.gas_features = torch.tensor(
                gas_sc.transform(
                    g.gas_features.squeeze(0).numpy().reshape(1, -1)),
                dtype=torch.float)
            if g.edge_attr.shape[0] > 0:
                gc.edge_attr = torch.tensor(
                    edge_sc.transform(g.edge_attr.numpy()), dtype=torch.float)
            out.append(gc)
        return out

    scaled_train = scale(train_graphs)
    scaled_val   = scale(val_graphs) if val_graphs else []
    return scaled_train, scaled_val, node_sc, edge_sc, gas_sc


def scale_targets(train_graphs, val_graphs):
    y_train = np.array([g.y.item() for g in train_graphs])
    y_sc    = StandardScaler()
    y_tr_sc = y_sc.fit_transform(y_train.reshape(-1, 1)).flatten()

    def apply(graphs, y_vals):
        return [
            (lambda gc: (setattr(gc, 'y',
                                 torch.tensor([y_vals[i]], dtype=torch.float)),
                          gc)[1])(g.clone())
            for i, g in enumerate(graphs)
        ]

    train_out = apply(train_graphs, y_tr_sc)

    if val_graphs:
        y_val   = np.array([g.y.item() for g in val_graphs])
        y_vl_sc = y_sc.transform(y_val.reshape(-1, 1)).flatten()
        val_out = apply(val_graphs, y_vl_sc)
    else:
        val_out = []

    return train_out, val_out, y_sc

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
        self.node_embedding   = nn.Linear(node_features, hidden_dim)
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
            raise ValueError(f"gas_features shape {gas_features.shape} vs "
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

# TRAINING / EVALUATION HELPERS
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


def run_training_fold(train_raw, val_raw, feature_dim, hp,
                      max_epochs, patience):
    """Scale, train, return best val MSE (in scaled space). Used in Stage 1."""
    train_y, val_y, _ = scale_targets(train_raw, val_raw)
    train_sc, val_sc, _, _, _ = standardize_features(train_y, val_y)

    train_loader = DataLoader(train_sc, batch_size=hp['batch_size'],
                              shuffle=True,  num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY)
    val_loader   = DataLoader(val_sc,   batch_size=hp['batch_size'],
                              shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY)

    model = GasConditionedMPNN(
        node_features=7, edge_features=7, gas_features=feature_dim,
        hidden_dim=hp['hidden_dim'], num_mp_layers=hp['num_mp_layers'],
        fusion_dim=hp['fusion_dim'], l2_lambda=hp['l2_lambda'],
        dropout=hp['dropout'], pooling=hp['pooling'],
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=hp['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_val_mse = float('inf')
    no_improve   = 0

    for epoch in range(max_epochs):
        train_epoch(model, train_loader, criterion, optimizer, scheduler)
        vp, vt = evaluate(model, val_loader)
        val_mse = float(np.mean((vp - vt) ** 2))
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            no_improve   = 0
        else:
            no_improve  += 1
        if no_improve >= patience:
            break

    return best_val_mse

# STAGE 1 - OPTUNA OBJECTIVE  (all 6 gases, LOGO rotation)
def make_hpo_objective(graphs_by_gas, feature_dim):
    """
    For each trial:
      For each gas V in ALL_GASES:
        val  = graphs for gas V
        train = graphs for the other 5 gases
        Compute inner val MSE
      Return mean val MSE over all 6 rotations.
    No gas is locked away - this is pure cross-gas CV for HPO.
    """
    def objective(trial: optuna.Trial) -> float:
        hp = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-1, log=True),
            'l2_lambda':     trial.suggest_float('l2_lambda',     1e-5, 1e-1, log=True),
            'hidden_dim':    trial.suggest_categorical('hidden_dim',    [32, 64, 128, 256, 512]),
            'num_mp_layers': trial.suggest_categorical('num_mp_layers', [2, 3, 4, 5, 6, 7]),
            'fusion_dim':    trial.suggest_categorical('fusion_dim',    [16, 32, 64, 128, 256, 512]),
            'dropout':       trial.suggest_float('dropout',       0.001, 0.75),
            'pooling':       trial.suggest_categorical('pooling', ['mean', 'attention', 'sum']),
            'batch_size':    trial.suggest_categorical('batch_size', [4, 8, 16, 32, 64, 128, 256]),
        }

        rotation_mses = []

        for step, val_gas in enumerate(ALL_GASES):
            val_raw   = graphs_by_gas.get(val_gas, [])
            train_raw = []
            for g_name in ALL_GASES:
                if g_name != val_gas:
                    train_raw.extend(graphs_by_gas.get(g_name, []))

            if not val_raw or not train_raw:
                continue

            try:
                fold_mse = run_training_fold(
                    train_raw, val_raw, feature_dim, hp,
                    max_epochs=HPO_EPOCHS, patience=HPO_PATIENCE)
                rotation_mses.append(fold_mse)
            except Exception as e:
                raise optuna.TrialPruned(f"Fold {val_gas} failed: {e}")

            trial.report(float(np.mean(rotation_mses)), step=step)
            if trial.should_prune():
                raise optuna.TrialPruned(
                    f"Pruned after {val_gas} "
                    f"(mean MSE={np.mean(rotation_mses):.4f})")

        return float(np.mean(rotation_mses))

    return objective

# STAGE 2 - FINAL TRAINING ON ALL 6 GASES + SAVE MODEL
def train_and_save(all_graphs, feature_dim, hp, experiment_name):
    """
    Train on all 6 gases with best hyperparameters.
    Save:
      - state_dict only : gc_mpnn_pretrained.pt
      - full checkpoint : gc_mpnn_pretrained_checkpoint.pt
        Contains model weights + all scalers + config needed for H2S inference.
    """
    print(f"\n  Training on all {len(all_graphs)} graphs "
          f"({len(ALL_GASES)} gases) with best hyperparameters...")

    # Scale targets
    # No separate val set - use the whole dataset for training
    train_sc_y, _, y_sc = scale_targets(all_graphs, [])

    # Scale features
    train_sc, _, node_sc, edge_sc, gas_sc = standardize_features(
        train_sc_y, [], fit=True)

    train_loader = DataLoader(
        train_sc, batch_size=hp['batch_size'],
        shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # Build model
    model = GasConditionedMPNN(
        node_features=7, edge_features=7, gas_features=feature_dim,
        hidden_dim=hp['hidden_dim'], num_mp_layers=hp['num_mp_layers'],
        fusion_dim=hp['fusion_dim'], l2_lambda=hp['l2_lambda'],
        dropout=hp['dropout'], pooling=hp['pooling'],
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=hp['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=FINAL_EPOCHS)

    best_loss  = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(FINAL_EPOCHS):
        tr_loss = train_epoch(model, train_loader, criterion,
                              optimizer, scheduler)
        if tr_loss < best_loss:
            best_loss  = tr_loss
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if (epoch + 1) % 100 == 0:
            print(f"    Epoch {epoch+1:>4}  train_loss={tr_loss:.4f}  "
                  f"best={best_loss:.4f}")
        if no_improve >= FINAL_PATIENCE:
            print(f"    Early stop at epoch {epoch+1}")
            break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Quick train-set metrics
    tp, tt = evaluate(model, train_loader)
    y_pred_tr = y_sc.inverse_transform(tp.reshape(-1, 1)).flatten()
    y_true_tr = y_sc.inverse_transform(tt.reshape(-1, 1)).flatten()
    ss_res = np.sum((y_true_tr - y_pred_tr) ** 2)
    ss_tot = np.sum((y_true_tr - np.mean(y_true_tr)) ** 2)
    train_r2   = float(1 - ss_res / ss_tot)
    train_rmse = float(np.sqrt(np.mean((y_pred_tr - y_true_tr) ** 2)))
    print(f"\n  Final train metrics (all 6 gases):")
    print(f"    R2   = {train_r2:.4f}")
    print(f"    RMSE = {train_rmse:.4f}  (log10-Barrer)")

    # Save state_dict only
    weights_path = 'gc_mpnn_pretrained.pt'
    torch.save(best_state, weights_path)
    print(f"\n  Saved weights : {weights_path}")

    # Save full inference checkpoint
    # Everything needed to run inference on a new gas (e.g. H2S)
    # without re-importing any training code.
    checkpoint = {
        # Model weights
        'model_state_dict': best_state,

        # Architecture config (re-instantiate the model)
        'model_config': {
            'node_features':  7,
            'edge_features':  7,
            'gas_features':   feature_dim,
            'hidden_dim':     hp['hidden_dim'],
            'num_mp_layers':  hp['num_mp_layers'],
            'fusion_dim':     hp['fusion_dim'],
            'l2_lambda':      hp['l2_lambda'],
            'dropout':        hp['dropout'],
            'pooling':        hp['pooling'],
        },

        # Sklearn scalers (needed for preprocessing)
        'node_scaler':    node_sc,   # StandardScaler for node features
        'edge_scaler':    edge_sc,   # StandardScaler for edge features
        'gas_scaler':     gas_sc,    # StandardScaler for gas features
        'target_scaler':  y_sc,      # StandardScaler for log10-Barrer targets

        # Experiment metadata
        'experiment_name': experiment_name,
        'gas_feature_dim': feature_dim,
        'gas_feature_desc': EXPERIMENT_CONFIGS[experiment_name]['description'],
        'trained_on_gases': ALL_GASES,

        # H2S inference note
        # For physical descriptor experiments (Thermodynamic, Full, etc.):
        # gas_feat = get_gas_features_<exp>(H2S)  ->  gas_scaler.transform(...)
        # For OneHot:
        # H2S has no training index; it will receive a zero vector.
        # Physical descriptors are strongly recommended for H2S.
        'h2s_inference_note': (
            "Use GAS_PROPERTIES['H2S'] with the same feature function as "
            f"'{experiment_name}' to build the H2S gas feature vector. "
            "Apply gas_scaler.transform() before passing to the model. "
            "For Kinetic: H2S uses d=3.6 A, Vd=32.9 cm3/mol - both are "
            "within/near the training gas range so extrapolation risk is low. "
            "For Full/Thermodynamic: all 9 H2S properties are in GAS_PROPERTIES. "
            "OneHot encodes H2S as all-zeros (unseen class) - avoid for H2S inference."
        ),

        # Training metrics
        'train_r2':   train_r2,
        'train_rmse': train_rmse,
        'best_hpo_mse': None,  # filled in by caller
    }

    ckpt_path = 'gc_mpnn_pretrained_checkpoint.pt'
    torch.save(checkpoint, ckpt_path)
    print(f"  Saved checkpoint: {ckpt_path}")
    print(f"    (contains weights + node/edge/gas/target scalers + config)")

    return checkpoint, train_r2, train_rmse

# MAIN
config      = EXPERIMENT_CONFIGS[EXPERIMENT_NAME]
feature_dim = config['feature_dim']

print("\n" + "="*80)
print("GC-MPNN  -  HPO on ALL 6 GASES  +  PRETRAINED MODEL SAVE")
print(f"Experiment  : {EXPERIMENT_NAME}  ({config['description']})")
print(f"All gases   : {ALL_GASES}")
print(f"Optuna trials          : {N_OPTUNA_TRIALS}")
print(f"HPO epochs / patience  : {HPO_EPOCHS} / {HPO_PATIENCE}")
print(f"Final epochs / patience: {FINAL_EPOCHS} / {FINAL_PATIENCE}")
print("="*80)

# Build full dataset (all 6 gases)
print("\nBuilding datasets...")
all_graphs = build_pyg_dataset(create_dataset(EXPERIMENT_NAME, ALL_GASES))
print(f"Total graphs (all 6 gases): {len(all_graphs)}")

graphs_by_gas = {}
for g in all_graphs:
    graphs_by_gas.setdefault(g.gas_name, []).append(g)
print("Per-gas counts: " +
      "  ".join(f"{g}={len(graphs_by_gas.get(g, []))}" for g in ALL_GASES))

# STAGE 1: Optuna HPO
print(f"\n{''*60}")
print(f"Stage 1 - Optuna HPO  ({N_OPTUNA_TRIALS} trials)")
print(f"Objective : mean inner val MSE across all 6 LOGO rotations")
print(f"{''*60}")

study = optuna.create_study(
    direction  = 'minimize',
    sampler    = TPESampler(seed=42),
    pruner     = MedianPruner(n_startup_trials=5, n_warmup_steps=2,
                              interval_steps=1),
    study_name = f"all6_hpo_{EXPERIMENT_NAME}",
)

study.optimize(
    make_hpo_objective(graphs_by_gas, feature_dim),
    n_trials          = N_OPTUNA_TRIALS,
    show_progress_bar = True,
    gc_after_trial    = True,
)

best_params = study.best_params
best_mse    = study.best_value
n_complete  = sum(1 for t in study.trials
                  if t.state == optuna.trial.TrialState.COMPLETE)
n_pruned    = sum(1 for t in study.trials
                  if t.state == optuna.trial.TrialState.PRUNED)

print(f"\nTrials - complete: {n_complete}  pruned: {n_pruned}")
print(f"Best mean inner val MSE (scaled): {best_mse:.6f}")
print("Best hyperparameters:")
for k, v in best_params.items():
    print(f"  {k:<20} : {v}")

print("\nPer-rotation val MSE for best trial:")
for step, gas in enumerate(ALL_GASES):
    val = study.best_trial.intermediate_values.get(step, float('nan'))
    print(f"  val_gas={gas:<4}  MSE={val:.4f}")

study.trials_dataframe().to_csv('all6_optuna_trials.csv', index=False)

best_params_clean = {
    k: (float(v) if isinstance(v, (np.floating, float)) else v)
    for k, v in best_params.items()
}
with open('all6_best_params.json', 'w') as f:
    json.dump({'params': best_params_clean,
               'best_hpo_mse': float(best_mse),
               'experiment': EXPERIMENT_NAME}, f, indent=2)

print("Saved: all6_optuna_trials.csv")
print("Saved: all6_best_params.json")

print("\nHyperparameter Importances (FAnova):")
try:
    for param, imp in optuna.importance.get_param_importances(study).items():
        print(f"  {param:<20} : {imp:.4f}")
except Exception:
    print("  (requires >= 4 completed non-pruned trials)")

# STAGE 2: Final training + save
print(f"\n{''*60}")
print(f"Stage 2 - Final Training on ALL 6 gases  +  Model Save")
print(f"{''*60}")

checkpoint, train_r2, train_rmse = train_and_save(
    all_graphs, feature_dim, best_params, EXPERIMENT_NAME)

# Backfill best_hpo_mse into saved checkpoint
checkpoint['best_hpo_mse'] = float(best_mse)
torch.save(checkpoint, 'gc_mpnn_pretrained_checkpoint.pt')

# SUMMARY
print("\n" + "="*80)
print("COMPLETE - Pretrained GC-MPNN saved")
print("="*80)
print(f"  Experiment        : {EXPERIMENT_NAME}  ({config['description']})")
print(f"  Trained on        : {ALL_GASES}  ({len(all_graphs)} graphs)")
print(f"  Final train R2    : {train_r2:.4f}")
print(f"  Final train RMSE  : {train_rmse:.4f}  (log10-Barrer)")
print(f"  Best HPO MSE      : {best_mse:.6f}  (scaled, avg over 6 LOGO folds)")
print()
print("  Saved files:")
print("    gc_mpnn_pretrained.pt             - state_dict only")
print("    gc_mpnn_pretrained_checkpoint.pt  - full inference checkpoint")
print("    all6_optuna_trials.csv            - Optuna trial history")
print("    all6_best_params.json             - best hyperparameters")
print()
print("  To predict H2S permeability, load 'gc_mpnn_pretrained_checkpoint.pt'")
print("  and use GAS_PROPERTIES['H2S'] with the gas feature function for")
print(f"  '{EXPERIMENT_NAME}' to build the gas feature vector.")
if EXPERIMENT_NAME == 'OneHot':
    print()
    print("    WARNING: OneHot encodes H2S as all-zeros (unseen class).")
    print("     Consider re-running with EXPERIMENT_NAME = 'Full' or")
    print("     'Thermodynamic' for physically meaningful H2S predictions.")
elif EXPERIMENT_NAME == 'Kinetic':
    print()
    print("    Kinetic descriptors (d, Vd): H2S is encodable at inference.")
    print("     Supply d=3.6, Vd=32.9 for H2S - both are within/near the")
    print("     training gas range; the gas_scaler handles normalisation.")
print("="*80)
