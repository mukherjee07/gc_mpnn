#!/usr/bin/env python
# coding: utf-8
"""
GC-MPNN - Evidential Deep Regression with LOGO Lambda Calibration
==================================================================

Three-stage pipeline.  TEST_GAS is locked away for the entire first
two stages and is only used once, at the very end.

Stage 0 - lambda_reg calibration  (TEST_GAS locked away)
---------------------------------------------------------
  Training pool = all gases EXCEPT TEST_GAS  (5 gases)

  For a given lambda_reg candidate:
    For each gas V in TRAINING_POOL:
      train  on the other 4 gases (evidential loss with this lambda_reg)
      predict on gas V  ->  compute PICP_95
    mean_PICP = mean over 5 rotations

  Bisection search over lambda_reg until |mean_PICP - TARGET_PICP| < TOL
  -> optimal lambda_reg*

  Monotonicity guarantee: higher lambda_reg -> wider sigma_tot -> higher PICP.
  Bisection is therefore exact and requires O(log2(range/tol)) evaluations.

Stage 1 - Final training  (TEST_GAS still locked away)
------------------------------------------------------
  Train evidential GC-MPNN on ALL 5 pool gases with lambda_reg*.
  Save model + scalers to checkpoint.

Stage 2 - Test evaluation  (TEST_GAS unlocked)
----------------------------------------------
  Predict on TEST_GAS. Report R2, RMSE, MAE, PICP, MPIW.
  Save predictions CSV + summary JSON.

Outputs
-------
  {experiment}_evidential_test_{TEST_GAS}_predictions.csv
  {experiment}_evidential_train_pool_predictions.csv
  {experiment}_evidential_summary.json
  {experiment}_lambda_calibration_log.csv   <- bisection history
"""

import pandas as pd
import numpy as np
import os, json
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

# KEY SETTINGS
EXPERIMENT_NAME = 'Kinetic'
PARAMS_JSON     = f'{EXPERIMENT_NAME}_best_params.json'
TEST_GAS        = 'CO2'
DATA_CSV        = '../../../data/Gas_permeability_solubility_diffusivity_wide.csv'
SEED            = 42

# Stage 0: lambda calibration
TARGET_PICP      = 0.95   # desired coverage
PICP_TOL         = 0.01   # stop bisection when |PICP - TARGET| < this
MAX_BISECT_ITERS = 20     # safety cap on bisection iterations
LAMBDA_LOW       = 1e-4   # lower bound of search (expect PICP < 0.95)
LAMBDA_HIGH      = 5.0    # upper bound of search (expect PICP > 0.95)

# Epochs used inside calibration LOGO folds (can be lower than final
# to save compute; increase if you have GPU budget)
CAL_EPOCHS   = 100
CAL_PATIENCE = 20

# Stage 1: final training
FINAL_EPOCHS   = 500
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

set_seeds(SEED)

# LOAD HYPERPARAMETERS
def load_hyperparameters(json_path: str) -> dict:
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Hyperparameter file not found: {json_path}\n"
            "Run the nested HPO script first.")
    with open(json_path) as f:
        data = json.load(f)
    hp = data['params']
    hp['hidden_dim']    = int(hp['hidden_dim'])
    hp['num_mp_layers'] = int(hp['num_mp_layers'])
    hp['fusion_dim']    = int(hp['fusion_dim'])
    hp['batch_size']    = int(hp['batch_size'])
    print(f"\nLoaded hyperparameters from: {json_path}")
    for k, v in hp.items():
        print(f"  {k:<20} : {v}")
    return hp

# GAS PROPERTIES  (q / alpha descriptors; H2S included)
GAS_PROPERTIES = {
    'He':  {'sigma': 2.551, 'epsilon':  10.2,  'omega': -0.383, 'Tc':   5.2, 'Pc':  2.28, 'd': 2.6,  'Vd':  2.67, 'q': 0.0,   'alpha': 0.208},
    'H2':  {'sigma': 2.827, 'epsilon':  59.7,  'omega': -0.265, 'Tc':  33.2, 'Pc': 13.00, 'd': 2.89, 'Vd':  6.12, 'q': 0.0,   'alpha': 0.787},
    'N2':  {'sigma': 3.798, 'epsilon':  71.4,  'omega':  0.037, 'Tc': 126.2, 'Pc': 63.14, 'd': 3.64, 'Vd': 18.5,  'q': 0.964, 'alpha': 1.710},
    'O2':  {'sigma': 3.467, 'epsilon': 106.7,  'omega':  0.022, 'Tc': 154.6, 'Pc': 50.43, 'd': 3.46, 'Vd': 16.3,  'q': 0.226, 'alpha': 1.562},
    'CH4': {'sigma': 3.758, 'epsilon': 148.6,  'omega':  0.011, 'Tc': 190.6, 'Pc': 46.1,  'd': 3.8,  'Vd': 24.42, 'q': 0.0,   'alpha': 2.448},
    'CO2': {'sigma': 3.941, 'epsilon': 195.2,  'omega':  0.253, 'Tc': 304.1, 'Pc': 73.80, 'd': 3.3,  'Vd': 26.9,  'q': 0.70,  'alpha': 2.507},
    'H2S': {'sigma': 3.623, 'epsilon': 301.1,  'omega':  0.100, 'Tc': 373.3, 'Pc': 89.63, 'd': 3.6,  'Vd': 32.9,  'q': 0.42,  'alpha': 3.631},
}

EXPERIMENT_CONFIGS = {
    'Thermodynamic':      {'feature_func': lambda g: np.array([GAS_PROPERTIES[g]['sigma'], GAS_PROPERTIES[g]['epsilon'], GAS_PROPERTIES[g]['omega'], GAS_PROPERTIES[g]['Tc'], GAS_PROPERTIES[g]['Pc']], dtype=np.float32), 'feature_dim': 5},
    'Kinetic':            {'feature_func': lambda g: np.array([GAS_PROPERTIES[g]['d'], GAS_PROPERTIES[g]['Vd']], dtype=np.float32), 'feature_dim': 2},
    'Electrostatics':     {'feature_func': lambda g: np.array([GAS_PROPERTIES[g]['q'], GAS_PROPERTIES[g]['alpha']], dtype=np.float32), 'feature_dim': 2},
    'Thermo_and_Kinetic': {'feature_func': lambda g: np.array([GAS_PROPERTIES[g]['sigma'], GAS_PROPERTIES[g]['epsilon'], GAS_PROPERTIES[g]['omega'], GAS_PROPERTIES[g]['Tc'], GAS_PROPERTIES[g]['Pc'], GAS_PROPERTIES[g]['d'], GAS_PROPERTIES[g]['Vd']], dtype=np.float32), 'feature_dim': 7},
    'Full':               {'feature_func': lambda g: np.array([GAS_PROPERTIES[g]['sigma'], GAS_PROPERTIES[g]['epsilon'], GAS_PROPERTIES[g]['omega'], GAS_PROPERTIES[g]['Tc'], GAS_PROPERTIES[g]['Pc'], GAS_PROPERTIES[g]['d'], GAS_PROPERTIES[g]['Vd'], GAS_PROPERTIES[g]['q'], GAS_PROPERTIES[g]['alpha']], dtype=np.float32), 'feature_dim': 9},
    'OneHot':             {'feature_func': lambda g: np.eye(6, dtype=np.float32)[{'He':0,'H2':1,'N2':2,'O2':3,'CH4':4,'CO2':5}.get(g, 0)], 'feature_dim': 6},
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
                    'smiles':       smi, 'gas': gas,
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
        g.smiles   = rec['smiles']
        dataset.append(g)
    return dataset

# FEATURE & TARGET STANDARDISATION
def standardize_features(train_graphs, eval_graphs):
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
            if g.edge_attr.shape[0] > 0 and edge_data:
                gc.edge_attr = torch.tensor(
                    edge_sc.transform(g.edge_attr.numpy()), dtype=torch.float)
            out.append(gc)
        return out

    return scale(train_graphs), scale(eval_graphs), node_sc, edge_sc, gas_sc


def scale_targets(train_graphs, eval_graphs):
    y_train = np.array([g.y.item() for g in train_graphs])
    y_sc    = StandardScaler()
    y_tr_sc = y_sc.fit_transform(y_train.reshape(-1, 1)).flatten()

    def apply(graphs, y_vals):
        out = []
        for i, g in enumerate(graphs):
            gc = g.clone(); gc.y = torch.tensor([y_vals[i]], dtype=torch.float)
            out.append(gc)
        return out

    train_out = apply(train_graphs, y_tr_sc)
    if eval_graphs:
        y_ev_sc = y_sc.transform(
            np.array([g.y.item() for g in eval_graphs]).reshape(-1, 1)).flatten()
        eval_out = apply(eval_graphs, y_ev_sc)
    else:
        eval_out = []
    return train_out, eval_out, y_sc

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


class EvidentialHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 4)

    def forward(self, x):
        out   = self.fc(x)
        gamma = out[:, 0]
        nu    = F.softplus(out[:, 1]) + 1e-6
        alpha = F.softplus(out[:, 2]) + 1.0 + 1e-6
        beta  = F.softplus(out[:, 3]) + 1e-6
        return gamma, nu, alpha, beta


class EvidentialGasConditionedMPNN(nn.Module):
    def __init__(self, node_features=7, edge_features=7, gas_features=2,
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

        pre_head_dim = 64
        self.fusion_body = nn.Sequential(
            nn.Linear(hidden_dim + fusion_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),                     nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, pre_head_dim),             nn.ReLU())

        self.evidential_head = EvidentialHead(pre_head_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch)
        batch_size = batch.max().item() + 1
        if data.gas_features.shape != (batch_size, self.expected_gas_dim):
            raise ValueError(
                f"gas_features shape {data.gas_features.shape} vs "
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
        g_emb    = self.gas_encoder(data.gas_features)
        features = self.fusion_body(torch.cat([p_emb, g_emb], dim=-1))
        return self.evidential_head(features)

    def l2_regularization(self):
        l2 = torch.tensor(0., device=next(self.parameters()).device)
        for p in self.parameters():
            l2 += torch.norm(p, 2)
        return self.l2_lambda * l2

# EVIDENTIAL LOSS
def nig_nll(y, gamma, nu, alpha, beta):
    two_beta_v = 2.0 * beta * (1.0 + nu)
    return (
        0.5 * torch.log(torch.tensor(np.pi, device=y.device) / nu)
        - alpha * torch.log(two_beta_v)
        + (alpha + 0.5) * torch.log(nu * (y - gamma) ** 2 + two_beta_v)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    ).mean()


def nig_reg(y, gamma, nu, alpha):
    return (torch.abs(y - gamma) * (2.0 * nu + alpha)).mean()


def evidential_loss(y, gamma, nu, alpha, beta, lambda_reg):
    return nig_nll(y, gamma, nu, alpha, beta) + lambda_reg * nig_reg(y, gamma, nu, alpha)

# TRAINING HELPERS
def train_epoch(model, loader, optimizer, scheduler, lambda_reg):
    model.train()
    total = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        gamma, nu, alpha, beta = model(data)
        loss = evidential_loss(data.y.squeeze(-1), gamma, nu, alpha, beta,
                               lambda_reg) + model.l2_regularization()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += loss.item()
    if scheduler:
        scheduler.step()
    return total / len(loader)


def build_model(hp, feature_dim):
    return EvidentialGasConditionedMPNN(
        node_features=7, edge_features=7, gas_features=feature_dim,
        hidden_dim    = hp['hidden_dim'],
        num_mp_layers = hp['num_mp_layers'],
        fusion_dim    = hp['fusion_dim'],
        l2_lambda     = hp['l2_lambda'],
        dropout       = hp['dropout'],
        pooling       = hp['pooling'],
    ).to(device)


@torch.no_grad()
def predict_nig(model, loader):
    """
    Returns raw NIG params (gamma, nu, alpha, beta) and scaled targets
    - all still in the StandardScaler-transformed space.
    """
    model.eval()
    G, N, A, B, Y = [], [], [], [], []
    for data in loader:
        data = data.to(device)
        g, n, a, b = model(data)
        G.extend(g.cpu().numpy()); N.extend(n.cpu().numpy())
        A.extend(a.cpu().numpy()); B.extend(b.cpu().numpy())
        Y.extend(data.y.squeeze(-1).cpu().numpy())
    return (np.array(G), np.array(N), np.array(A),
            np.array(B), np.array(Y))


def compute_picp(gamma, nu, alpha, beta, y_true_scaled, y_sc,
                 z: float = 1.96) -> float:
    """
    Compute PICP in the original (log10-Barrer) scale.

    sigma_tot in original scale = sqrt(var_al + var_ep) * target_std
    where target_std = y_sc.scale_[0]
    """
    target_std   = float(y_sc.scale_[0])
    var_al       = beta / (alpha - 1.0)
    var_ep       = beta / (nu * (alpha - 1.0))
    sigma_tot_sc = np.sqrt(np.clip(var_al + var_ep, 0, None))
    sigma_orig   = sigma_tot_sc * target_std

    y_pred_orig  = y_sc.inverse_transform(gamma.reshape(-1, 1)).flatten()
    y_true_orig  = y_sc.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()

    within = (y_true_orig >= y_pred_orig - z * sigma_orig) & \
             (y_true_orig <= y_pred_orig + z * sigma_orig)
    return float(np.mean(within))


def train_fold(train_raw, val_raw, hp, feature_dim,
               lambda_reg, max_epochs, patience, seed_offset=0):
    """
    Train one evidential GC-MPNN fold and return PICP on val_raw.
    seed_offset lets each fold get a reproducible but distinct seed.
    """
    set_seeds(SEED + seed_offset)

    train_y, val_y, y_sc = scale_targets(train_raw, val_raw)
    train_sc, val_sc, _, _, _ = standardize_features(train_y, val_y)

    train_loader = DataLoader(train_sc, batch_size=hp['batch_size'],
                              shuffle=True,  num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY)
    val_loader   = DataLoader(val_sc,   batch_size=hp['batch_size'],
                              shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY)

    model     = build_model(hp, feature_dim)
    optimizer = optim.Adam(model.parameters(), lr=hp['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_loss  = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(max_epochs):
        tr_loss = train_epoch(model, train_loader, optimizer, scheduler, lambda_reg)
        if tr_loss < best_loss:
            best_loss  = tr_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    gamma, nu, alpha, beta, y_sc_vals = predict_nig(model, val_loader)
    picp = compute_picp(gamma, nu, alpha, beta, y_sc_vals, y_sc)
    return picp

# STAGE 0 - LOGO-BASED LAMBDA CALIBRATION (bisection)
def logo_mean_picp(lambda_reg: float,
                   graphs_by_gas: dict,
                   hp: dict,
                   feature_dim: int,
                   verbose: bool = True) -> float:
    """
    Run 5-fold LOGO rotation on TRAINING_POOL and return mean PICP.
    TEST_GAS is never touched here.
    """
    picps = []
    for fold_idx, val_gas in enumerate(TRAINING_POOL):
        train_raw = []
        for g_name in TRAINING_POOL:
            if g_name != val_gas:
                train_raw.extend(graphs_by_gas[g_name])
        val_raw = graphs_by_gas[val_gas]

        picp = train_fold(
            train_raw, val_raw, hp, feature_dim,
            lambda_reg  = lambda_reg,
            max_epochs  = CAL_EPOCHS,
            patience    = CAL_PATIENCE,
            seed_offset = fold_idx,        # each fold gets its own seed
        )
        picps.append(picp)
        if verbose:
            print(f"      val_gas={val_gas:<4}  PICP={picp:.4f}")

    mean_picp = float(np.mean(picps))
    if verbose:
        print(f"    -> mean PICP = {mean_picp:.4f}  (lambda_reg={lambda_reg:.6f})")
    return mean_picp


def calibrate_lambda(graphs_by_gas: dict,
                     hp: dict,
                     feature_dim: int) -> tuple:
    """
    Bisection search for lambda_reg* such that
    |logo_mean_picp(lambda_reg*) - TARGET_PICP| < PICP_TOL.

    Returns (lambda_star, calibration_log)
    where calibration_log is a list of dicts for CSV export.
    """
    print(f"\n{''*60}")
    print(f"Stage 0 - lambda_reg calibration  ('{TEST_GAS}' locked away)")
    print(f"Target PICP : {TARGET_PICP}  +/-  {PICP_TOL}")
    print(f"Search range: [{LAMBDA_LOW}, {LAMBDA_HIGH}]")
    print(f"Max bisection iterations: {MAX_BISECT_ITERS}")
    print(f"{''*60}")

    lo, hi = LAMBDA_LOW, LAMBDA_HIGH
    cal_log = []

    # Step 1: verify the bounds bracket the target
    print(f"\n  Checking lower bound  lambda={lo:.6f} ...")
    picp_lo = logo_mean_picp(lo, graphs_by_gas, hp, feature_dim)
    cal_log.append({'iteration': 0, 'lambda_reg': lo,
                    'mean_picp': picp_lo, 'bracket': 'low'})

    print(f"\n  Checking upper bound  lambda={hi:.6f} ...")
    picp_hi = logo_mean_picp(hi, graphs_by_gas, hp, feature_dim)
    cal_log.append({'iteration': 0, 'lambda_reg': hi,
                    'mean_picp': picp_hi, 'bracket': 'high'})

    if picp_lo >= TARGET_PICP:
        print(f"\n  WARNING: lower bound already achieves PICP={picp_lo:.4f} "
              f">= {TARGET_PICP}. Returning lambda_low={lo}.")
        print("  Consider decreasing LAMBDA_LOW for a tighter bound.")
        return lo, cal_log

    if picp_hi <= TARGET_PICP:
        print(f"\n  WARNING: upper bound only achieves PICP={picp_hi:.4f} "
              f"<= {TARGET_PICP}. Returning lambda_high={hi}.")
        print("  Consider increasing LAMBDA_HIGH for a wider bound.")
        return hi, cal_log

    # Step 2: bisection
    print(f"\n  Bounds verified: PICP({lo:.1e})={picp_lo:.4f} < {TARGET_PICP} "
          f"< {picp_hi:.4f}=PICP({hi:.1e})")
    print(f"  Starting bisection...\n")

    lambda_star = (lo + hi) / 2.0

    for i in range(1, MAX_BISECT_ITERS + 1):
        mid = (lo + hi) / 2.0
        print(f"  Iter {i:02d}/{MAX_BISECT_ITERS}  "
              f"bracket=[{lo:.6f}, {hi:.6f}]  mid={mid:.6f}")

        picp_mid = logo_mean_picp(mid, graphs_by_gas, hp, feature_dim)
        cal_log.append({'iteration': i, 'lambda_reg': mid,
                        'mean_picp': picp_mid, 'bracket': 'mid'})

        if abs(picp_mid - TARGET_PICP) < PICP_TOL:
            lambda_star = mid
            print(f"\n   Converged at iter {i}: "
                  f"lambda_reg*={mid:.6f}  PICP={picp_mid:.4f} "
                  f"(|error|={abs(picp_mid-TARGET_PICP):.4f} < {PICP_TOL})")
            break

        # PICP increases with lambda_reg
        if picp_mid < TARGET_PICP:
            lo = mid      # need higher lambda to widen intervals
        else:
            hi = mid      # need lower lambda to narrow intervals

        lambda_star = mid
    else:
        print(f"\n  Bisection reached max iterations ({MAX_BISECT_ITERS}).")
        print(f"  Best lambda_reg = {lambda_star:.6f}  "
              f"PICP = {picp_mid:.4f}")

    return lambda_star, cal_log

# STAGE 1 - FINAL TRAINING ON FULL POOL WITH LAMBDA*
def final_train(pool_graphs, test_graphs, hp, feature_dim, lambda_reg):
    """
    Train on all TRAINING_POOL gases with calibrated lambda_reg.
    Returns model, y_sc, node/edge/gas scalers for inference.
    """
    print(f"\n{''*60}")
    print(f"Stage 1 - Final training  (lambda_reg*={lambda_reg:.6f})")
    print(f"  Pool: {len(pool_graphs)} graphs  |  "
          f"Test ({TEST_GAS}): {len(test_graphs)} graphs  [still locked]")
    print(f"{''*60}")

    set_seeds(SEED)
    train_y, test_y, y_sc = scale_targets(pool_graphs, test_graphs)
    train_sc, test_sc, node_sc, edge_sc, gas_sc = \
        standardize_features(train_y, test_y)

    train_loader = DataLoader(train_sc, batch_size=hp['batch_size'],
                              shuffle=True,  num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY)
    test_loader  = DataLoader(test_sc,  batch_size=hp['batch_size'],
                              shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY)

    model     = build_model(hp, feature_dim)
    optimizer = optim.Adam(model.parameters(), lr=hp['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=FINAL_EPOCHS)

    best_loss  = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(FINAL_EPOCHS):
        tr_loss = train_epoch(model, train_loader, optimizer,
                              scheduler, lambda_reg)
        if tr_loss < best_loss:
            best_loss  = tr_loss
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1:>4}  loss={tr_loss:.4f}  best={best_loss:.4f}")
        if no_improve >= FINAL_PATIENCE:
            print(f"  Early stop at epoch {epoch+1}")
            break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"  Training complete. Best loss: {best_loss:.4f}")

    return model, y_sc, node_sc, edge_sc, gas_sc, train_loader, test_loader

# STAGE 2 - EVALUATE ON TEST GAS
@torch.no_grad()
def evaluate_full(model, loader, y_sc):
    """
    Full evaluation: returns predictions and decomposed uncertainties
    in the original log10-Barrer scale.
    """
    target_std = float(y_sc.scale_[0])
    model.eval()
    G, N, A, B, Y = [], [], [], [], []
    for data in loader:
        data = data.to(device)
        g, n, a, b = model(data)
        G.extend(g.cpu().numpy()); N.extend(n.cpu().numpy())
        A.extend(a.cpu().numpy()); B.extend(b.cpu().numpy())
        Y.extend(data.y.squeeze(-1).cpu().numpy())

    gamma = np.array(G); nu    = np.array(N)
    alpha = np.array(A); beta  = np.array(B)
    y_sc_ = np.array(Y)

    y_pred = y_sc.inverse_transform(gamma.reshape(-1,1)).flatten()
    y_true = y_sc.inverse_transform(y_sc_.reshape(-1,1)).flatten()

    aleatoric_std = np.sqrt(np.clip(beta / (alpha - 1.0),           0, None)) * target_std
    epistemic_std = np.sqrt(np.clip(beta / (nu * (alpha - 1.0)),    0, None)) * target_std
    total_std     = np.sqrt(np.clip(aleatoric_std**2 + epistemic_std**2, 0, None))

    return dict(y_true=y_true, y_pred=y_pred,
                aleatoric_std=aleatoric_std,
                epistemic_std=epistemic_std,
                total_std=total_std,
                nig_gamma=gamma, nig_nu=nu,
                nig_alpha=alpha, nig_beta=beta)


def compute_metrics(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return {
        'r2':   float(1 - ss_res/ss_tot) if ss_tot > 0 else 0.,
        'rmse': float(np.sqrt(np.mean((y_pred-y_true)**2))),
        'mae':  float(np.mean(np.abs(y_pred-y_true))),
    }


def calibration_report(y_true, y_pred, total_std, label=''):
    ci_lo  = y_pred - 1.96 * total_std
    ci_hi  = y_pred + 1.96 * total_std
    within = (y_true >= ci_lo) & (y_true <= ci_hi)
    picp   = float(np.mean(within))
    mpiw   = float(np.mean(ci_hi - ci_lo))
    rng    = float(np.max(y_true) - np.min(y_true))
    nmpiw  = mpiw / rng if rng > 0 else float('nan')
    print(f"  {label}PICP_95 = {picp:.4f}  "
          f"MPIW = {mpiw:.4f}  NMPIW = {nmpiw:.4f}")
    return picp, mpiw, nmpiw, ci_lo, ci_hi, within

# MAIN
def main():
    print('\n' + '='*70)
    print('GC-MPNN - EVIDENTIAL REGRESSION + LOGO LAMBDA CALIBRATION')
    print(f'Experiment       : {EXPERIMENT_NAME}')
    print(f'Test gas (locked): {TEST_GAS}')
    print(f'Training pool    : {TRAINING_POOL}')
    print(f'Target PICP      : {TARGET_PICP}  (tol={PICP_TOL})')
    print('='*70)

    hp          = load_hyperparameters(PARAMS_JSON)
    feature_dim = EXPERIMENT_CONFIGS[EXPERIMENT_NAME]['feature_dim']
    print(f'\nGas feature dimension: {feature_dim}')

    # Build datasets
    print('\nBuilding datasets...')
    smiles, p_exp_map = load_data(DATA_CSV)
    pool_graphs = build_pyg_dataset(
        create_dataset(smiles, p_exp_map, EXPERIMENT_NAME, TRAINING_POOL))
    test_graphs = build_pyg_dataset(
        create_dataset(smiles, p_exp_map, EXPERIMENT_NAME, [TEST_GAS]))
    print(f'  Pool ({"+".join(TRAINING_POOL)}): {len(pool_graphs)} graphs')
    print(f'  Test ({TEST_GAS})            : {len(test_graphs)} graphs  [LOCKED]')

    graphs_by_gas = {}
    for g in pool_graphs:
        graphs_by_gas.setdefault(g.gas_name, []).append(g)
    print('  Pool per gas: ' +
          '  '.join(f'{g}={len(graphs_by_gas[g])}' for g in TRAINING_POOL))

    # Stage 0: calibrate lambda_reg
    lambda_star, cal_log = calibrate_lambda(
        graphs_by_gas, hp, feature_dim)

    # Save bisection history
    cal_df = pd.DataFrame(cal_log)
    cal_csv = f'{EXPERIMENT_NAME}_lambda_calibration_log.csv'
    cal_df.to_csv(cal_csv, index=False)
    print(f'\nSaved calibration log: {cal_csv}')
    print(f'\n{""*70}')
    print(f'Calibrated lambda_reg* = {lambda_star:.6f}')
    print(f'{""*70}')

    # Stage 1: final training
    model, y_sc, node_sc, edge_sc, gas_sc, train_loader, test_loader = \
        final_train(pool_graphs, test_graphs, hp, feature_dim, lambda_star)

    # Stage 2: evaluate on test gas
    print(f"\n{''*60}")
    print(f"Stage 2 - Test evaluation  ('{TEST_GAS}' unlocked)")
    print(f"{''*60}")

    test_out  = evaluate_full(model, test_loader, y_sc)
    pool_out  = evaluate_full(model, train_loader, y_sc)

    test_met  = compute_metrics(test_out['y_true'], test_out['y_pred'])
    pool_met  = compute_metrics(pool_out['y_true'], pool_out['y_pred'])

    print(f'\nTest ({TEST_GAS}):')
    print(f'  R2   = {test_met["r2"]:.4f}')
    print(f'  RMSE = {test_met["rmse"]:.4f}  (log10 Barrer)')
    print(f'  MAE  = {test_met["mae"]:.4f}  (log10 Barrer)')
    print(f'\nUncertainty (test set):')
    print(f'  Mean aleatoric sigma : {test_out["aleatoric_std"].mean():.4f}')
    print(f'  Mean epistemic sigma : {test_out["epistemic_std"].mean():.4f}')
    print(f'  Mean total sigma     : {test_out["total_std"].mean():.4f}')
    print(f'\nCalibration on test set (using calibrated lambda_reg*):')
    picp, mpiw, nmpiw, ci_lo, ci_hi, within = \
        calibration_report(test_out['y_true'], test_out['y_pred'],
                           test_out['total_std'], label='test  ')
    print(f'\nCalibration on train pool (in-distribution reference):')
    picp_pool, mpiw_pool, nmpiw_pool, *_ = \
        calibration_report(pool_out['y_true'], pool_out['y_pred'],
                           pool_out['total_std'], label='pool  ')
    print(f'\nTrain pool:  R2={pool_met["r2"]:.4f}  RMSE={pool_met["rmse"]:.4f}')

    # Save test CSV
    test_df = pd.DataFrame({
        'smiles':        [g.smiles for g in test_graphs],
        'y_true':        test_out['y_true'],
        'y_pred':        test_out['y_pred'],
        'aleatoric_std': test_out['aleatoric_std'],
        'epistemic_std': test_out['epistemic_std'],
        'total_std':     test_out['total_std'],
        'ci_lower_95':   ci_lo,
        'ci_upper_95':   ci_hi,
        'within_ci':     within.astype(int),
        'residual':      test_out['y_pred'] - test_out['y_true'],
        'nig_gamma':     test_out['nig_gamma'],
        'nig_nu':        test_out['nig_nu'],
        'nig_alpha':     test_out['nig_alpha'],
        'nig_beta':      test_out['nig_beta'],
        'gas':           TEST_GAS,
        'experiment':    EXPERIMENT_NAME,
    })
    csv_test = f'{EXPERIMENT_NAME}_evidential_test_{TEST_GAS}_predictions.csv'
    test_df.to_csv(csv_test, index=False)
    print(f'\nSaved: {csv_test}')

    # Save pool CSV
    ci_lo_p = pool_out['y_pred'] - 1.96 * pool_out['total_std']
    ci_hi_p = pool_out['y_pred'] + 1.96 * pool_out['total_std']
    pool_df = pd.DataFrame({
        'smiles':        [g.smiles for g in pool_graphs],
        'y_true':        pool_out['y_true'],
        'y_pred':        pool_out['y_pred'],
        'aleatoric_std': pool_out['aleatoric_std'],
        'epistemic_std': pool_out['epistemic_std'],
        'total_std':     pool_out['total_std'],
        'ci_lower_95':   ci_lo_p,
        'ci_upper_95':   ci_hi_p,
        'within_ci':     ((pool_out['y_true'] >= ci_lo_p) &
                          (pool_out['y_true'] <= ci_hi_p)).astype(int),
        'residual':      pool_out['y_pred'] - pool_out['y_true'],
        'nig_gamma':     pool_out['nig_gamma'],
        'nig_nu':        pool_out['nig_nu'],
        'nig_alpha':     pool_out['nig_alpha'],
        'nig_beta':      pool_out['nig_beta'],
        'split':         'train_pool',
        'experiment':    EXPERIMENT_NAME,
    })
    csv_pool = f'{EXPERIMENT_NAME}_evidential_train_pool_predictions.csv'
    pool_df.to_csv(csv_pool, index=False)
    print(f'Saved: {csv_pool}')

    # Save summary JSON
    summary = {
        'experiment':        EXPERIMENT_NAME,
        'test_gas':          TEST_GAS,
        'lambda_reg_star':   lambda_star,
        'lambda_search':     {'low': LAMBDA_LOW, 'high': LAMBDA_HIGH,
                              'target_picp': TARGET_PICP, 'tol': PICP_TOL},
        'hyperparameters':   hp,
        'test_metrics':      test_met,
        'pool_metrics':      pool_met,
        'test_calibration':  {'picp_95': picp,  'mpiw_95': mpiw,
                              'nmpiw_95': nmpiw},
        'pool_calibration':  {'picp_95': picp_pool, 'mpiw_95': mpiw_pool,
                              'nmpiw_95': nmpiw_pool},
        'uncertainty_summary': {
            'mean_aleatoric_std': float(test_out['aleatoric_std'].mean()),
            'mean_epistemic_std': float(test_out['epistemic_std'].mean()),
            'mean_total_std':     float(test_out['total_std'].mean()),
        },
        'bisection_iters':   len([r for r in cal_log if r['bracket'] == 'mid']),
    }
    json_out = f'{EXPERIMENT_NAME}_evidential_summary.json'
    with open(json_out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved: {json_out}')

    print('\n' + '='*70)
    print('COMPLETE')
    print(f'  lambda_reg*    = {lambda_star:.6f}')
    print(f'  Test PICP_95   = {picp:.4f}  (calibrated target: {TARGET_PICP})')
    print(f'  Test R2        = {test_met["r2"]:.4f}')
    print(f'  Test RMSE      = {test_met["rmse"]:.4f}  (log10 Barrer)')
    print('='*70)


if __name__ == '__main__':
    main()
