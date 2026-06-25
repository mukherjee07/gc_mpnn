#!/usr/bin/env python
# coding: utf-8

"""
Gas-Conditioned MPNN — Multi-Descriptor Edition
================================================
Select a gas descriptor category by setting DESCRIPTOR_CATEGORY (1–6):

  1  →  Thermo       : [Tc, Pc, omega, sigma, epsilon]          dim = 5
  2  →  Kinetic      : [d, Vd]                                  dim = 2
  3  →  Electrostatic: [q+, q-]                                 dim = 2
  4  →  Total        : Thermo + Kinetic + Electrostatic         dim = 9
  5  →  OHE          : one-hot over 6 gases                     dim = 6
  6  →  Thermo+Kin   : [sigma, epsilon, omega, Tc, Pc, d, Vd]   dim = 7
                       (original implementation)

Everything else (model, scalers, LOGO loop) adjusts automatically.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from rdkit import Chem
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (MessagePassing, global_mean_pool,
                                 global_add_pool, GlobalAttention)

# ============================================================
# USER SETTING 
# ============================================================
DESCRIPTOR_CATEGORY = 6   # 

# ============================================================
# Device
# ============================================================
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon)")
else:
    device = torch.device("cpu")
    print("Using CPU")

NUM_WORKERS = 2 if device.type == "cuda" else 0
PIN_MEMORY  = device.type == "cuda"

# ============================================================
# Fixed hyperparameters
# ============================================================
BEST_PARAMS = {
    "learning_rate": 0.0004989483263215937,
    "l2_lambda":     0.0045419658909029055,
    "hidden_dim":    256,
    "num_mp_layers": 3,
    "fusion_dim":    16,
    "dropout":       0.013135488457537477,
    "pooling":       "mean",
    "batch_size":    128,
}

FINAL_EPOCHS   = 500
FINAL_PATIENCE = 100

# ============================================================
# Gas properties
# Note: electrostatic partial charges (q+, q-) are approximate
# ============================================================
GAS_PROPERTIES = {
    'He':  {'sigma': 2.551, 'epsilon': 10.2,  'omega': -0.383,
            'Tc':   5.2,   'Pc': 2.28,
            'd': 2.6,  'Vd': 2.67,
            'q+': 0.00, 'q-': 0.00},
    'H2':  {'sigma': 2.827, 'epsilon': 59.7,  'omega': -0.265,
            'Tc':  33.2,   'Pc': 13.00,
            'd': 2.89, 'Vd': 6.12,
            'q+': 0.18, 'q-':-0.18},
    'N2':  {'sigma': 3.798, 'epsilon': 71.4,  'omega':  0.037,
            'Tc': 126.2,   'Pc': 63.14,
            'd': 3.64, 'Vd': 18.5,
            'q+': 0.00, 'q-': 0.00},
    'O2':  {'sigma': 3.467, 'epsilon': 106.7, 'omega':  0.022,
            'Tc': 154.6,   'Pc': 50.43,
            'd': 3.46, 'Vd': 16.3,
            'q+': 0.00, 'q-': 0.00},
    'CH4': {'sigma': 3.758, 'epsilon': 148.6, 'omega':  0.011,
            'Tc': 190.6,   'Pc': 46.1,
            'd': 3.8,  'Vd': 24.42,
            'q+': 0.00, 'q-': 0.00},
    'CO2': {'sigma': 3.941, 'epsilon': 195.2, 'omega':  0.253,
            'Tc': 304.1,   'Pc': 73.80,
            'd': 3.3,  'Vd': 26.9,
            'q+': 0.65, 'q-':-0.33},
}

GASES     = ['He', 'H2', 'N2', 'O2', 'CH4', 'CO2']
GAS_INDEX = {g: i for i, g in enumerate(GASES)}   # used for OHE

# ============================================================
# Descriptor registry
# ============================================================

def _feats_thermo(gas):
    p = GAS_PROPERTIES[gas]
    return np.array([p['Tc'], p['Pc'], p['omega'], p['sigma'], p['epsilon']],
                    dtype=np.float32)

def _feats_kinetic(gas):
    p = GAS_PROPERTIES[gas]
    return np.array([p['d'], p['Vd']], dtype=np.float32)

def _feats_electrostatic(gas):
    p = GAS_PROPERTIES[gas]
    return np.array([p['q+'], p['q-']], dtype=np.float32)

def _feats_total(gas):
    return np.concatenate([_feats_thermo(gas),
                           _feats_kinetic(gas),
                           _feats_electrostatic(gas)])

def _feats_ohe(gas):
    vec = np.zeros(len(GASES), dtype=np.float32)
    vec[GAS_INDEX[gas]] = 1.0
    return vec

def _feats_thermo_kinetic(gas):
    p = GAS_PROPERTIES[gas]
    return np.array([p['sigma'], p['epsilon'], p['omega'],
                     p['Tc'], p['Pc'], p['d'], p['Vd']],
                    dtype=np.float32)

# Maps category number --> (function, feature_dim)
DESCRIPTOR_REGISTRY = {
    1: (_feats_thermo,         5,  "Thermo [Tc, Pc, omega, sigma, epsilon]"),
    2: (_feats_kinetic,        2,  "Kinetic [d, Vd]"),
    3: (_feats_electrostatic,  2,  "Electrostatic [q+, q-]"),
    4: (_feats_total,          9,  "Total [Thermo + Kinetic + Electrostatic]"),
    5: (_feats_ohe,            6,  "OHE [one-hot over 6 gases]"),
    6: (_feats_thermo_kinetic, 7,  "Thermo+Kinetic [sigma, epsilon, omega, Tc, Pc, d, Vd]"),
}

assert DESCRIPTOR_CATEGORY in DESCRIPTOR_REGISTRY, \
    f"DESCRIPTOR_CATEGORY must be 1–6, got {DESCRIPTOR_CATEGORY}"

_get_gas_features, FEATURE_DIM, _DESC_NAME = DESCRIPTOR_REGISTRY[DESCRIPTOR_CATEGORY]

print(f"\n{'='*60}")
print(f"  Descriptor category : {DESCRIPTOR_CATEGORY}")
print(f"  Description         : {_DESC_NAME}")
print(f"  Feature dimension   : {FEATURE_DIM}")
print(f"{'='*60}\n")

# ============================================================
# Seed helper
# ============================================================
def set_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seeds(42)

# ============================================================
# Graph construction
# ============================================================
def smiles_to_graph(smiles_str):
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        return None

    node_features = []
    for atom in mol.GetAtoms():
        node_features.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetHybridization().real,
            int(atom.GetIsAromatic()),
            atom.GetTotalNumHs(),
            atom.GetNumImplicitHs(),
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
            int(bond.GetIsConjugated()),
            int(bond.IsInRing()),
            int(bond.GetBondType() == Chem.BondType.SINGLE
                and not bond.IsInRing()
                and bond.GetBeginAtom().GetDegree() > 1
                and bond.GetEndAtom().GetDegree() > 1),
        ]
        edge_features.extend([ef, ef])

    if not edge_indices:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, 7), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(edge_features, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def build_dataset(pol_sd):
    cols = {
        'CH4': 'p_exp_CH4', 'CO2': 'p_exp_CO2',
        'H2':  'p_exp_H2',  'N2':  'p_exp_N2',
        'O2':  'p_exp_O2',  'He':  'p_exp_He',
    }
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
            g = graph.clone()
            # ← _get_gas_features auto-selected by DESCRIPTOR_CATEGORY
            g.gas_features = torch.tensor(
                _get_gas_features(gas), dtype=torch.float
            ).unsqueeze(0)
            g.y        = torch.tensor([perm], dtype=torch.float)
            g.gas_name = gas
            g.smiles   = smi
            dataset.append(g)
    return dataset

# ============================================================
# Polymer train coverage count
# ============================================================
def build_polymer_train_counts(all_graphs):
    train_counts = {}
    for held_out_gas in GASES:
        counts = {}
        for g in all_graphs:
            if g.gas_name == held_out_gas:
                continue
            counts[g.smiles] = counts.get(g.smiles, 0) + 1
        train_counts[held_out_gas] = counts
    return train_counts

# ============================================================
# Model
# ============================================================
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
    def __init__(self, node_features=7, edge_features=7,
                 gas_features=FEATURE_DIM,   # ← auto-set from registry
                 hidden_dim=256, num_mp_layers=3, fusion_dim=16,
                 l2_lambda=0.0045419658909029055,
                 dropout=0.013135488457537477, pooling='mean'):
        super().__init__()
        self.l2_lambda        = l2_lambda
        self.expected_gas_dim = gas_features
        self.pooling_type     = pooling

        self.node_embedding = nn.Linear(node_features, hidden_dim)
        self.mp_layers  = nn.ModuleList([
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
            nn.Linear(hidden_dim + fusion_dim, 256), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64, 1))

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch)
        gas_features = data.gas_features
        batch_size   = batch.max().item() + 1

        if gas_features.shape != (batch_size, self.expected_gas_dim):
            raise ValueError(
                f"gas_features shape mismatch: {gas_features.shape} "
                f"vs [{batch_size}, {self.expected_gas_dim}]")

        x = F.relu(self.node_embedding(x))
        for mp_layer, bn in zip(self.mp_layers, self.batch_norms):
            x_new = self.dropout(F.relu(bn(mp_layer(x, edge_index, edge_attr))))
            x = x + x_new

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
        l2 = torch.tensor(0.0, device=next(self.parameters()).device)
        for p in self.parameters():
            l2 += torch.norm(p, 2)
        return self.l2_lambda * l2

# ============================================================
# Scaling
# ============================================================
def scale_graphs_with_existing_scalers(graphs, node_scaler, edge_scaler,
                                        gas_scaler):
    scaled = []
    for g in graphs:
        gc    = g.clone()
        gc.x  = torch.tensor(node_scaler.transform(g.x.numpy()),
                              dtype=torch.float)
        gas_np = g.gas_features.squeeze(0).numpy().reshape(1, -1)
        gc.gas_features = torch.tensor(gas_scaler.transform(gas_np),
                                        dtype=torch.float)
        if g.edge_attr.shape[0] > 0 and edge_scaler is not None:
            gc.edge_attr = torch.tensor(
                edge_scaler.transform(g.edge_attr.numpy()), dtype=torch.float)
        scaled.append(gc)
    return scaled


def standardize_features(train_graphs, val_graphs):
    node_scaler = StandardScaler()
    gas_scaler  = StandardScaler()
    edge_scaler = None

    node_scaler.fit(np.vstack([g.x.numpy() for g in train_graphs]))
    gas_scaler.fit(np.array([g.gas_features.squeeze(0).numpy()
                              for g in train_graphs]))
    edge_arrays = [g.edge_attr.numpy() for g in train_graphs
                   if g.edge_attr.shape[0] > 0]
    if edge_arrays:
        edge_scaler = StandardScaler()
        edge_scaler.fit(np.vstack(edge_arrays))

    train_scaled = scale_graphs_with_existing_scalers(
        train_graphs, node_scaler, edge_scaler, gas_scaler)
    val_scaled   = scale_graphs_with_existing_scalers(
        val_graphs,   node_scaler, edge_scaler, gas_scaler)
    return train_scaled, val_scaled, node_scaler, edge_scaler, gas_scaler


def precompute_logo_splits(all_graphs):
    splits = {}
    for test_gas in GASES:
        train_raw = [g for g in all_graphs if g.gas_name != test_gas]
        test_raw  = [g for g in all_graphs if g.gas_name == test_gas]
        if not test_raw:
            continue

        y_scaler = StandardScaler()
        y_tr = y_scaler.fit_transform(
            np.array([g.y.item() for g in train_raw]).reshape(-1, 1)).flatten()
        y_te = y_scaler.transform(
            np.array([g.y.item() for g in test_raw]).reshape(-1, 1)).flatten()

        def _apply_y(graphs, y_scaled):
            out = []
            for i, g in enumerate(graphs):
                gc = g.clone()
                gc.y = torch.tensor([y_scaled[i]], dtype=torch.float)
                out.append(gc)
            return out

        tr_scaled, te_scaled, ns, es, gs = standardize_features(
            _apply_y(train_raw, y_tr), _apply_y(test_raw, y_te))

        splits[test_gas] = dict(
            train_graphs=tr_scaled, test_graphs=te_scaled,
            y_scaler=y_scaler, node_scaler=ns,
            edge_scaler=es, gas_scaler=gs,
            n_train=len(train_raw), n_test=len(test_raw))
    return splits


def prepare_eval_graphs(eval_graphs_raw, split):
    y_raw    = np.array([g.y.item() for g in eval_graphs_raw])
    y_scaled = split["y_scaler"].transform(y_raw.reshape(-1, 1)).flatten()
    ev_y = []
    for i, g in enumerate(eval_graphs_raw):
        gc = g.clone()
        gc.y = torch.tensor([y_scaled[i]], dtype=torch.float)
        ev_y.append(gc)
    return scale_graphs_with_existing_scalers(
        ev_y, split["node_scaler"], split["edge_scaler"], split["gas_scaler"])

# ============================================================
# Training / evaluation
# ============================================================
def train_epoch(model, loader, criterion, optimizer, scheduler=None):
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = (criterion(model(data), data.y.view(-1))
                + model.l2_regularization())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if scheduler is not None:
        scheduler.step()
    return total_loss / len(loader)


def predict(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            preds.append(model(data).cpu().numpy())
            targets.append(data.y.view(-1).cpu().numpy())
    return np.concatenate(preds), np.concatenate(targets)


def train_single_model(train_graphs):
    loader = DataLoader(train_graphs, batch_size=BEST_PARAMS["batch_size"],
                        shuffle=True, num_workers=NUM_WORKERS,
                        pin_memory=PIN_MEMORY)
    model = GasConditionedMPNN(
        node_features  = 7,
        edge_features  = 7,
        gas_features   = FEATURE_DIM,        #  auto from registry
        hidden_dim     = BEST_PARAMS["hidden_dim"],
        num_mp_layers  = BEST_PARAMS["num_mp_layers"],
        fusion_dim     = BEST_PARAMS["fusion_dim"],
        l2_lambda      = BEST_PARAMS["l2_lambda"],
        dropout        = BEST_PARAMS["dropout"],
        pooling        = BEST_PARAMS["pooling"],
    ).to(device)

    criterion  = nn.MSELoss()
    optimizer  = optim.Adam(model.parameters(), lr=BEST_PARAMS["learning_rate"])
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=FINAL_EPOCHS)

    best_loss, patience_counter, best_state = float("inf"), 0, None
    for epoch in range(FINAL_EPOCHS):
        loss = train_epoch(model, loader, criterion, optimizer, scheduler)
        if loss < best_loss:
            best_loss, patience_counter = loss, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        if (epoch + 1) % 100 == 0:
            print(f"    Epoch {epoch+1:4d} | Loss = {loss:.4f}")
        if patience_counter >= FINAL_PATIENCE:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    return model

# ============================================================
# CSV saving
# ============================================================
def save_prediction_csv(logo_results, all_graphs, polymer_train_counts,
                         output_path="entropy_per_pair.csv"):
    rows = []
    for held_out_gas, result in logo_results.items():
        y_pred_dict   = result["y_pred_mean"]
        y_true_dict   = result["y_true_per_gas"]
        counts        = polymer_train_counts[held_out_gas]
        for eval_gas in GASES:
            if eval_gas not in y_pred_dict:
                continue
            eval_graphs = [g for g in all_graphs if g.gas_name == eval_gas]
            y_pred = y_pred_dict[eval_gas]
            y_true = y_true_dict[eval_gas]
            mse    = (y_pred - y_true) ** 2
            for i, g in enumerate(eval_graphs):
                rows.append(dict(
                    smiles                = g.smiles,
                    polymer_idx           = i,
                    eval_gas              = eval_gas,
                    fold_held_out         = held_out_gas,
                    is_ood                = eval_gas == held_out_gas,
                    polymer_train_gas_count = counts.get(g.smiles, 0),
                    y_true                = float(y_true[i]),
                    y_pred_mean           = float(y_pred[i]),
                    mse                   = float(mse[i]),
                    descriptor_category   = DESCRIPTOR_CATEGORY,
                    descriptor_name       = _DESC_NAME,
                ))

    df = pd.DataFrame(rows)[[
        "smiles", "polymer_idx", "eval_gas", "fold_held_out",
        "is_ood", "polymer_train_gas_count",
        "y_true", "y_pred_mean", "mse",
        "descriptor_category", "descriptor_name",
    ]]
    df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    print(f"Total rows : {len(df):,}")
    print(f"OOD rows   : {df['is_ood'].sum():,}")
    print(df.head(4).to_string(index=False))
    return df

# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("GC-MPNN LOGO PREDICTION — MULTI-DESCRIPTOR EDITION")
    print(f"Descriptor : {DESCRIPTOR_CATEGORY} — {_DESC_NAME}")
    print(f"Feature dim: {FEATURE_DIM}")
    print("=" * 60)

    print("\nLoading dataset …")
    pol_sd     = pd.read_csv("Gas_permeability_solubility_diffusivity_wide.csv")
    all_graphs = build_dataset(pol_sd)
    print(f"Loaded {len(all_graphs)} polymer-gas pairs")

    polymer_train_counts = build_polymer_train_counts(all_graphs)

    print("\nPrecomputing LOGO splits …")
    splits = precompute_logo_splits(all_graphs)
    print(f"Prepared {len(splits)} LOGO folds")

    logo_results = {}
    for held_out_gas in GASES:
        if held_out_gas not in splits:
            continue
        print("\n" + "=" * 70)
        print(f"Held-out gas : {held_out_gas}")
        split = splits[held_out_gas]
        print(f"Train: {split['n_train']}  |  Test: {split['n_test']}")

        model = train_single_model(split["train_graphs"])

        fold_pred, fold_true = {}, {}
        for eval_gas in GASES:
            eval_raw = [g for g in all_graphs if g.gas_name == eval_gas]
            if not eval_raw:
                continue
            eval_scaled = prepare_eval_graphs(eval_raw, split)
            loader = DataLoader(eval_scaled,
                                batch_size=BEST_PARAMS["batch_size"],
                                shuffle=False,
                                num_workers=NUM_WORKERS,
                                pin_memory=PIN_MEMORY)
            y_pred_s, y_true_s = predict(model, loader)
            y_scaler = split["y_scaler"]
            fold_pred[eval_gas] = y_scaler.inverse_transform(
                y_pred_s.reshape(-1, 1)).flatten()
            fold_true[eval_gas] = y_scaler.inverse_transform(
                y_true_s.reshape(-1, 1)).flatten()

        logo_results[held_out_gas] = dict(y_pred_mean=fold_pred,
                                          y_true_per_gas=fold_true)

    # Output filename encodes the descriptor category for easy comparison
    out_path = f"entropy_per_pair_desc{DESCRIPTOR_CATEGORY}.csv"
    save_prediction_csv(logo_results, all_graphs, polymer_train_counts,
                         output_path=out_path)


if __name__ == "__main__":
    main()
