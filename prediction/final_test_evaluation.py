#!/usr/bin/env python
# coding: utf-8
"""
GC-MPNN Pretrained Model - Inference on new_test_set.csv
=========================================================

Loads gc_mpnn_pretrained_checkpoint.pt and predicts log10-Barrer
permeability for 10 gases across polymers in new_test_set.csv.

Runs on every polymer (homopolymers and copolymers) in the CSV.

Copolymer blending
------------------
  log10(P_blend) = mf_1 * log10(P1) + mf_2 * log10(P2)

Ground truth units
------------------
  CSV values are in Barrer -> converted to log10 Barrer internally.

Metrics reported
----------------
  R2   (coefficient of determination)
  r    (Pearson correlation coefficient)
  rho    (Spearman rank correlation coefficient)
  RMSE (root mean squared error, log10 Barrer)
  MAE  (mean absolute error, log10 Barrer)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (MessagePassing, global_mean_pool,
                                 global_add_pool, GlobalAttention)
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from rdkit import Chem
import warnings

warnings.filterwarnings('ignore')

# SETTINGS
CHECKPOINT_PATH = 'gc_mpnn_pretrained_checkpoint.pt'
TEST_CSV        = '../data/new_test_set.csv'

OUTPUT_CSV      = 'inference_predictions_all.csv'

TARGET_GASES    = ['He', 'H2', 'N2', 'O2', 'CH4', 'CO2',
                   'C2H4', 'C2H6', 'C3H6', 'C3H8', 'n-C4H10']
OOD_GASES       = {'C2H4', 'C2H6', 'C3H6', 'C3H8', 'n-C4H10'}

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

# GAS PROPERTIES - Kinetic descriptors (d, Vd)
GAS_PROPERTIES = {
    'He':      {'d': 2.60,  'Vd':  2.67},
    'H2':      {'d': 2.89,  'Vd':  6.12},
    'N2':      {'d': 3.64,  'Vd': 18.50},
    'O2':      {'d': 3.46,  'Vd': 16.30},
    'CH4':     {'d': 3.80,  'Vd': 24.42},
    'CO2':     {'d': 3.30,  'Vd': 26.90},
    'C2H4':    {'d': 3.90,  'Vd': 41.04},
    'C2H6':    {'d': 4.44,  'Vd': 45.66},
    'C3H6':    {'d': 4.50,  'Vd': 61.56},
    'C3H8':    {'d': 4.30,  'Vd': 66.18},
    'n-C4H10': {'d': 4.30,  'Vd': 86.70},
}


def get_gas_features(gas: str) -> np.ndarray:
    p = GAS_PROPERTIES[gas]
    return np.array([p['d'], p['Vd']], dtype=np.float32)

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
    def __init__(self, node_features=7, edge_features=7, gas_features=2,
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
        g_emb = self.gas_encoder(data.gas_features)
        return self.fusion(torch.cat([p_emb, g_emb], dim=-1)).squeeze(-1)

# GRAPH CONSTRUCTION
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

# SMILES CANDIDATE VALIDATOR
def is_valid_smiles_candidate(s: str) -> bool:
    if not s or s.lower() == 'nan':
        return False
    try:
        float(s)
        return False
    except ValueError:
        return True

# APPLY SAVED SCALERS
def apply_scalers(graphs, node_sc, edge_sc, gas_sc):
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

# SINGLE-SMILES INFERENCE
@torch.no_grad()
def predict_smiles(smi: str, gas: str,
                   model, node_sc, edge_sc, gas_sc, y_sc):
    graph = smiles_to_graph(smi)
    if graph is None:
        return None
    graph.gas_features = torch.tensor(
        get_gas_features(gas), dtype=torch.float).unsqueeze(0)
    graph.y = torch.tensor([0.0], dtype=torch.float)
    scaled  = apply_scalers([graph], node_sc, edge_sc, gas_sc)
    loader  = DataLoader(scaled, batch_size=1, shuffle=False)
    model.eval()
    for data in loader:
        pred_sc = model(data.to(device)).cpu().numpy()
    return float(y_sc.inverse_transform(pred_sc.reshape(-1, 1)).flatten()[0])

# CSV READER
def read_test_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine='python',
                     header=0, index_col=None)
    df.columns = [str(c).strip() for c in df.columns]
    print(f"Loaded {path}: {len(df)} rows, {len(df.columns)} columns")
    print("Columns:", df.columns.tolist())
    return df

# COLUMN NAME RESOLVER
def resolve_columns(df: pd.DataFrame) -> dict:
    mapping = {}
    cols = df.columns.tolist()

    # Gas columns - match by first whitespace-split token
    for gas in TARGET_GASES + ['He', 'H2', 'O2', 'N2', 'CO2', 'CH4']:
        for col in cols:
            if str(col).strip().split()[0] == gas:
                mapping[gas] = col
                break

    # SMILES and mf columns
    for key in ['smiles_string_1', 'smiles_string_2', 'mf_1', 'mf_2']:
        for col in cols:
            if str(col).strip() == key or str(col).strip().startswith(key):
                mapping[key] = col
                break

    # Polymer name column
    for col in cols:
        c = str(col).strip()
        if c == 'Polymer' or (c.startswith('Polymer') and
                               'smiles' not in c.lower() and
                               len(c) < 20):
            mapping['polymer_name'] = col
            break
    if 'polymer_name' not in mapping:
        mapping['polymer_name'] = cols[2] if len(cols) > 2 else cols[0]

    # Polymer type column
    for col in cols:
        c = str(col).strip()
        if 'Polymer Type' in c or c == 'polymer_type':
            mapping['polymer_type'] = col
            break

    print("\nColumn mapping resolved:")
    for k, v in mapping.items():
        print(f"  {k:<20} -> '{v}'")

    return mapping

# POLYMER-TYPE CLASSIFIER
def classify_polymer(row, smi2_col) -> str:
    smi2_raw = row.get(smi2_col, np.nan) if smi2_col else np.nan
    return 'co' if is_valid_smiles_candidate(str(smi2_raw).strip()) else 'homo'

# METRICS  (R2, Pearson r, Spearman rho, RMSE, MAE)
def compute_metrics(yt: np.ndarray, yp: np.ndarray) -> dict:
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    rmse   = float(np.sqrt(np.mean((yp - yt) ** 2)))
    mae    = float(np.mean(np.abs(yp - yt)))
    r_p    = float(np.corrcoef(yt, yp)[0, 1]) if len(yt) > 1 else float('nan')
    rho, _ = spearmanr(yt, yp)  if len(yt) > 1 else (float('nan'), None)
    return {'r2': r2, 'r': r_p, 'rho': float(rho), 'rmse': rmse, 'mae': mae}

# MAIN
def main():
    print('\n' + '='*70)
    print('GC-MPNN INFERENCE')
    print(f'  Checkpoint   : {CHECKPOINT_PATH}')
    print(f'  Test CSV     : {TEST_CSV}')
    print(f'  Output CSV   : {OUTPUT_CSV}')
    print('='*70)

    # Load model
    print('\nLoading checkpoint...')
    ckpt    = torch.load(CHECKPOINT_PATH, map_location=device,
                         weights_only=False)
    cfg     = ckpt['model_config']
    node_sc = ckpt['node_scaler']
    edge_sc = ckpt['edge_scaler']
    gas_sc  = ckpt['gas_scaler']
    y_sc    = ckpt['target_scaler']

    print(f'  Trained on     : {ckpt.get("trained_on_gases")}')
    print(f'  Gas descriptor : {ckpt.get("gas_feature_desc")}')
    print(f'  Train R2       : {ckpt.get("train_r2", "N/A")}')

    model = GasConditionedMPNN(**cfg).to(device)
    model.load_state_dict(
        {k: v.to(device) for k, v in ckpt['model_state_dict'].items()})
    model.eval()
    print(f'  Parameters     : {sum(p.numel() for p in model.parameters()):,}')

    # Load CSV
    df      = read_test_csv(TEST_CSV)
    col_map = resolve_columns(df)

    smi1_col  = col_map.get('smiles_string_1')
    smi2_col  = col_map.get('smiles_string_2')
    mf1_col   = col_map.get('mf_1')
    mf2_col   = col_map.get('mf_2')
    pname_col = col_map.get('polymer_name')
    ptype_col = col_map.get('polymer_type')

    if not smi1_col:
        raise ValueError("Could not find smiles_string_1 column. "
                         "Check column names printed above.")

    # Classify polymers (homopolymer / copolymer)
    df['_poly_type'] = df.apply(
        lambda r: classify_polymer(r, smi2_col), axis=1)

    n_homo = (df['_poly_type'] == 'homo').sum()
    n_co   = (df['_poly_type'] == 'co').sum()
    print(f'\nPolymer counts in CSV:')
    print(f'  Homopolymers : {n_homo}')
    print(f'  Copolymers   : {n_co}')
    print(f'  -> Keeping all: {len(df)} rows')

    if len(df) == 0:
        print('\nNo rows in CSV. Exiting.')
        return

    # Inference loop
    rows    = []
    skipped = []

    for row_idx, row in df.iterrows():

        poly_name     = str(row.get(pname_col, row_idx)).strip() \
                        if pname_col else str(row_idx)
        poly_type_lbl = str(row.get(ptype_col, '')).strip() \
                        if ptype_col else ''
        smi1          = str(row.get(smi1_col, '')).strip()

        smi2_str = str(row.get(smi2_col, np.nan)).strip() if smi2_col else ''
        smi2     = smi2_str if is_valid_smiles_candidate(smi2_str) else None
        is_co    = smi2 is not None

        # Mole fractions
        mf1, mf2 = np.nan, np.nan
        if is_co:
            try:
                mf1 = float(row.get(mf1_col, np.nan)) if mf1_col else np.nan
                mf2 = float(row.get(mf2_col, np.nan)) if mf2_col else np.nan
            except (ValueError, TypeError):
                pass
            if pd.isna(mf2) and not pd.isna(mf1):
                mf2 = 1.0 - mf1
            if pd.isna(mf1) and not pd.isna(mf2):
                mf1 = 1.0 - mf2

        # Validate smiles_string_1
        if not is_valid_smiles_candidate(smi1):
            skipped.append((row_idx, poly_name,
                            f'smiles_string_1 not a valid candidate: "{smi1}"'))
            continue
        if Chem.MolFromSmiles(smi1) is None:
            skipped.append((row_idx, poly_name,
                            f'RDKit cannot parse: {smi1[:50]}'))
            continue
        if is_co and Chem.MolFromSmiles(smi2) is None:
            print(f'  WARNING: smiles_string_2 unparseable for '
                  f'"{poly_name}" - treating as homopolymer.')
            is_co = False
            smi2  = None

        for gas in TARGET_GASES:

            # Ground truth: Barrer -> log10 Barrer
            y_true_log10 = np.nan
            gas_col = col_map.get(gas)
            if gas_col:
                try:
                    raw_f = float(row.get(gas_col, np.nan))
                    if raw_f > 0:
                        y_true_log10 = np.log10(raw_f)
                except (ValueError, TypeError):
                    pass

            # Monomer 1 prediction
            pred1 = predict_smiles(smi1, gas, model,
                                   node_sc, edge_sc, gas_sc, y_sc)
            pred2      = np.nan
            pred_final = np.nan

            if pred1 is not None:
                if is_co and not (pd.isna(mf1) or pd.isna(mf2)):
                    pred2 = predict_smiles(smi2, gas, model,
                                           node_sc, edge_sc, gas_sc, y_sc)
                    pred_final = (mf1 * pred1 + mf2 * pred2
                                  if pred2 is not None else pred1)
                else:
                    pred_final = pred1

            rows.append({
                'polymer_type':   poly_type_lbl,
                'polymer_name':   poly_name,
                'smiles_1':       smi1,
                'smiles_2':       smi2 if is_co else '',
                'is_copolymer':   'Yes' if is_co else 'No',
                'mf_1':           mf1   if is_co else '',
                'mf_2':           mf2   if is_co else '',
                'gas':            gas,
                'is_ood':         'Yes' if gas in OOD_GASES else 'No',
                'y_true_log10':   round(y_true_log10, 4)
                                  if not np.isnan(y_true_log10) else np.nan,
                'y_pred_log10':   round(pred_final, 4)
                                  if pred_final is not None
                                  and not np.isnan(pred_final) else np.nan,
                'residual':       round(pred_final - y_true_log10, 4)
                                  if (pred_final is not None
                                      and not np.isnan(pred_final)
                                      and not np.isnan(y_true_log10))
                                  else np.nan,
                'y_pred_1_log10': round(pred1, 4)
                                  if pred1 is not None else np.nan,
                'y_pred_2_log10': round(pred2, 4)
                                  if not np.isnan(pred2) else np.nan,
            })

        print(f'  [{row_idx}] {poly_name:<50} '
              f'{"copolymer" if is_co else "homopolymer"}')

    if skipped:
        print(f'\nSkipped {len(skipped)} rows:')
        for idx, name, reason in skipped:
            print(f'  Row {idx}: {name} - {reason}')

    # Summary & metrics
    out_df  = pd.DataFrame(rows)
    has_gt  = out_df['y_true_log10'].notna() & out_df['y_pred_log10'].notna()

    print(f'\n{"="*70}')
    print(f'RESULTS')
    print(f'{"="*70}')
    print(f'  Polymer rows used      : {len(df)}')
    print(f'  (polymer, gas) pairs   : {len(out_df)}')
    print(f'  Pairs with ground truth: {has_gt.sum()}')
    print(f'  Copolymer pairs        : {(out_df["is_copolymer"]=="Yes").sum()}')
    print(f'  OOD gas pairs          : {(out_df["is_ood"]=="Yes").sum()}')

    # Ground truth availability per gas
    print(f'\n  Ground truth availability per gas:')
    print(f'  {"Gas":<10} {"n_with_gt":>10} {"n_predicted":>12}  {"OOD":>5}')
    print(f'  {""*44}')
    for gas in TARGET_GASES:
        n_pred = ((out_df['gas'] == gas) & out_df['y_pred_log10'].notna()).sum()
        n_gt   = ((out_df['gas'] == gas) & has_gt).sum()
        ood    = 'Yes' if gas in OOD_GASES else 'No'
        print(f'  {gas:<10} {n_gt:>10} {n_pred:>12}  {ood:>5}')

    # Overall metrics
    if has_gt.sum() > 1:
        yt  = out_df.loc[has_gt, 'y_true_log10'].values
        yp  = out_df.loc[has_gt, 'y_pred_log10'].values
        met = compute_metrics(yt, yp)

        print(f'\n  Overall metrics (n={has_gt.sum()}):')
        print(f'    R2   = {met["r2"]:.4f}')
        print(f'    r    = {met["r"]:.4f}  (Pearson)')
        print(f'    rho    = {met["rho"]:.4f}  (Spearman)')
        print(f'    RMSE = {met["rmse"]:.4f}  (log10 Barrer)')
        print(f'    MAE  = {met["mae"]:.4f}  (log10 Barrer)')

        # Per-gas metrics
        print(f'\n  Per-gas breakdown:')
        print(f'  {"Gas":<10} {"n":>4} {"R2":>8} {"r":>8} '
              f'{"rho":>8} {"RMSE":>8} {"MAE":>8}  {"OOD":>5}')
        print(f'  {""*72}')
        for gas in TARGET_GASES:
            mask = (out_df['gas'] == gas) & has_gt
            if mask.sum() < 2:
                continue
            yt_g = out_df.loc[mask, 'y_true_log10'].values
            yp_g = out_df.loc[mask, 'y_pred_log10'].values
            m_g  = compute_metrics(yt_g, yp_g)
            ood  = 'Yes' if gas in OOD_GASES else 'No'
            print(f'  {gas:<10} {mask.sum():>4} {m_g["r2"]:>8.4f} '
                  f'{m_g["r"]:>8.4f} {m_g["rho"]:>8.4f} '
                  f'{m_g["rmse"]:>8.4f} {m_g["mae"]:>8.4f}  {ood:>5}')

    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f'\n  Saved: {OUTPUT_CSV}')
    print('='*70)


if __name__ == '__main__':
    main()
