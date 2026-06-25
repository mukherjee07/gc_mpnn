
from psmiles import PolymerSmiles as PS
import psmiles

import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool, GlobalAttention
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import warnings
import json

warnings.filterwarnings('ignore')

# Set device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon GPU (MPS)")
else:
    device = torch.device("cpu")
    print("Using CPU")
print(f"Using device: {device}\n")

# DataLoader workers: use 2 on CUDA, 0 on MPS/CPU (MPS doesn't benefit from multiprocess workers)
NUM_WORKERS = 2 if device.type == 'cuda' else 0
PIN_MEMORY = device.type == 'cuda'

def set_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seeds(42)

# ============================================================================
# GAS PROPERTIES DATABASE
# ============================================================================

GAS_PROPERTIES = {
    'He': {'sigma': 2.551, 'epsilon': 10.2, 'omega': -0.383, 'Tc': 5.2, 'Pc': 2.28, 
           'd': 2.6, 'Vd': 2.67, 'q_pos': 0.0, 'q_neg': 0.0},
    'H2': {'sigma': 2.827, 'epsilon': 59.7, 'omega': -0.265, 'Tc': 33.2, 'Pc': 13.00,
           'd': 2.89, 'Vd': 6.12, 'q_pos': 0.0, 'q_neg': 0.0},
    'N2': {'sigma': 3.798, 'epsilon': 71.4, 'omega': 0.037, 'Tc': 126.2, 'Pc': 63.14,
           'd': 3.64, 'Vd': 18.5, 'q_pos': 0.964, 'q_neg': 0.482},
    'O2': {'sigma': 3.467, 'epsilon': 106.7, 'omega': 0.022, 'Tc': 154.6, 'Pc': 50.43,
           'd': 3.46, 'Vd': 16.3, 'q_pos': 0.226, 'q_neg': 0.113},
    'CH4': {'sigma': 3.758, 'epsilon': 148.6, 'omega': 0.011, 'Tc': 190.6, 'Pc': 46.1,
            'd': 3.8, 'Vd': 24.42, 'q_pos': 0.0, 'q_neg': 0.0},
    'CO2': {'sigma': 3.941, 'epsilon': 195.2, 'omega': 0.253, 'Tc': 304.1, 'Pc': 73.80,
            'd': 3.3, 'Vd': 26.9, 'q_pos': 0.70, 'q_neg': 0.35},
    'H2S': {'sigma': 3.623, 'epsilon': 301.1, 'omega': 0.100, 'Tc': 373.3, 'Pc': 89.63,
            'd': 3.6, 'Vd': 32.9, 'q_pos': 0.20, 'q_neg': 0.40},
}

GASES = ['He', 'H2', 'N2', 'O2', 'CH4', 'CO2']

# ============================================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================================

def get_gas_features_thermodynamic(gas_name):
    props = GAS_PROPERTIES[gas_name]
    return np.array([props['sigma'], props['epsilon'], props['omega'],
                     props['Tc'], props['Pc']], dtype=np.float32)

def get_gas_features_kinetic(gas_name):
    props = GAS_PROPERTIES[gas_name]
    return np.array([props['d'], props['Vd']], dtype=np.float32)

def get_gas_features_electrostatics(gas_name):
    props = GAS_PROPERTIES[gas_name]
    return np.array([props['q_pos'], props['q_neg']], dtype=np.float32)

def get_gas_features_full(gas_name):
    props = GAS_PROPERTIES[gas_name]
    return np.array([props['sigma'], props['epsilon'], props['omega'],
                     props['Tc'], props['Pc'], props['d'], props['Vd'],
                     props['q_pos'], props['q_neg']], dtype=np.float32)

def get_gas_features_onehot(gas_name):
    gas_to_idx = {'He': 0, 'H2': 1, 'N2': 2, 'O2': 3, 'CH4': 4, 'CO2': 5}
    onehot = np.zeros(6, dtype=np.float32)
    if gas_name in gas_to_idx:
        onehot[gas_to_idx[gas_name]] = 1.0
    return onehot

EXPERIMENT_CONFIGS = {
    'Thermodynamic': {'feature_func': get_gas_features_thermodynamic, 'feature_dim': 5, 
                      'description': 'σ, ε, ω, Tc, Pc'},
    'Kinetic': {'feature_func': get_gas_features_kinetic, 'feature_dim': 2, 
                'description': 'd, Vd'},
    'Electrostatics': {'feature_func': get_gas_features_electrostatics, 'feature_dim': 2, 
                       'description': 'q⁺, q⁻'},
    'Full': {'feature_func': get_gas_features_full, 'feature_dim': 9, 
             'description': 'σ, ε, ω, Tc, Pc, d, Vd, q⁺, q⁻'},
    'OneHot': {'feature_func': get_gas_features_onehot, 'feature_dim': 6, 
               'description': 'Categorical (1-of-6)'}
}

# ============================================================================
# DATA LOADING
# ============================================================================

print("Loading data...")
pol_sd = pd.read_csv('Gas_permeability_solubility_diffusivity_wide.csv', delimiter=',')

smiles = pol_sd['smiles_string']
p_exp_ch4 = pol_sd['p_exp_CH4']
p_exp_co2 = pol_sd['p_exp_CO2']
p_exp_h2 = pol_sd['p_exp_H2']
p_exp_n2 = pol_sd['p_exp_N2']
p_exp_o2 = pol_sd['p_exp_O2']
p_exp_he = pol_sd['p_exp_He']

def create_dataset_for_experiment(experiment_name):
    config = EXPERIMENT_CONFIGS[experiment_name]
    feature_func = config['feature_func']
    
    data_records = []
    for idx in range(len(smiles)):
        smi = smiles.iloc[idx]
        gas_data = {
            'CH4': p_exp_ch4.iloc[idx], 'CO2': p_exp_co2.iloc[idx],
            'H2': p_exp_h2.iloc[idx], 'N2': p_exp_n2.iloc[idx],
            'O2': p_exp_o2.iloc[idx], 'He': p_exp_he.iloc[idx],
        }
        for gas_name, permeability in gas_data.items():
            if not np.isnan(permeability):
                data_records.append({
                    'smiles': smi, 'gas': gas_name,
                    'permeability': permeability,
                    'gas_features': feature_func(gas_name)
                })
    return data_records

# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def smiles_to_graph(smiles_str):
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        return None
    
    node_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(), atom.GetDegree(), atom.GetFormalCharge(),
            atom.GetHybridization().real, int(atom.GetIsAromatic()),
            atom.GetTotalNumHs(), atom.GetNumImplicitHs(),
        ]
        node_features.append(features)
    
    x = torch.tensor(node_features, dtype=torch.float)
    
    edge_indices = []
    edge_features = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices.extend([[i, j], [j, i]])
        
        edge_feat = [
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
        edge_features.extend([edge_feat, edge_feat])
    
    if len(edge_indices) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 7), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def create_multi_gas_dataset(data_records, feature_dim):
    dataset = []
    for record in data_records:
        graph = smiles_to_graph(record['smiles'])
        if graph is not None:
            graph.gas_features = torch.tensor(record['gas_features'], dtype=torch.float).unsqueeze(0)
            graph.y = torch.tensor([record['permeability']], dtype=torch.float)
            graph.gas_name = record['gas']
            dataset.append(graph)
    return dataset

# ============================================================================
# MPNN ARCHITECTURE WITH FLEXIBLE POOLING
# ============================================================================

class MPNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim):
        super(MPNNLayer, self).__init__(aggr='add')
        self.message_mlp = nn.Sequential(
            nn.Linear(in_channels * 2 + edge_dim, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
    
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        return self.message_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
    
    def update(self, aggr_out, x):
        return self.update_mlp(torch.cat([x, aggr_out], dim=-1))


class GasConditionedMPNN(nn.Module):
    def __init__(self, node_features=7, edge_features=7, gas_features=5,
                 hidden_dim=64, num_mp_layers=3, fusion_dim=128,
                 l2_lambda=0.001, dropout=0.3, pooling='mean'):
        super(GasConditionedMPNN, self).__init__()
        
        self.l2_lambda = l2_lambda
        self.expected_gas_dim = gas_features
        self.pooling_type = pooling
        
        self.node_embedding = nn.Linear(node_features, hidden_dim)
        
        self.mp_layers = nn.ModuleList([
            MPNNLayer(hidden_dim, hidden_dim, edge_features) 
            for _ in range(num_mp_layers)
        ])
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_mp_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        if pooling == 'attention':
            self.attention_gate = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
            self.attention_pool = GlobalAttention(self.attention_gate)
        
        self.gas_encoder = nn.Sequential(
            nn.Linear(gas_features, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(dropout),
            nn.Linear(64, fusion_dim),
            nn.ReLU(),
            nn.LayerNorm(fusion_dim),
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        gas_features = data.gas_features
        
        batch_size = batch.max().item() + 1
        
        if gas_features.shape != (batch_size, self.expected_gas_dim):
            raise ValueError(f"gas_features shape mismatch: {gas_features.shape} vs [{batch_size}, {self.expected_gas_dim}]")
        
        x = F.relu(self.node_embedding(x))
        
        for mp_layer, bn in zip(self.mp_layers, self.batch_norms):
            x_new = mp_layer(x, edge_index, edge_attr)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = self.dropout(x_new)
            x = x + x_new
        
        if self.pooling_type == 'mean':
            polymer_embedding = global_mean_pool(x, batch)
        elif self.pooling_type == 'sum':
            polymer_embedding = global_add_pool(x, batch)
        elif self.pooling_type == 'attention':
            polymer_embedding = self.attention_pool(x, batch)
        else:
            polymer_embedding = global_mean_pool(x, batch)
        
        gas_embedding = self.gas_encoder(gas_features)
        combined = torch.cat([polymer_embedding, gas_embedding], dim=-1)
        permeability = self.fusion(combined)
        
        return permeability.squeeze(-1)
    
    def l2_regularization(self):
        l2_reg = torch.tensor(0., device=device)
        for param in self.parameters():
            l2_reg += torch.norm(param, 2)
        return self.l2_lambda * l2_reg

# ============================================================================
# TRAINING AND EVALUATION FUNCTIONS
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, scheduler=None):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, data.y) + model.l2_regularization()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if scheduler is not None:
        scheduler.step()
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            outputs = model(data)
            loss = criterion(outputs, data.y)
            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy().tolist())
            all_targets.extend(data.y.cpu().numpy().tolist())
    return total_loss / len(val_loader), np.array(all_preds), np.array(all_targets)

def calculate_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.nan
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape, 'r2': r2}

def standardize_features(train_graphs, val_graphs, feature_dim):
    all_node_features = np.vstack([g.x.numpy() for g in train_graphs])
    all_edge_features = []
    all_gas_features = []
    
    for g in train_graphs:
        if g.edge_attr.shape[0] > 0:
            all_edge_features.append(g.edge_attr.numpy())
        all_gas_features.append(g.gas_features.squeeze(0).numpy())
    
    node_scaler = StandardScaler()
    edge_scaler = StandardScaler()
    gas_scaler = StandardScaler()
    
    node_scaler.fit(all_node_features)
    gas_scaler.fit(np.array(all_gas_features))
    if len(all_edge_features) > 0:
        edge_scaler.fit(np.vstack(all_edge_features))
    
    train_scaled = []
    for g in train_graphs:
        g_copy = g.clone()
        g_copy.x = torch.tensor(node_scaler.transform(g.x.numpy()), dtype=torch.float)
        gas_np = g.gas_features.squeeze(0).numpy().reshape(1, -1)
        g_copy.gas_features = torch.tensor(gas_scaler.transform(gas_np), dtype=torch.float)
        if g.edge_attr.shape[0] > 0:
            g_copy.edge_attr = torch.tensor(edge_scaler.transform(g.edge_attr.numpy()), dtype=torch.float)
        train_scaled.append(g_copy)
    
    val_scaled = []
    for g in val_graphs:
        g_copy = g.clone()
        g_copy.x = torch.tensor(node_scaler.transform(g.x.numpy()), dtype=torch.float)
        gas_np = g.gas_features.squeeze(0).numpy().reshape(1, -1)
        g_copy.gas_features = torch.tensor(gas_scaler.transform(gas_np), dtype=torch.float)
        if g.edge_attr.shape[0] > 0:
            g_copy.edge_attr = torch.tensor(edge_scaler.transform(g.edge_attr.numpy()), dtype=torch.float)
        val_scaled.append(g_copy)
    
    return train_scaled, val_scaled, node_scaler, edge_scaler, gas_scaler

# ============================================================================
# PRECOMPUTE LOGO SPLITS (scaling done once, reused every trial)
# ============================================================================

def precompute_logo_splits(all_graphs, feature_dim):
    """Precompute all 6 LOGO splits with scaled features and targets.
    
    Scaling (node/edge/gas features + target y) depends only on the data,
    not on hyperparameters, so we do it once and cache the results.
    Returns a dict keyed by test_gas.
    """
    splits = {}
    for test_gas in GASES:
        train_graphs_logo = [g for g in all_graphs if g.gas_name != test_gas]
        test_graphs_logo = [g for g in all_graphs if g.gas_name == test_gas]
        
        if len(test_graphs_logo) == 0:
            continue
        
        # Scale targets
        y_train_raw = np.array([g.y.item() for g in train_graphs_logo])
        y_test_raw = np.array([g.y.item() for g in test_graphs_logo])
        
        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
        y_test_scaled = y_scaler.transform(y_test_raw.reshape(-1, 1)).flatten()
        
        train_graphs_scaled_y = []
        for i, g in enumerate(train_graphs_logo):
            g_copy = g.clone()
            g_copy.y = torch.tensor([y_train_scaled[i]], dtype=torch.float)
            train_graphs_scaled_y.append(g_copy)
        
        test_graphs_scaled_y = []
        for i, g in enumerate(test_graphs_logo):
            g_copy = g.clone()
            g_copy.y = torch.tensor([y_test_scaled[i]], dtype=torch.float)
            test_graphs_scaled_y.append(g_copy)
        
        # Scale node/edge/gas features
        train_scaled, test_scaled, node_scaler, edge_scaler, gas_scaler = standardize_features(
            train_graphs_scaled_y, test_graphs_scaled_y, feature_dim
        )
        
        splits[test_gas] = {
            'train_graphs': train_scaled,
            'test_graphs': test_scaled,
            'y_scaler': y_scaler,
            'n_train': len(train_graphs_logo),
            'n_test': len(test_graphs_logo),
        }
    
    return splits

# ============================================================================
# LOGO VALIDATION FUNCTION (Uses precomputed splits)
# ============================================================================

def run_logo_validation(precomputed_splits, feature_dim, hyperparams,
                        num_epochs=100, patience=15, verbose=False, trial=None):
    """Run full LOGO validation using precomputed splits.
    
    CHANGE 2: Reports R² after each gas fold so Optuna can prune early.
    CHANGE 4: Uses cosine annealing LR scheduler and fewer epochs/patience.
    """
    test_r2_list = []
    
    for step, test_gas in enumerate(GASES):
        if test_gas not in precomputed_splits:
            continue
        
        split = precomputed_splits[test_gas]
        
        # CHANGE 5: pin_memory + num_workers in DataLoader
        train_loader = DataLoader(
            split['train_graphs'], batch_size=hyperparams['batch_size'],
            shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
        )
        test_loader = DataLoader(
            split['test_graphs'], batch_size=hyperparams['batch_size'],
            shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
        )
        
        model = GasConditionedMPNN(
            node_features=7, edge_features=7, gas_features=feature_dim,
            hidden_dim=hyperparams['hidden_dim'],
            num_mp_layers=hyperparams['num_mp_layers'],
            fusion_dim=hyperparams['fusion_dim'],
            l2_lambda=hyperparams['l2_lambda'],
            dropout=hyperparams['dropout'],
            pooling=hyperparams['pooling']
        ).to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
        # CHANGE 4: Cosine annealing scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        best_train_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(num_epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler=scheduler)
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            if patience_counter >= patience:
                break
        
        model.load_state_dict(best_model_state)
        _, y_pred_test_scaled, y_true_test_scaled = validate(model, test_loader, criterion)
        
        y_scaler = split['y_scaler']
        y_pred_test = y_scaler.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).flatten()
        y_true_test = y_scaler.inverse_transform(y_true_test_scaled.reshape(-1, 1)).flatten()
        
        test_metrics = calculate_metrics(y_true_test, y_pred_test)
        test_r2_list.append(test_metrics['r2'])
        
        if verbose:
            print(f"    {test_gas}: R² = {test_metrics['r2']:.3f}")
        
        # CHANGE 2: Per-fold pruning — report running mean R² after each gas
        if trial is not None:
            running_mean_r2 = np.mean(test_r2_list)
            trial.report(running_mean_r2, step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()
    
    return np.mean(test_r2_list), np.std(test_r2_list), test_r2_list

# ============================================================================
# OPTUNA OBJECTIVE FUNCTION
# ============================================================================

def create_objective(precomputed_splits, feature_dim, experiment_name):
    def objective(trial):
        hyperparams = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'l2_lambda': trial.suggest_float('l2_lambda', 1e-4, 1e-2, log=True),
            'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128, 256, 512]),
            'num_mp_layers': trial.suggest_categorical('num_mp_layers', [2, 3, 4, 5, 6, 7]),
            'fusion_dim': trial.suggest_categorical('fusion_dim', [16, 32, 64, 128, 256, 512]),
            'dropout': trial.suggest_float('dropout', 0.01, 0.5),
            'pooling': trial.suggest_categorical('pooling', ['mean', 'attention', 'sum']),
            'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128]),
        }
        
        # CHANGE 4: Fewer epochs (100 vs 200) and tighter patience (15 vs 30) during search
        mean_r2, std_r2, per_gas_r2 = run_logo_validation(
            precomputed_splits, feature_dim, hyperparams,
            num_epochs=100, patience=15, verbose=False, trial=trial
        )
        
        for gas, r2 in zip(GASES, per_gas_r2):
            trial.set_user_attr(f'r2_{gas}', r2)
        trial.set_user_attr('r2_std', std_r2)
        
        return mean_r2
    
    return objective

# ============================================================================
# MAIN LOOP
# ============================================================================

N_OPTUNA_TRIALS = 75
FINAL_EPOCHS = 500
FINAL_PATIENCE = 100

print("="*80)
print("GAS-CONDITIONED MPNN WITH OPTUNA HYPERPARAMETER OPTIMIZATION")
print("="*80)
print(f"\nOptuna trials per experiment: {N_OPTUNA_TRIALS}")
print(f"Optimization objective: Mean R^2 across LOGO validation")

all_experiment_results = {}
all_best_params = {}
#exp_list = ['Thermodynamic', 'Kinetic', 'Electrostatics', 'Full', 'OneHot']
exp_list = ['Electrostatics']

for experiment_name in exp_list:
    print("\n" + "="*80)
    print(f"EXPERIMENT: {experiment_name}")
    print("="*80)
    
    config = EXPERIMENT_CONFIGS[experiment_name]
    feature_dim = config['feature_dim']
    print(f"Gas Features: {config['description']} ({feature_dim}D)")
    
    print(f"\nCreating dataset...")
    data_records = create_dataset_for_experiment(experiment_name)
    all_graphs = create_multi_gas_dataset(data_records, feature_dim)
    print(f"Created {len(all_graphs)} polymer-gas pairs")
    
    # CHANGE 1: Precompute all LOGO splits once before Optuna
    print(f"Precomputing LOGO splits (scaling features + targets)...")
    precomputed_splits = precompute_logo_splits(all_graphs, feature_dim)
    print(f"Precomputed {len(precomputed_splits)} LOGO folds")
    
    # PHASE 1: Optuna optimization
    print(f"\n{'='*60}")
    print(f"PHASE 1: Optuna Hyperparameter Optimization")
    print(f"{'='*60}")
    
    # CHANGE 2: Increased n_warmup_steps for per-fold pruning (6 steps = 6 gases)
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    )
    
    objective = create_objective(precomputed_splits, feature_dim, experiment_name)
    
    print(f"Running {N_OPTUNA_TRIALS} trials...")
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True, gc_after_trial=True)
    
    best_params = study.best_params
    best_value = study.best_value
    
    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"\nTrials completed: {n_complete}, pruned: {n_pruned}")
    
    print(f"\n{'='*40}")
    print(f"BEST HYPERPARAMETERS (R² = {best_value:.4f})")
    print(f"{'='*40}")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    all_best_params[experiment_name] = {'params': best_params, 'best_r2': best_value, 'n_trials': len(study.trials)}
    
    # PHASE 2: Final evaluation (uses full epochs, cosine scheduler, precomputed splits)
    print(f"\n{'='*60}")
    print(f"PHASE 2: Final LOGO Evaluation with Optimized Hyperparameters")
    print(f"{'='*60}")
    
    logo_results = {}
    
    for test_gas in GASES:
        if test_gas not in precomputed_splits:
            continue
        
        print(f"\n{'-'*50}")
        print(f"LOGO: Holding out {test_gas}")
        
        split = precomputed_splits[test_gas]
        print(f"Train: {split['n_train']}, Test: {split['n_test']}")
        
        # CHANGE 5: pin_memory + num_workers
        train_loader = DataLoader(
            split['train_graphs'], batch_size=best_params['batch_size'],
            shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
        )
        test_loader = DataLoader(
            split['test_graphs'], batch_size=best_params['batch_size'],
            shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
        )
        
        model = GasConditionedMPNN(
            node_features=7, edge_features=7, gas_features=feature_dim,
            hidden_dim=best_params['hidden_dim'],
            num_mp_layers=best_params['num_mp_layers'],
            fusion_dim=best_params['fusion_dim'],
            l2_lambda=best_params['l2_lambda'],
            dropout=best_params['dropout'],
            pooling=best_params['pooling']
        ).to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])
        # CHANGE 4: Cosine annealing for final training too
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINAL_EPOCHS)
        
        best_train_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        print(f"Training (max {FINAL_EPOCHS} epochs)...")
        
        for epoch in range(FINAL_EPOCHS):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler=scheduler)
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            if (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}: Loss={train_loss:.4f}")
            if patience_counter >= FINAL_PATIENCE:
                print(f"  Early stopping at epoch {epoch + 1}")
                break
        
        model.load_state_dict(best_model_state)
        
        y_scaler = split['y_scaler']
        
        _, y_pred_train_scaled, y_true_train_scaled = validate(model, train_loader, criterion)
        y_pred_train = y_scaler.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).flatten()
        y_true_train = y_scaler.inverse_transform(y_true_train_scaled.reshape(-1, 1)).flatten()
        train_metrics = calculate_metrics(y_true_train, y_pred_train)
        
        _, y_pred_test_scaled, y_true_test_scaled = validate(model, test_loader, criterion)
        y_pred_test = y_scaler.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).flatten()
        y_true_test = y_scaler.inverse_transform(y_true_test_scaled.reshape(-1, 1)).flatten()
        test_metrics = calculate_metrics(y_true_test, y_pred_test)
        
        logo_results[test_gas] = {
            'n_train': split['n_train'], 'n_test': split['n_test'],
            'train_metrics': train_metrics, 'test_metrics': test_metrics,
            'y_true_train': y_true_train, 'y_pred_train': y_pred_train,
            'y_true_test': y_true_test, 'y_pred_test': y_pred_test
        }
        
        print(f"{test_gas}: Train R²={train_metrics['r2']:.3f}, Test R²={test_metrics['r2']:.3f}")
    
    all_experiment_results[experiment_name] = logo_results
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {experiment_name}")
    print(f"{'='*60}")
    
    train_r2_list = [logo_results[g]['train_metrics']['r2'] for g in GASES if g in logo_results]
    test_r2_list = [logo_results[g]['test_metrics']['r2'] for g in GASES if g in logo_results]
    
    print(f"Mean Train R²: {np.mean(train_r2_list):.3f} ± {np.std(train_r2_list):.3f}")
    print(f"Mean Test R²:  {np.mean(test_r2_list):.3f} ± {np.std(test_r2_list):.3f}")

# ============================================================================
# FINAL COMPARISON
# ============================================================================

print("\n" + "="*80)
print("FINAL COMPARISON: ALL EXPERIMENTS")
print("="*80)

comparison_data = []
for exp_name in exp_list:
    if exp_name in all_experiment_results:
        logo_res = all_experiment_results[exp_name]
        best_p = all_best_params[exp_name]['params']
        
        test_r2_all = [logo_res[g]['test_metrics']['r2'] for g in GASES if g in logo_res]
        
        comparison_data.append({
            'Experiment': exp_name,
            'Dim': EXPERIMENT_CONFIGS[exp_name]['feature_dim'],
            'Test R²': f"{np.mean(test_r2_all):.3f}±{np.std(test_r2_all):.3f}",
            'LR': f"{best_p['learning_rate']:.2e}",
            'Hidden': best_p['hidden_dim'],
            'Layers': best_p['num_mp_layers'],
            'Pooling': best_p['pooling']
        })

print(f"\n{'Experiment':<15} {'Dim':<5} {'Test R²':<15} {'LR':<12} {'Hidden':<8} {'Layers':<8} {'Pooling':<10}")
print("-" * 85)
for row in comparison_data:
    print(f"{row['Experiment']:<15} {row['Dim']:<5} {row['Test R²']:<15} {row['LR']:<12} {row['Hidden']:<8} {row['Layers']:<8} {row['Pooling']:<10}")

# Ranking
test_r2_means = [(row['Experiment'], float(row['Test R²'].split('±')[0])) for row in comparison_data]
test_r2_means.sort(key=lambda x: x[1], reverse=True)

print(f"\nRanking by Test R²:")
for i, (exp, r2) in enumerate(test_r2_means):
    print(f"  {i+1}. {exp}: {r2:.3f}")

# Save results
results_to_save = {
    'all_experiment_results': all_experiment_results,
    'all_best_params': all_best_params,
    'comparison_data': comparison_data,
    'experiment_configs': {k: {'feature_dim': v['feature_dim'], 'description': v['description']} 
                          for k, v in EXPERIMENT_CONFIGS.items()},
}

torch.save(results_to_save, 'optuna_logo_results.pt')
print("\nSaved: optuna_logo_results.pt")

best_params_json = {exp: {'params': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                                      for k, v in data['params'].items()},
                          'best_r2': float(data['best_r2'])}
                    for exp, data in all_best_params.items()}

with open('best_hyperparameters.json', 'w') as f:
    json.dump(best_params_json, f, indent=2)
print("Saved: best_hyperparameters.json")

print("\n" + "="*80)
print("OPTIMIZATION COMPLETE!")
print("="*80)
