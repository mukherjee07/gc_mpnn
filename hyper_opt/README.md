# Hyperparameter optimization (HPO) + final evaluation

This finds the best model settings and then tests the model on one held-out
gas. The held-out gas (`CO2` by default) is never seen during tuning, so the
final score is a fair test.

## What's in here

- `gnn_logo_optuna.py` — the script you run. It does two stages:
  1. **Tuning** — Optuna searches for the best hyperparameters using the other
     five gases only (leave-one-gas-out). The test gas is locked away.
  2. **Final evaluation** — trains with the best settings and predicts the
     held-out gas.
- `all_six_gases/` — a related version that trains on all six gases to create
  the pretrained model. See its own README.

It reads the training set from `../data/Gas_permeability_solubility_diffusivity_wide.csv`.

## What you need (one-time install)

```bash
pip install torch torch_geometric pandas numpy scikit-learn scipy rdkit optuna
```

GPU / Apple Silicon / CPU is selected automatically.

## How to run

From inside this folder:

```bash
cd hyper_opt
python gnn_logo_optuna.py
```

To change which gas is held out, open `gnn_logo_optuna.py` and edit
`TEST_GAS` near the top (default `'CO2'`).

**Note:** this is a long job (many Optuna trials, each trains a model). It is
meant to run on a workstation or cluster, not in a few minutes on a laptop.

## What you get (CSV files written in this folder)

- `<experiment>_best_params.json` — the best hyperparameters found.
- `<experiment>_test_<gas>_predictions.csv` — predictions on the held-out gas.
- `<experiment>_train_pool_predictions.csv` — predictions on the training pool.
- `<experiment>_optuna_trials.csv` — the full tuning history.
- `all_experiments_summary.csv` — summary across the descriptor experiments.
