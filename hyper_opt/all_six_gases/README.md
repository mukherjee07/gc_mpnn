# Pretrained model — train on all six gases

This trains the GC-MPNN on all six gases (He, H2, N2, O2, CH4, CO2) and saves
a reusable "pretrained" model. That saved model is the one used later in the
`final_test/` folder (e.g. to predict H2S, which has no labelled data to tune
against).

## What's in here

- `gnn_logo_optuna.py` — the script you run. It tunes hyperparameters across
  all six gases, then trains a final model on all of them and saves it.

It reads the training set from `../../data/Gas_permeability_solubility_diffusivity_wide.csv`.

## Setup

Make sure the `poly_net` environment is set up (one-time install is described in
the repository's root README), then activate it:

```bash
conda activate poly_net
```

GPU (CUDA) / Apple Silicon (MPS) / CPU is selected automatically.

## How to run

From inside this folder:

```bash
cd hyper_opt/all_six_gases
python gnn_logo_optuna.py
```

**Note:** this is a long job (Optuna tuning + final training). Run it on a
workstation or cluster.

## What you get

- `gc_mpnn_pretrained.pt` — the trained model weights.
- `gc_mpnn_pretrained_checkpoint.pt` — a self-contained checkpoint (weights +
  scalers + settings). **This is the file the `final_test/` script loads.**
- `all6_optuna_trials.csv` — the tuning history.
- `all6_best_params.json` — the best hyperparameters.
