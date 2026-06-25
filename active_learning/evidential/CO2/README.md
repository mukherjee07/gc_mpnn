# Active learning — evidential uncertainty (CO2)

Goal: same idea as the ensemble version, but here a single "evidential" model
predicts its own uncertainty (no need to train 20 models). We then add the
most-uncertain CO2 polymers to training and check if that beats random.

## What's in here

- `gnn_evidential_calibrated.py` — **Step 1.** Trains the evidential model.
  It first auto-tunes the uncertainty so that the 95% intervals really cover
  about 95% of points (calibration), then saves predictions + uncertainty.
- `gnn_active.py` — **Step 2.** Picks polymers to add (by uncertainty or at
  random), retrains one model, and reports the result.
- `Kinetic_best_params.json` — the hyperparameters used (already provided).
- `bash_al`, `ml_job` — cluster job scripts (ignore if running locally).

It reads the training set from `../../../data/Gas_permeability_solubility_diffusivity_wide.csv`.

## What you need (one-time install)

```bash
pip install torch torch_geometric pandas numpy scikit-learn scipy rdkit
```

GPU / Apple Silicon / CPU is selected automatically.

## How to run (do these in order)

From inside this folder:

```bash
cd active_learning/evidential/CO2

# Step 1: train the evidential model (long job)
python gnn_evidential_calibrated.py

# Step 2: run active learning (uses the file made in Step 1)
python gnn_active.py
```

To compare strategies, open `gnn_active.py` and edit near the top:
- `N_ACTIVE` — how many CO2 polymers to add (e.g. 10).
- `SELECTION_STRATEGY` — `'uncertainty'` or `'random'`.

Run Step 2 again for each setting and compare the scores.

## What you get

- `Kinetic_evidential_test_CO2_predictions.csv` — evidential predictions +
  uncertainty (from Step 1).
- `Kinetic_evidential_summary.json`, `Kinetic_lambda_calibration_log.csv` —
  diagnostics from the calibration step.
- `Kinetic_AL_<strategy>_N<N>_test_CO2_predictions.csv` and a matching
  `_summary.json` — the active-learning results (from Step 2).
