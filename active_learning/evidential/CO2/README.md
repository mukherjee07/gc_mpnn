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

## Setup

Make sure the `poly_net` environment is set up (one-time install is described in
the repository's root README), then activate it:

```bash
conda activate poly_net
```

GPU (CUDA) / Apple Silicon (MPS) / CPU is selected automatically.

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

## Using a different held-out gas

`CO2` is just the example provided here. To make any other gas the held-out
species, set `TEST_GAS` to that gas (e.g. `'O2'`) at the top of **both**
`gnn_evidential_calibrated.py` and `gnn_active.py`, then re-run the two steps.

## What you get

- `Kinetic_evidential_test_CO2_predictions.csv` — evidential predictions +
  uncertainty (from Step 1).
- `Kinetic_evidential_summary.json`, `Kinetic_lambda_calibration_log.csv` —
  diagnostics from the calibration step.
- `Kinetic_AL_<strategy>_N<N>_test_CO2_predictions.csv` and a matching
  `_summary.json` — the active-learning results (from Step 2).
