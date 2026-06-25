# Active learning — ensemble uncertainty (CO2)

Goal: train many models (a "deep ensemble"), use their disagreement as an
uncertainty score, then add the most-uncertain CO2 polymers to the training
data and check if that beats adding random ones.

## What's in here

- `ensemble.py` — **Step 1.** Trains the ensemble and saves predictions with
  an uncertainty (`y_std`) for every test polymer.
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
cd active_learning/ensemble_and_random/CO2

# Step 1: train the ensemble (long job — trains 20 models)
python ensemble.py

# Step 2: run active learning (uses the file made in Step 1)
python gnn_active.py
```

To compare strategies, open `gnn_active.py` and edit near the top:
- `N_ACTIVE` — how many CO2 polymers to add (e.g. 10).
- `SELECTION_STRATEGY` — `'uncertainty'` or `'random'`.

Run Step 2 again for each setting and compare the scores.

## What you get

- `Kinetic_ensemble_test_CO2_predictions.csv` — ensemble predictions +
  uncertainty (from Step 1).
- `Kinetic_ensemble_summary.json`, `Kinetic_ensemble_member_metrics.csv` —
  ensemble diagnostics.
- `Kinetic_AL_<strategy>_N<N>_test_CO2_predictions.csv` and a matching
  `_summary.json` — the active-learning results (from Step 2).
