# Active learning

These experiments ask a simple question: if we are allowed to add just a few
real measurements of the held-out gas (CO2) to the training data, which ones
should we pick to improve the model the most?

The idea: train a model that also gives an *uncertainty* for each prediction,
then add the polymers the model is least sure about, retrain, and see how much
better it gets compared to just picking random polymers.

## Two variants (each in its own folder)

- `ensemble_and_random/CO2/` — uncertainty comes from a **deep ensemble**
  (train many models, look at how much they disagree).
- `evidential/CO2/` — uncertainty comes from a single **evidential** model
  (the model predicts its own uncertainty directly).

Each folder has its own README with the exact steps. In both cases the order
is the same: **first** train the uncertainty model, **then** run the active
learning script.

**Held-out gas:** the examples here use **CO2** as the held-out gas, but any of
the six gases can be used instead — just set `TEST_GAS` to the desired gas in the
scripts (see each folder's README). Only the CO2 example is provided here.

**Selection strategies:** the active-learning script runs both the
**uncertainty-based** selection and the **random** baseline (chosen with
`SELECTION_STRATEGY`), so you can compare them.

All scripts read the training set from
`../../../data/Gas_permeability_solubility_diffusivity_wide.csv`.

The `bash_al` / `ml_job` files you may see are cluster job scripts (for SLURM).
You can ignore them if you are running locally.
