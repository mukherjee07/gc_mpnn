# Final test — evaluate the pretrained model

This runs the already-trained GC-MPNN model on the external (MSA) test set and
reports how well it predicts gas permeability. This is the final held-out
evaluation of the pretrained model (separate from the held-out-gas predictions
done in `hyper_opt/`).

## What's in here

- `final_test_evaluation.py` — the script you run.
- `gc_mpnn_pretrained_checkpoint.pt` — the trained model (weights + scalers). Already provided.
- `all6_best_params.json` — the hyperparameters the model was trained with.
- `inference_predictions_all.csv` — output (created when you run the script).

It reads the test polymers from `../data/new_test_set.csv`.

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
cd final_test
python final_test_evaluation.py
```

That's it. It runs on every polymer (homopolymers and copolymers) in the CSV.

## What you get

- Printed metrics in the terminal: R², Pearson r, Spearman ρ, RMSE, MAE —
  both overall and per gas.
- A file `inference_predictions_all.csv` with one row per (polymer, gas),
  showing the true value, the predicted value, and the difference.
