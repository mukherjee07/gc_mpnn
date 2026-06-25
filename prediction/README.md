# Prediction — test the pretrained model

This runs the already-trained GC-MPNN model on a set of polymers and reports
how well it predicts gas permeability.

## What's in here

- `final_test_evaluation.py` — the script you run.
- `gc_mpnn_pretrained_checkpoint.pt` — the trained model (weights + scalers). Already provided.
- `all6_best_params.json` — the hyperparameters the model was trained with.
- `inference_predictions_all.csv` — output (created when you run the script).

It reads the test polymers from `../data/new_test_set.csv`.

## What you need (one-time install)

```bash
pip install torch torch_geometric pandas numpy scikit-learn scipy rdkit
```

The script automatically uses your GPU (CUDA), Apple Silicon (MPS), or CPU —
you don't have to change anything.

## How to run

From inside this folder:

```bash
cd prediction
python final_test_evaluation.py
```

That's it. It runs on every polymer (homopolymers and copolymers) in the CSV.

## What you get

- Printed metrics in the terminal: R², Pearson r, Spearman ρ, RMSE, MAE —
  both overall and per gas.
- A file `inference_predictions_all.csv` with one row per (polymer, gas),
  showing the true value, the predicted value, and the difference.
