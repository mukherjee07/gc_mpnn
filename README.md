# gc-mpnn (gas-conditioned message passing graph neural network) for permeability prediction in polymer membranes
<img width="11875" height="5888" alt="Figure_2" src="https://github.com/user-attachments/assets/e29eaaf7-a15e-48dc-883d-80175755cd30" />
 
**Author --- Krishnendu Mukherjee**  
The University of Texas at Austin  
McKetta Department of Chemical Engineering — Ganesan Polymer Physics Lab
 
---
 
## Overview
 
This repository contains machine learning model: **GC-MPNN** (Graph Convolution Message Passing Neural Network).
 
All code has been developed within the `poly_net` conda environment.
 
---
 
## Repository structure
 
Each folder is self-contained and has its own README with step-by-step
instructions. This table is just the map.
 
| Folder | What it does |
|--------|--------------|
| `data/` | The datasets. Every script reads its data from here. |
| `prediction/` | Test the **already-trained** model on new polymers. Easiest place to start. |
| `hyper_opt/` | Tune the model and evaluate it on one held-out gas (leave-one-gas-out). |
| `hyper_opt/all_six_gases/` | Train on all six gases to produce the **pretrained model** used by `prediction/`. |
| `active_learning/` | Ask which few extra measurements would improve the model most (two variants: ensemble and evidential). |
 
**How the pieces fit together**
- `hyper_opt/all_six_gases/` trains and saves the pretrained model
  (`gc_mpnn_pretrained_checkpoint.pt`), which `prediction/` then loads.
- `hyper_opt/` is the held-out-gas study (tune on five gases, test on the sixth).
- `active_learning/` builds on the same model: first train a model that reports
  its own uncertainty, then add the most-uncertain samples and re-check.
 
Always run a script from inside its own folder, since data paths are relative
(e.g. `../data/...`).
 
---
 
## Environment Setup
 
### Prerequisites
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda installed
- CUDA 12.2 (recommended for GPU support)
### 1. Create the `poly_net` Conda Environment
 
```bash
conda env create -f environment.yml --name poly_net
```
 
### 2. Verify and Activate the Environment
 
```bash
conda env list        # verify poly_net appears in the list
conda activate poly_net
```
 
### 3. Install p-smiles Libraries (Manual Step Required)
 
> Due to compatibility issues, these cannot be installed via `environment.yml` and must be installed manually after activation.
 
```bash
pip install git+https://github.com/Ramprasad-Group/canonicalize_psmiles.git
pip install git+https://github.com/kuennethgroup/psmiles.git
```
 
### 4. Install torch-geometric
 
```bash
pip install torch-geometric
```
 
---
 
## Quick start
 
After creating and activating the `poly_net` environment (above), the fastest
thing to try is the prediction script, which uses the trained model that is
already included:
 
```bash
conda activate poly_net
cd prediction
python final_test_evaluation.py
```
 
This prints the prediction metrics (R², Pearson r, etc.) and writes
`inference_predictions_all.csv`. See `prediction/README.md` for what the output
means, and the README inside each folder for the other workflows.
 
---
 
## Hardware Requirements
 
All codes have been tested on:
- **NVIDIA A100 (40 GB VRAM)** or more — recommended
- Apple Silicon M4 Pro (earlier versions)
 
Hyperparameter optimization protocol
<img width="12657" height="6229" alt="Figure_3" src="https://github.com/user-attachments/assets/28ae56a9-fdb2-4536-a5ce-3941c1a4b8f2" />
