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
 
## Associated paper (under review)
 
This repository accompanies the following manuscript, which is currently **under review**:
 
**Gas-Conditioned Message Passing Graph Neural Network for Permeability Prediction in Polymeric Membranes**
 
Krishnendu Mukherjee, Zidan Zhang, Mohammed Alshammasi, Mohammed G. Hashim, Hussain H. Naji, Zainab A. Aithan, Jihad A. Badra, Jalal Yagoubi, Hussain B. Tuwailib, Ali Hayek, and Venkat Ganesan
 
Affiliations:
- **The University of Texas at Austin** — Krishnendu Mukherjee, Zidan Zhang, Venkat Ganesan
- **Saudi Aramco** — Mohammed Alshammasi, Mohammed G. Hashim, Hussain H. Naji, Zainab A. Aithan, Jihad A. Badra, Jalal Yagoubi, Hussain B. Tuwailib, Ali Hayek
 
---
 
## Repository structure
 
Each folder is self-contained and has its own README with step-by-step
instructions. This table is just the map.
 
| Folder | What it does |
|--------|--------------|
| `data/` | The datasets. Every script reads its data from here. |
| `final_test/` | Evaluate the **already-trained** model on the external (MSA) test set. Easiest place to start. |
| `hyper_opt/` | Tune the model and evaluate it on one held-out gas (leave-one-gas-out). |
| `hyper_opt/all_six_gases/` | Train on all six gases to produce the **pretrained model** used by `final_test/`. |
| `active_learning/` | Ask which few extra measurements would improve the model most (two variants: ensemble and evidential). |
 
**How the pieces fit together**
- `hyper_opt/all_six_gases/` trains and saves the pretrained model
  (`gc_mpnn_pretrained_checkpoint.pt`).
- `final_test/` loads that pretrained model and evaluates it on the external
  test set (`data/new_test_set.csv`, the MSA data) — the held-out test of the
  final model. (Note: `hyper_opt/` also produces predictions, but on its
  internal held-out gas; `final_test/` is the separate external evaluation.)
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
thing to try is the final-test script, which uses the trained model that is
already included:
 
```bash
conda activate poly_net
cd final_test
python final_test_evaluation.py
```
 
This prints the evaluation metrics (R², Pearson r, etc.) and writes
`inference_predictions_all.csv`. See `final_test/README.md` for what the output
means, and the README inside each folder for the other workflows.
 
---
 
## Hardware Requirements
 
All codes have been tested on:
- **NVIDIA A100 (40 GB VRAM)** or more — recommended
- Apple Silicon M4 Pro (earlier versions)
 
Hyperparameter optimization protocol
<img width="12657" height="6229" alt="Figure_3" src="https://github.com/user-attachments/assets/28ae56a9-fdb2-4536-a5ce-3941c1a4b8f2" />
 
---
 
## Data sources and acknowledgements
 
The datasets in `data/` come from prior work by other groups, and we gratefully
acknowledge them. Please cite the original sources if you use these data.
 
### Primary dataset (training, validation, and testing) — `Gas_permeability_solubility_diffusivity_wide.csv`
 
This dataset was developed by **Rampi Ramprasad's group** and is used here for
model **training, validation, and testing**. If you use it, please cite:
 
> Phan, B.K., Shen, KH., Gurnani, R. et al. *Gas permeability, diffusivity, and
> solubility in polymers: Simulation-experiment data fusion and multi-task
> machine learning.* npj Computational Materials **10**, 186 (2024).
> https://doi.org/10.1038/s41524-024-01373-9
 
```bibtex
@article{Phan2024,
  title   = {Gas permeability, diffusivity, and solubility in polymers: Simulation-experiment data fusion and multi-task machine learning},
  author  = {Phan, B. K. and Shen, K.-H. and Gurnani, R. and Tran, H. and Lively, R. and Ramprasad, R.},
  journal = {npj Computational Materials},
  volume  = {10},
  pages   = {186},
  year    = {2024},
  doi     = {10.1038/s41524-024-01373-9}
}
```
 
### External test set (testing only) — `new_test_set.csv`
 
This data is used **only for testing/evaluation**. The experimental permeability
values were collected from the **Membrane Society of Australasia** Polymer Gas
Separation Membrane Database. The polymer SMILES (p-SMILES) in this file were
**manually curated and added by us**.
 
> A. W. Thornton, B. D. Freeman and L. M. Robeson. *Polymer Gas Separation
> Membrane Database* (2012).
> https://membrane-australasia.org/polymer-gas-separation-membrane-database/
