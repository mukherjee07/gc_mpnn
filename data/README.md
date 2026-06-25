# Data

This folder holds the datasets used by all the other scripts. The other
folders read their data from here, so keep these files in place.

## Files

- `Gas_permeability_solubility_diffusivity_wide.csv` — the primary dataset,
  used for **training, validation, and testing**. One row per polymer
  (`smiles_string`), with experimental permeability columns (`p_exp_He`,
  `p_exp_H2`, `p_exp_N2`, `p_exp_O2`, `p_exp_CH4`, `p_exp_CO2`). This is the file
  the hyper-opt and active-learning scripts use. **Source:** developed by Rampi
  Ramprasad's group (Phan et al., 2024) — see citations below.

- `new_test_set.csv` — separate polymers used by the final-test script
  (`final_test/final_test_evaluation.py`) **only for testing** the pretrained
  model. **Source:** experimental data from the Membrane Society of Australasia
  database (Thornton et al., 2012); the p-SMILES were manually curated and added
  by us — see citations below.

- `clean.py` — a small helper that drops a few columns from `new_test_set.csv`
  and writes `new_test_set_cleaned.csv`. Optional; only run it if you need that
  cleaned version.

## Running the helper (optional)

```bash
cd data
python clean.py
```

Needs only pandas:

```bash
pip install pandas
```

## Data sources and citations

We gratefully acknowledge the original sources of these datasets. Please cite
them if you use the data.

**Primary dataset** (`Gas_permeability_solubility_diffusivity_wide.csv`) — used
for training, validation, and testing; from Rampi Ramprasad's group:

> Phan, B.K., Shen, KH., Gurnani, R. et al. *Gas permeability, diffusivity, and
> solubility in polymers: Simulation-experiment data fusion and multi-task
> machine learning.* npj Computational Materials **10**, 186 (2024).
> https://doi.org/10.1038/s41524-024-01373-9

**External test set** (`new_test_set.csv`) — used for testing only; experimental
data from the Membrane Society of Australasia database; the p-SMILES were
manually curated and added by us:

> A. W. Thornton, B. D. Freeman and L. M. Robeson. *Polymer Gas Separation
> Membrane Database* (2012).
> https://membrane-australasia.org/polymer-gas-separation-membrane-database/
