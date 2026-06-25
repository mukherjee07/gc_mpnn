# Data

This folder holds the datasets used by all the other scripts. The other
folders read their data from here, so keep these files in place.

## Files

- `Gas_permeability_solubility_diffusivity_wide.csv` — the main training set.
  One row per polymer (`smiles_string`), with experimental permeability
  columns (`p_exp_He`, `p_exp_H2`, `p_exp_N2`, `p_exp_O2`, `p_exp_CH4`,
  `p_exp_CO2`). This is the file the hyper-opt and active-learning scripts use.

- `new_test_set.csv` — separate polymers used by the prediction script
  (`prediction/final_test_evaluation.py`) to test the pretrained model.

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
