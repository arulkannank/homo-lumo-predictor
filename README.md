# HOMO/LUMO predictor for organic molecules

A Streamlit app that predicts the **HOMO and LUMO energies** (and the gap) of an organic molecule from its SMILES.
The models were trained on the DFT frontier-orbital energies of 48,115 molecules from crystal structures in the
**CSD_EES** dataset (Omar, Nematiaram, Troisi & Padula, 2021).

Two models predict every molecule:

- **GNN ensemble** (main prediction): the mean of five graph neural networks on the molecular graph: GINE,
  GINE with tuned hyperparameters, GINE + whole-graph descriptors, AttentiveFP and a Chemprop-style directed MPNN.
- **LightGBM** (classical machine learning, for comparison): gradient-boosted trees on Avalon + ECFP4 count
  fingerprints, hyperparameters tuned with Optuna.

## Performance

Held-out test set of 9,617 molecules (no SMILES in both training and test set), and 5-fold cross-validation grouped by SMILES:

| | GNN ensemble HOMO | GNN ensemble LUMO | LightGBM HOMO | LightGBM LUMO |
|---|---|---|---|---|
| Test R² | 0.916 | 0.927 | 0.830 | 0.857 |
| Test MAE (eV) | 0.122 | 0.120 | 0.178 | 0.171 |
| 5-fold CV R² | 0.920 ± 0.003 | 0.925 ± 0.002 | 0.843 ± 0.006 | 0.860 ± 0.004 |

For comparison, the DFT values of the same molecule in two different crystal structures differ by a median of about
0.05 eV in the dataset.

## Run it

Python 3.12:

```bash
pip install -r requirements.txt
streamlit run app.py
```

On Windows, `run_app.cmd` starts the app from a virtual environment in `.venv`. The app opens at
http://localhost:8501. Everything runs on the CPU; 500 molecules take a few seconds.

- **Single molecule:** type a SMILES (or pick an example) to get both predictions, a drawing, and optionally a heat map of
  which atoms raise or lower each orbital energy (Integrated Gradients of the GINE model).
- **Many molecules:** paste one SMILES per line (optionally followed by a name) or upload a CSV with a `SMILES` column,
  and download the results as CSV.

From Python:

```python
from predictor import Predictor

p = Predictor()
r = p.predict(['c1ccc2cc3cc4cc5ccccc5cc4cc3cc2c1'])[0]
print(r['HOMO'], r['LUMO'], r['LGBM_HOMO'], r['LGBM_LUMO'], r['warnings'])
```

## How far to trust a prediction

Every prediction comes with:

- **Similarity to the training data:** the ECFP4 Tanimoto similarity to the nearest training molecule, and the test-set
  error for molecules that similar. The GNN ensemble's MAE ranges from about 0.09 eV (similarity ≥ 0.7) to
  0.22–0.24 eV (similarity < 0.3).
- **Disagreement between the five GNNs:** flagged when it is larger than for 95% of test molecules. On those
  molecules the error was more than twice the usual one.
- **Warnings** for elements outside the training data (only H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br and I were seen),
  radicals, very large molecules and multi-component entries.
- **The DFT values themselves** if the molecule is in CSD_EES, and whether it was in the training set.

The predicted values are single-molecule DFT energies as computed for CSD_EES, not measured or solid-state levels.

## How the models were built

- **Graph features:** heavy atoms as nodes (element, degree, charge, hydrogens, hybridization, aromaticity, ring
  membership, chirality, mass), bonds in both directions as edges (bond type, stereo, conjugation, ring), plus 30
  whole-graph descriptors (`mol_graph.py`). The graphs must be built with RDKit 2026.09, the version used for
  training, which is why `requirements.txt` pins it.
- **Training:** multi-task (HOMO and LUMO together) on standardized targets, AdamW, early stopping on 10% of the
  training molecules. GNN code is in `gnn_utils.py`.
- **Splits:** 80/20 train/test split grouped by SMILES, so the same molecule from different crystal structures is
  never in both sets. Cross-validation uses 5 folds grouped the same way.

## Data and attribution

The models are trained on, and `model/reference.npz` contains, data derived from:

> Omar, O., Nematiaram, T., Troisi, A. and Padula, D. (2021). *Organic Materials Repurposing: a Data Set for
> Theoretical Predictions of New Applications for Existing Compounds.* University of Liverpool [Data Collection].
> https://doi.org/10.17638/datacat.liverpool.ac.uk/1472

The dataset's data files are licensed under
[Creative Commons Attribution 3.0](https://creativecommons.org/licenses/by/3.0/). Changes made here:

- dropped the entries without a SMILES and the 52 SMILES that RDKit cannot read;
- rewrote 394 SMILES with dative bonds in charge-separated form, accepting a rewrite only if it kept the CSD formula
  and was neutral overall;
- stored canonical RDKit SMILES with the DFT HOMO and LUMO values.

If you use this app or its models, please cite the dataset above.
