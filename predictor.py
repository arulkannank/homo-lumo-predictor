"""HOMO/LUMO prediction from SMILES with the models in model/ (exported from the training pipeline).

    from predictor import Predictor
    p = Predictor()
    p.predict(['c1ccc2cc3ccccc3cc2c1'])[0]['HOMO']

Two models predict every molecule:
- the GNN ensemble (main prediction): molecules are turned into graphs exactly as for training (mol_graph.py,
  RDKit 2026.09), every GNN predicts HOMO and LUMO, and the mean is the prediction;
- LightGBM (classical machine learning) on Avalon + ECFP4 count fingerprints, preprocessed exactly as for training
  (train_utils.prepare_xy).
Each result also says how far the molecule is from the training data (ECFP4 similarity to the nearest training
molecule, and the test-set error at that similarity) and, if the molecule is in the CSD_EES dataset, its DFT values.
"""
import io
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from PIL import Image
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import Draw, rdFingerprintGenerator
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import gnn_utils as gu
import mol_graph

RDLogger.DisableLog('rdApp.*')
HERE = os.path.dirname(os.path.abspath(__file__))
LOW_SIMILARITY = 0.3      # below this, no training molecule is similar
MORGAN2 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
# The LightGBM fingerprints, as CSD_Other_fingerprints_descriptors.ipynb computed them
ML_FINGERPRINTS = {'avalon_counts': lambda m: pyAvalonTools.GetAvalonCountFP(m, nBits=512).ToList(),
                   'ecfp4_counts': MORGAN2.GetCountFingerprintAsNumPy}
FLOAT32_MAX = np.float32(1e30)


class Predictor:
    def __init__(self, model_dir=os.path.join(HERE, 'model')):
        with open(os.path.join(model_dir, 'members.json')) as f:
            self.info = json.load(f)
        sc = self.info['scaler']
        self.y_mean, self.y_std = np.array(sc['y_mean'], np.float32), np.array(sc['y_std'], np.float32)
        self.g_mean, self.g_std = np.array(sc['g_mean']), np.array(sc['g_std'])
        self.graph_names = self.info['names']['graph']
        self.models = {}
        for m in self.info['members']:
            model = gu.build_model(m['name'], self.info['dims'], **m['arch'])
            model.load_state_dict(torch.load(os.path.join(model_dir, m['file']), map_location='cpu'))
            self.models[m['name']] = model.to(gu.DEVICE).eval()

        ref = np.load(os.path.join(model_dir, 'reference.npz'))
        table = pd.DataFrame({'smiles': ref['smiles'], 'HOMO': ref['HOMO'], 'LUMO': ref['LUMO'], 'in_train': ref['in_train']})
        # the same molecule can be in several crystal structures: average them
        self.dataset = table.groupby('smiles').agg(HOMO=('HOMO', 'mean'), LUMO=('LUMO', 'mean'),
                                                  structures=('HOMO', 'size'), in_train=('in_train', 'any'))
        self.train_rows = ref['train_rows']
        self.ref_smiles, self.ref_y = ref['smiles'], np.stack([ref['HOMO'], ref['LUMO']], 1)
        self.train_fps = [DataStructs.CreateFromFPSText(str(t)) for t in ref['train_fps']]
        self.error_table = pd.read_csv(os.path.join(model_dir, 'error_by_similarity.csv'))

        # LightGBM: one booster per target, and the preprocessing fitted on its training rows
        pre = np.load(os.path.join(model_dir, 'lgbm_preprocess.npz'))
        self.ml_sets = [str(s) for s in pre['sets']]
        self.ml_pre = {s: {k: pre[f'{s}|{k}'] for k in ('keep', 'center', 'scale')} for s in self.ml_sets}
        self.ml_clip = np.float32(pre['clip'])
        self.ml_models = {t: lgb.Booster(model_file=os.path.join(model_dir, f'lgbm_{t}.txt')) for t in self.info['targets']}

    # ---- featurization ----
    def to_data(self, mol):
        """PyG graph of an RDKit molecule, standardized like the training data."""
        x, edge_index, edge_attr, gd = mol_graph.mol_to_arrays(mol)
        g = np.array([gd[k] for k in self.graph_names], dtype=np.float64)
        g = np.clip(np.nan_to_num((g - self.g_mean) / self.g_std), -10, 10).astype(np.float32)
        return Data(x=torch.from_numpy(x), edge_index=torch.from_numpy(edge_index),
                    edge_attr=torch.from_numpy(edge_attr), g=torch.from_numpy(g[None, :]))

    def ml_features(self, mols):
        """Avalon + ECFP4 count fingerprints, preprocessed like the LightGBM training data (train_utils.prepare_xy):
        training medians for missing values, the same kept columns, (x - median) / IQR, clipped to +-100."""
        parts = []
        for s in self.ml_sets:
            p = self.ml_pre[s]
            raw = np.full((len(mols), len(p['keep'])), np.nan, dtype=np.float32)
            for i, m in enumerate(mols):
                try:
                    raw[i] = ML_FINGERPRINTS[s](m)
                except Exception:      # e.g. Avalon cannot kekulize the molecule: imputed below, as in training
                    pass
            raw = np.where(np.isfinite(raw), np.clip(raw, -FLOAT32_MAX, FLOAT32_MAX), np.nan).astype(np.float32)
            nan = np.isnan(raw)
            raw[nan] = np.broadcast_to(p['center'], raw.shape)[nan]
            x = raw[:, p['keep']]
            x -= p['center'][p['keep']]
            x /= p['scale'][p['keep']]
            np.clip(x, -self.ml_clip, self.ml_clip, out=x)
            parts.append(x)
        return np.hstack(parts)

    def check(self, mol):
        """Reasons to trust a prediction less."""
        warnings = []
        other = sorted({a.GetSymbol() for a in mol.GetAtoms() if a.GetSymbol() not in mol_graph.ELEMENTS})
        if other:
            warnings.append(f"Contains {', '.join(other)}: not among the elements of the training data "
                            f"({', '.join(mol_graph.ELEMENTS)}), so the prediction is unreliable.")
        if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
            warnings.append('Has unpaired electrons (radical); the training molecules are closed-shell.')
        if len(Chem.GetMolFrags(mol)) > 1:
            warnings.append('Several separate components (e.g. a salt or co-crystal): HOMO/LUMO are predicted for '
                            'the whole entry, as in the CSD_EES data.')
        if mol.GetNumAtoms() > 150:
            warnings.append(f'Very large ({mol.GetNumAtoms()} heavy atoms); few training molecules are this big.')
        return warnings

    def expected_mae(self, similarity, prefix=''):
        """Test-set MAE (HOMO, LUMO) for molecules this similar to the training set: of the GNN ensemble, or of
        LightGBM with prefix='LGBM_'."""
        lows = self.error_table['low'].to_numpy()
        row = self.error_table.iloc[max(0, int(np.searchsorted(lows, similarity, side='right')) - 1)]
        return float(row[f'{prefix}HOMO_MAE']), float(row[f'{prefix}LUMO_MAE'])

    # ---- prediction ----
    @torch.no_grad()
    def _predict_graphs(self, graphs, batch_size=128):
        """{model name: predictions in eV (n x 2)}."""
        out = {}
        for name, model in self.models.items():
            preds = [model(b.to(gu.DEVICE)).cpu().numpy() for b in DataLoader(graphs, batch_size=batch_size)]
            out[name] = np.concatenate(preds) * self.y_std + self.y_mean
        return out

    def predict(self, smiles_list, progress=None):
        """One result dict per input SMILES (with 'error' set if it could not be predicted)."""
        results, graphs, ok = [], [], []
        for k, s in enumerate(smiles_list):
            r = {'input': s, 'error': None, 'warnings': []}
            mol = Chem.MolFromSmiles(str(s).strip()) if str(s).strip() else None
            if mol is None or mol.GetNumAtoms() == 0:
                r['error'] = 'RDKit could not read this SMILES.'
            else:
                try:
                    graphs.append(self.to_data(mol))
                    ok.append(k)
                    r.update(mol=mol, smiles=Chem.MolToSmiles(mol), warnings=self.check(mol))
                except Exception as e:     # e.g. a descriptor that fails for an unusual structure
                    r['error'] = f'Could not build the molecular graph: {e}'
            results.append(r)
        if not graphs:
            return results
        member_preds = self._predict_graphs(graphs)
        stacked = np.stack(list(member_preds.values()))           # models x molecules x 2
        mean, spread = stacked.mean(0), stacked.std(0)
        X = self.ml_features([results[k]['mol'] for k in ok])
        ml = {t: self.ml_models[t].predict(X) for t in self.info['targets']}
        for n, k in enumerate(ok):
            r = results[k]
            r.update(HOMO=float(mean[n, 0]), LUMO=float(mean[n, 1]), gap=float(mean[n, 1] - mean[n, 0]),
                     HOMO_spread=float(spread[n, 0]), LUMO_spread=float(spread[n, 1]),
                     members={name: (float(p[n, 0]), float(p[n, 1])) for name, p in member_preds.items()},
                     LGBM_HOMO=float(ml['HOMO'][n]), LGBM_LUMO=float(ml['LUMO'][n]),
                     LGBM_gap=float(ml['LUMO'][n] - ml['HOMO'][n]))
            fp = MORGAN2.GetFingerprint(r['mol'])
            sims = np.array(DataStructs.BulkTanimotoSimilarity(fp, self.train_fps))
            best = int(sims.argmax())
            row = self.train_rows[best]
            r.update(similarity=float(sims[best]), nearest_smiles=str(self.ref_smiles[row]),
                     nearest_HOMO=float(self.ref_y[row, 0]), nearest_LUMO=float(self.ref_y[row, 1]))
            r['expected_MAE_HOMO'], r['expected_MAE_LUMO'] = self.expected_mae(r['similarity'])
            r['expected_LGBM_MAE_HOMO'], r['expected_LGBM_MAE_LUMO'] = self.expected_mae(r['similarity'], 'LGBM_')
            if r['similarity'] < LOW_SIMILARITY:
                r['warnings'].append(f"No similar molecule in the training set (nearest Tanimoto {r['similarity']:.2f}): "
                                     'expect larger errors.')
            for t in ('HOMO', 'LUMO'):
                s = self.info['spread'][t]
                if r[f'{t}_spread'] > s['p95']:
                    r['warnings'].append(
                        f"The models disagree on {t} (spread {r[f'{t}_spread']:.2f} eV) more than for 95% of test "
                        f"molecules; on those, the {t} error was {s['MAE_above']:.2f} eV on average "
                        f"(other molecules: {s['MAE_below']:.2f} eV).")
            if r['smiles'] in self.dataset.index:
                d = self.dataset.loc[r['smiles']]
                r.update(dataset_HOMO=float(d['HOMO']), dataset_LUMO=float(d['LUMO']),
                         dataset_structures=int(d['structures']), in_training_set=bool(d['in_train']))
            if progress:
                progress((n + 1) / len(ok))
        return results

    def to_table(self, results, names=None):
        """Results as a DataFrame for display / CSV download."""
        rows = []
        for i, r in enumerate(results):
            rows.append({'name': names[i] if names is not None else None, 'input SMILES': r['input'],
                         'GNN HOMO (eV)': r.get('HOMO'), 'GNN LUMO (eV)': r.get('LUMO'), 'GNN gap (eV)': r.get('gap'),
                         'LightGBM HOMO (eV)': r.get('LGBM_HOMO'), 'LightGBM LUMO (eV)': r.get('LGBM_LUMO'),
                         'LightGBM gap (eV)': r.get('LGBM_gap'),
                         'GNN HOMO spread (eV)': r.get('HOMO_spread'), 'GNN LUMO spread (eV)': r.get('LUMO_spread'),
                         'similarity to training set': r.get('similarity'),
                         'typical GNN HOMO error (eV)': r.get('expected_MAE_HOMO'),
                         'typical GNN LUMO error (eV)': r.get('expected_MAE_LUMO'),
                         'typical LightGBM HOMO error (eV)': r.get('expected_LGBM_MAE_HOMO'),
                         'typical LightGBM LUMO error (eV)': r.get('expected_LGBM_MAE_LUMO'),
                         'CSD_EES DFT HOMO (eV)': r.get('dataset_HOMO'), 'CSD_EES DFT LUMO (eV)': r.get('dataset_LUMO'),
                         'notes': r['error'] or ' '.join(r['warnings'])})
        table = pd.DataFrame(rows)
        return table if names is not None else table.drop(columns='name')

    # ---- explanation ----
    def explain(self, mol, model='GINE', steps=64):
        """Integrated Gradients atom attributions (eV, atoms x [HOMO, LUMO]) of one ensemble member."""
        scaler = {'y_mean': self.y_mean, 'y_std': self.y_std}
        return gu.integrated_gradients(self.models[model], self.to_data(mol), scaler, steps=steps)['atom']


def draw(mol, size=(420, 320)):
    return Draw.MolToImage(mol, size=size)


def draw_attributions(mol, weights, size=(420, 320)):
    """Atom weights as a heat map (red: raises the orbital energy, blue: lowers it), scaled to the largest |weight|."""
    d2d = rdMolDraw2D.MolDraw2DCairo(*size)
    w = np.asarray(weights, dtype=float)
    w = w / (np.abs(w).max() or 1.0)
    SimilarityMaps.GetSimilarityMapFromWeights(mol, [float(v) for v in w], draw2d=d2d, colorMap='bwr', contourLines=0)
    d2d.FinishDrawing()
    return Image.open(io.BytesIO(d2d.GetDrawingText()))
