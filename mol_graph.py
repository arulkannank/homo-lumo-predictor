"""SMILES -> molecular graph, exactly as CSD_graph_features.ipynb built graphs/ for the GNNs.

Heavy atoms are nodes (49 features), bonds are edges in both directions (11 features), plus 30 whole-graph
descriptors. Used by the Streamlit predictor (app/), which must featurize new molecules the same way.
Build the graphs with the same RDKit version as graphs/ (2026.09): aromaticity / hybridization perception can
differ between versions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import GraphDescriptors, rdMolDescriptors

HT = Chem.rdchem.HybridizationType
BT = Chem.rdchem.BondType
ST = Chem.rdchem.BondStereo
CT = Chem.rdchem.ChiralType
ELEMENTS = ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'I']
NODE_SPEC = [('element', ELEMENTS, lambda a: a.GetSymbol()),
             ('degree', [0, 1, 2, 3, 4, 5], lambda a: a.GetTotalDegree()),
             ('charge', [-1, 0, 1], lambda a: a.GetFormalCharge()),
             ('num_H', [0, 1, 2, 3], lambda a: a.GetTotalNumHs()),
             ('hybridization', [HT.SP, HT.SP2, HT.SP3, HT.SP3D, HT.SP3D2], lambda a: a.GetHybridization()),
             ('chirality', [CT.CHI_UNSPECIFIED, CT.CHI_TETRAHEDRAL_CW, CT.CHI_TETRAHEDRAL_CCW], lambda a: a.GetChiralTag())]
EDGE_SPEC = [('bond', [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC], lambda b: b.GetBondType()),
             ('stereo', [ST.STEREONONE, ST.STEREOZ, ST.STEREOE], lambda b: b.GetStereo())]
RING_SIZES = [3, 4, 5, 6, 7, 8]
GRAPH_INDICES = ['BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v',
                 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha', 'Kappa1', 'Kappa2', 'Kappa3']


def one_hot(value, choices):
    """One-hot over `choices` plus a final 'other' slot."""
    v = [0] * (len(choices) + 1)
    v[choices.index(value) if value in choices else -1] = 1
    return v


def atom_features(a):
    f = []
    for _, choices, get in NODE_SPEC:
        f += one_hot(get(a), choices)
    f += [int(a.GetIsAromatic()), int(a.IsInRing())] + [int(a.IsInRingSize(k)) for k in RING_SIZES]
    return f + [a.GetMass() / 100]


def bond_features(b):
    f = []
    for _, choices, get in EDGE_SPEC:
        f += one_hot(get(b), choices)
    return f + [int(b.GetIsConjugated()), int(b.IsInRing())]


def graph_descriptors(m):
    n, e = m.GetNumAtoms(), m.GetNumBonds()
    D = Chem.GetDistanceMatrix(m).astype(float)
    D[D > n] = np.nan                          # atoms in different fragments are not connected
    ecc = np.nanmax(D, axis=1)
    pairs = D[np.triu_indices(n, 1)]
    deg = np.array([a.GetDegree() for a in m.GetAtoms()])
    d = {'n_atoms': n, 'n_bonds': e, 'n_rings': rdMolDescriptors.CalcNumRings(m),
         'n_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(m),
         'cyclomatic': e - n + len(Chem.GetMolFrags(m)), 'diameter': np.nanmax(ecc), 'radius': np.nanmin(ecc),
         'wiener': np.nansum(pairs), 'mean_distance': np.nanmean(pairs) if len(pairs) else 0.0,
         'mean_degree': deg.mean(), 'max_degree': deg.max()}
    for name in GRAPH_INDICES:
        d[name] = getattr(GraphDescriptors, name)(m)
    d['log10_Ipc'] = np.log10(GraphDescriptors.Ipc(m) + 1)   # Ipc itself overflows for big molecules
    return d


def mol_to_arrays(m):
    """(node features, edge_index, edge features, graph descriptor dict) of an RDKit molecule."""
    x = np.array([atom_features(a) for a in m.GetAtoms()], dtype=np.float32)
    src, dst, ea = [], [], []
    for b in m.GetBonds():
        f = bond_features(b)
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        src += [u, v]
        dst += [v, u]
        ea += [f, f]                            # each bond in both directions
    edge_index = np.array([src, dst], dtype=np.int64).reshape(2, -1)
    edge_attr = np.array(ea, dtype=np.float32).reshape(-1, len(EDGE_NAMES))
    return x, edge_index, edge_attr, graph_descriptors(m)


NODE_NAMES = [f'{name}={c}' for name, choices, _ in NODE_SPEC for c in [str(x).split('.')[-1] for x in choices] + ['other']] \
    + ['aromatic', 'in_ring'] + [f'ring_size_{k}' for k in RING_SIZES] + ['mass/100']
EDGE_NAMES = [f'{name}={c}' for name, choices, _ in EDGE_SPEC for c in [str(x).split('.')[-1] for x in choices] + ['other']] \
    + ['conjugated', 'in_ring']
