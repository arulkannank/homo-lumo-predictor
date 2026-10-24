"""Shared code for the graph neural network notebooks (HOMO + LUMO with PyTorch Geometric).

- Graphs: loads the molecular graphs from graphs/ (CSD_graph_features.ipynb) and builds PyG Data objects with
  targets and whole-graph descriptors standardized on a given set of training rows.
- Models: GINE (optionally with the graph descriptors), AttentiveFP, a Chemprop-style D-MPNN and an MLP on the graph
  descriptors. The GINE / AttentiveFP / MLP classes are the ones CSD_graph_deep_learning.ipynb trained (same
  parameter names), so its saved weights in deep_learning/ load here.
- train_model: AdamW + ReduceLROnPlateau, early stopping on a validation set, a checkpoint every epoch (a rerun
  resumes), finished runs are skipped.
- integrated_gradients: atom / bond / descriptor attributions for one molecule.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import AttentiveFP, GINEConv, global_add_pool, global_mean_pool
from torch_geometric.utils import scatter

TARGETS = ['HOMO', 'LUMO']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training settings of CSD_graph_deep_learning.ipynb (the defaults of train_model)
DEFAULT_TRAIN = {'lr': 1e-3, 'batch_size': 128, 'weight_decay': 1e-5}
TRAIN_KEYS = tuple(DEFAULT_TRAIN)


class Graphs:
    """All molecular graphs of graphs/graphs.npz (same 48,115 rows as every feature file), with targets and
    whole-graph descriptors."""

    def __init__(self, folder='graphs'):
        G = np.load(f'{folder}/graphs.npz')
        self.desc = pd.read_csv(f'{folder}/graph_descriptors.csv')
        with open(f'{folder}/feature_names.json') as f:
            self.names = json.load(f)
        self.x = torch.from_numpy(G['x'])
        self.edge_index = torch.from_numpy(G['edge_index'].astype(np.int64))
        self.edge_attr = torch.from_numpy(G['edge_attr'].astype(np.float32))
        self.node_ptr, self.edge_ptr = G['node_ptr'], G['edge_ptr']
        self.smiles = self.desc['SMILES'].to_numpy()
        self.Y = self.desc[TARGETS].to_numpy(np.float32)
        self.GD = self.desc[self.names['graph']].to_numpy(np.float64)
        self.dims = {'node_dim': self.x.shape[1], 'edge_dim': self.edge_attr.shape[1], 'global_dim': self.GD.shape[1]}

    def __len__(self):
        return len(self.node_ptr) - 1

    def dataset(self, fit_idx):
        """One Data object per molecule, with HOMO/LUMO (y) and the graph descriptors (g) standardized with the
        mean/std of the rows `fit_idx` (descriptors: NaN -> mean, clipped to +-10 std). Returns (data, scaler);
        scaler turns standardized predictions back into eV."""
        y_mean, y_std = self.Y[fit_idx].mean(0), self.Y[fit_idx].std(0)
        g_mean, g_std = np.nanmean(self.GD[fit_idx], 0), np.nanstd(self.GD[fit_idx], 0) + 1e-8
        GD = torch.from_numpy(np.clip(np.nan_to_num((self.GD - g_mean) / g_std), -10, 10).astype(np.float32))
        Ys = torch.from_numpy((self.Y - y_mean) / y_std)
        data = []
        for i in range(len(self)):
            a, b = self.node_ptr[i], self.node_ptr[i + 1]
            c, d = self.edge_ptr[i], self.edge_ptr[i + 1]
            data.append(Data(x=self.x[a:b], edge_index=self.edge_index[:, c:d], edge_attr=self.edge_attr[c:d],
                             y=Ys[i:i + 1], g=GD[i:i + 1]))
        return data, {'y_mean': y_mean, 'y_std': y_std}


class GINENet(nn.Module):
    """GINE message passing (uses bond features) with residual connections, sum + mean pooling and an MLP head.
    With global_dim > 0 the whole-graph descriptors are joined to the pooled graph vector."""

    def __init__(self, node_dim, edge_dim, hidden=256, layers=5, dropout=0.1, global_dim=0, n_out=len(TARGETS)):
        super().__init__()
        self.node_in = nn.Linear(node_dim, hidden)
        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.convs.append(GINEConv(mlp, edge_dim=edge_dim, train_eps=True))
            self.norms.append(nn.BatchNorm1d(hidden))
        self.dropout, self.global_dim = dropout, global_dim
        self.pool_norm = nn.BatchNorm1d(2 * hidden + global_dim)
        self.head = nn.Sequential(nn.Linear(2 * hidden + global_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, n_out))

    def forward(self, data):
        h = self.node_in(data.x)
        for conv, norm in zip(self.convs, self.norms):
            h = h + F.dropout(F.relu(norm(conv(h, data.edge_index, data.edge_attr))), self.dropout, self.training)
        g = torch.cat([global_add_pool(h, data.batch), global_mean_pool(h, data.batch)], dim=1)
        if self.global_dim:
            g = torch.cat([g, data.g], dim=1)
        return self.head(self.pool_norm(g))


class AttentiveFPNet(nn.Module):
    """AttentiveFP (Xiong et al., J. Med. Chem. 2020): graph attention message passing and an attentive readout."""

    def __init__(self, node_dim, edge_dim, hidden=256, layers=4, timesteps=2, dropout=0.1, n_out=len(TARGETS)):
        super().__init__()
        self.net = AttentiveFP(node_dim, hidden, n_out, edge_dim=edge_dim, num_layers=layers,
                               num_timesteps=timesteps, dropout=dropout)

    def forward(self, data):
        return self.net(data.x, data.edge_index, data.edge_attr, data.batch)


class DMPNN(nn.Module):
    """Directed message passing neural network as in Chemprop (Yang et al., J. Chem. Inf. Model. 2019), with
    Chemprop's default architecture: hidden size 300, depth 3, mean aggregation, 2-layer feed-forward head.

    Hidden states live on directed bonds. The message into bond v->w sums the states of the bonds k->v except the
    reverse bond w->v, so information does not flow straight back. graphs.npz stores each bond as the consecutive
    pair (u->v, v->u), so bond e's reverse is e ^ 1 (this also holds in a batch: every graph has an even number of
    directed bonds).
    """

    def __init__(self, node_dim, edge_dim, hidden=300, depth=3, dropout=0.0, n_out=len(TARGETS)):
        super().__init__()
        self.W_i = nn.Linear(node_dim + edge_dim, hidden, bias=False)
        self.W_h = nn.Linear(hidden, hidden, bias=False)
        self.W_o = nn.Linear(node_dim + hidden, hidden)
        self.depth, self.dropout = depth, dropout
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, n_out))

    def forward(self, data):
        x, e = data.x, data.edge_attr
        src, dst = data.edge_index
        n = x.size(0)
        rev = torch.arange(src.numel(), device=x.device) ^ 1
        h0 = F.relu(self.W_i(torch.cat([x[src], e], dim=1)))
        h = h0
        for _ in range(self.depth - 1):
            incoming = scatter(h, dst, dim=0, dim_size=n, reduce='sum')   # sum of the states of bonds k->v, per atom v
            h = F.dropout(F.relu(h0 + self.W_h(incoming[src] - h[rev])), self.dropout, self.training)
        atoms = scatter(h, dst, dim=0, dim_size=n, reduce='sum')
        atoms = F.dropout(F.relu(self.W_o(torch.cat([x, atoms], dim=1))), self.dropout, self.training)
        return self.head(global_mean_pool(atoms, data.batch))


class DescriptorMLP(nn.Module):
    """Baseline: an MLP on the whole-graph descriptors only."""

    def __init__(self, global_dim, hidden=256, dropout=0.1, n_out=len(TARGETS)):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(global_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, n_out))

    def forward(self, data):
        return self.net(data.g)


def build_model(name, dims, **hp):
    """A new model by name; hp = architecture settings (hidden, layers, dropout, ...)."""
    n, e, g = dims['node_dim'], dims['edge_dim'], dims['global_dim']
    if name.startswith('GINE + graph descriptors'):
        return GINENet(n, e, global_dim=g, **hp)
    if name.startswith('GINE'):
        return GINENet(n, e, **hp)
    if name.startswith('AttentiveFP'):
        return AttentiveFPNet(n, e, **hp)
    if name.startswith('D-MPNN'):
        return DMPNN(n, e, **hp)
    if name.startswith('MLP'):
        return DescriptorMLP(g, **hp)
    raise ValueError(name)


def split_config(config):
    """Split a hyperparameter dict into (architecture kwargs, training kwargs)."""
    config = dict(config or {})
    return ({k: v for k, v in config.items() if k not in TRAIN_KEYS},
            {k: config[k] for k in TRAIN_KEYS if k in config})


def tag(name):
    """File-name form of a model name (as CSD_graph_deep_learning.ipynb used)."""
    return name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'plus')


@torch.no_grad()
def predict_std(model, data, idx, batch_size=256):
    """Standardized predictions (n x targets) for the molecules data[idx]."""
    model.eval()
    loader = DataLoader([data[i] for i in idx], batch_size=batch_size, shuffle=False)
    return torch.cat([model(b.to(DEVICE)).cpu() for b in loader]).numpy()


def predict(model, data, idx, scaler, batch_size=256):
    """Predictions in eV (n x targets) for the molecules data[idx]."""
    return predict_std(model, data, idx, batch_size) * scaler['y_std'] + scaler['y_mean']


def _save_atomic(obj, path):
    torch.save(obj, path + '.tmp')
    os.replace(path + '.tmp', path)


def train_model(model_fn, data, fit_idx, val_idx, prefix, lr=1e-3, batch_size=128, weight_decay=1e-5,
                epochs=300, patience=30, seed=42, label=''):
    """Train model_fn() on data[fit_idx] with MSE on the standardized targets, AdamW, ReduceLROnPlateau (x0.5 after
    10 epochs without improvement) and early stopping after `patience` epochs without a better validation loss
    (data[val_idx]); the best epoch is kept.

    Files: {prefix}.ckpt.pt (checkpoint, every epoch: a rerun resumes from it), {prefix}.best.pt (best weights) and
    {prefix}.done.json (summary; when it exists the run is not repeated). Returns (model with the best weights, summary).
    """
    ckpt, best_file, done_file = f'{prefix}.ckpt.pt', f'{prefix}.best.pt', f'{prefix}.done.json'
    torch.manual_seed(seed)
    model = model_fn().to(DEVICE)
    if os.path.exists(done_file):
        model.load_state_dict(torch.load(best_file, map_location=DEVICE))
        with open(done_file) as f:
            return model, json.load(f)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10, min_lr=1e-5)
    state = {'epoch': 0, 'best': float('inf'), 'best_epoch': 0, 'bad': 0, 'history': [], 'seconds': 0.0}
    if os.path.exists(ckpt):
        c = torch.load(ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(c['model'])
        opt.load_state_dict(c['opt'])
        sched.load_state_dict(c['sched'])
        state = c['state']
        print(f'{label}: resuming at epoch {state["epoch"]}', flush=True)
    train_set = [data[i] for i in fit_idx]
    y_val = torch.cat([data[i].y for i in val_idx]).numpy()
    while state['epoch'] < epochs and state['bad'] < patience:
        epoch, t = state['epoch'], time.time()
        torch.manual_seed(seed + epoch)                   # same shuffling order after a restart
        model.train()
        total = 0.0
        # BatchNorm cannot train on a batch of one graph: drop a last batch that would hold a single molecule
        for b in DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=len(train_set) % batch_size == 1):
            b = b.to(DEVICE)
            opt.zero_grad()
            loss = F.mse_loss(model(b), b.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item() * b.num_graphs
        vl = float(((predict_std(model, data, val_idx) - y_val) ** 2).mean())
        sched.step(vl)
        if vl < state['best']:
            state.update(best=vl, best_epoch=epoch, bad=0)
            _save_atomic(model.state_dict(), best_file)
        else:
            state['bad'] += 1
        state['seconds'] += time.time() - t
        state['history'].append({'epoch': epoch, 'train_loss': total / len(train_set), 'val_loss': vl,
                                 'lr': opt.param_groups[0]['lr']})
        state['epoch'] = epoch + 1
        _save_atomic({'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                      'state': state}, ckpt)
        if epoch % 25 == 0 or state['bad'] >= patience:
            print(f"{label:40s} epoch {epoch:3d}  train {total / len(train_set):.4f}  val {vl:.4f}  "
                  f"lr {opt.param_groups[0]['lr']:.1e}  ({time.time() - t:.1f} s/epoch)", flush=True)
    summary = {'epochs': state['epoch'], 'best_epoch': state['best_epoch'], 'best_val_loss': state['best'],
               'minutes': state['seconds'] / 60, 'history': state['history']}
    with open(done_file, 'w') as f:
        json.dump(summary, f)
    model.load_state_dict(torch.load(best_file, map_location=DEVICE))
    return model, summary


def integrated_gradients(model, d, scaler, steps=50):
    """Integrated Gradients (Sundararajan et al., ICML 2017) of every target for one molecule `d` (a Data object).

    Baseline: the same graph with all atom features, bond features and graph descriptors set to zero, so the
    attributions say how the chemistry of each atom and bond (and each descriptor) moves the prediction away from
    that of a featureless graph of the same shape (the topology itself is not attributed). The path integral uses
    the midpoint rule with `steps` points. Everything is in eV, one column per target:

    atom (atoms x T), bond (bonds x T, both directions added), node_feature (node features x T, summed over atoms),
    edge_feature (edge features x T, summed over bonds), global (graph descriptors x T), pred and base (prediction
    for the molecule and for the baseline, T). Completeness: atom.sum + bond.sum + global.sum = pred - base, up to
    the integration error.
    """
    model.eval()
    n_t = len(scaler['y_std'])
    b = Batch.from_data_list([d] * steps).to(DEVICE)
    alpha = (torch.arange(steps, device=DEVICE, dtype=torch.float32) + 0.5) / steps
    x = (b.x * alpha[b.batch, None]).requires_grad_()
    e = (b.edge_attr * alpha[b.batch[b.edge_index[0]], None]).requires_grad_()
    g = (b.g * alpha[:, None]).requires_grad_()
    b.x, b.edge_attr, b.g = x, e, g
    out = model(b)
    n, m = d.x.shape[0], d.edge_attr.shape[0]
    res = {k: [] for k in ('atom', 'bond', 'node_feature', 'edge_feature', 'global')}
    for j in range(n_t):
        gx, ge, gg = torch.autograd.grad(out[:, j].sum(), (x, e, g), retain_graph=j < n_t - 1, allow_unused=True)
        s = float(scaler['y_std'][j])
        ax = (gx.view(steps, n, -1).mean(0).cpu() * d.x).numpy() * s
        ae = (ge.view(steps, m, -1).mean(0).cpu() * d.edge_attr).numpy() * s if m else np.zeros((0, e.shape[1]))
        ag = (gg.mean(0).cpu() * d.g[0]).numpy() * s if gg is not None else np.zeros(d.g.shape[1])
        res['atom'].append(ax.sum(1))
        res['bond'].append(ae.sum(1).reshape(-1, 2).sum(1))
        res['node_feature'].append(ax.sum(0))
        res['edge_feature'].append(ae.sum(0))
        res['global'].append(ag)
    res = {k: np.stack(v, axis=-1) for k, v in res.items()}
    with torch.no_grad():
        ends = Batch.from_data_list([d, d]).to(DEVICE)
        ends.x = ends.x * (ends.batch[:, None] == 1)
        ends.edge_attr = ends.edge_attr * (ends.batch[ends.edge_index[0]][:, None] == 1)
        ends.g = ends.g * torch.tensor([[0.0], [1.0]], device=DEVICE)
        o = model(ends).cpu().numpy() * scaler['y_std'] + scaler['y_mean']
    res['base'], res['pred'] = o[0], o[1]
    return res
