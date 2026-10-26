"""CGCNN-style crystal graph network in plain PyTorch (no graph library).

The idea is the one of a convolutional network on an image: there, a pixel's new value depends on
its neighbouring pixels; here, each ATOM's new vector depends on its neighbour atoms through the
edges of the graph. Same idea, irregular grid.

Layers:
  embed     : atomic number -> 64-dim atom vector (learned)
  rbf       : bond distance -> 41 gaussians (the "pixel intensity" of an edge)
  conv x3   : message from neighbor j to atom i, gated (CGCNN, Xie & Grossman 2018)
  pool      : mean over the atoms of each crystal -> one crystal vector
  heads     : metal logit (classification) and log1p(gap) (regression on non-metals)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RBF(nn.Module):
    def __init__(self, dmin=0.0, dmax=8.0, step=0.2):
        super().__init__()
        self.register_buffer("centers", torch.arange(dmin, dmax + step, step))
        self.gamma = 1.0 / step ** 2

    def forward(self, d):
        return torch.exp(-self.gamma * (d[:, None] - self.centers[None, :]) ** 2)


class CGConv(nn.Module):
    """h_i <- softplus(h_i + sum_j sigmoid(W_f z_ij) * softplus(W_s z_ij)),  z_ij = [h_i, h_j, e_ij]"""

    def __init__(self, dim, edge_dim):
        super().__init__()
        # W [h_i, h_j, e_ij] == W_i h_i + W_j h_j + W_e e_ij : the node parts are computed once per
        # ATOM and gathered per edge, instead of once per EDGE (12x fewer matmul rows on CPU)
        self.lin_i = nn.Linear(dim, 2 * dim)
        self.lin_j = nn.Linear(dim, 2 * dim, bias=False)
        self.lin_e = nn.Linear(edge_dim, 2 * dim, bias=False)
        self.bn_msg = nn.BatchNorm1d(2 * dim)
        self.bn_out = nn.BatchNorm1d(dim)

    def forward(self, h, e, src, dst):
        z = self.lin_i(h)[src] + self.lin_j(h)[dst] + self.lin_e(e)
        z = self.bn_msg(z)
        gate, core = z.chunk(2, dim=1)
        msg = torch.sigmoid(gate) * F.softplus(core)
        agg = torch.zeros_like(h).index_add_(0, src, msg)
        return F.softplus(h + self.bn_out(agg))


class CGCNN(nn.Module):
    def __init__(self, dim=64, n_conv=3, n_rbf=41, hidden=128):
        super().__init__()
        self.embed = nn.Embedding(100, dim)
        self.rbf = RBF()
        self.convs = nn.ModuleList([CGConv(dim, n_rbf) for _ in range(n_conv)])
        self.fc = nn.Sequential(nn.Linear(dim, hidden), nn.Softplus())
        self.head_metal = nn.Linear(hidden, 1)
        self.head_gap = nn.Linear(hidden, 1)

    def forward(self, z, edge_src, edge_dst, edge_dist, batch, n_graphs):
        h = self.embed(z)
        e = self.rbf(edge_dist)
        for conv in self.convs:
            h = conv(h, e, edge_src, edge_dst)
        # mean pooling per crystal
        pooled = torch.zeros(n_graphs, h.shape[1], device=h.device).index_add_(0, batch, h)
        counts = torch.bincount(batch, minlength=n_graphs).clamp(min=1).unsqueeze(1).to(h.dtype)
        g = self.fc(pooled / counts)
        return self.head_metal(g).squeeze(1), self.head_gap(g).squeeze(1)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
