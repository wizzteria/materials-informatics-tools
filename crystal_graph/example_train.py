"""Minimal end-to-end example: build graphs for three crystals made in code, batch them, and run
two optimizer steps of the CGCNN on random targets. It only shows that the API fits together.

Run (from the repository root):  python -m crystal_graph.example_train
"""
import numpy as np
import torch
from pymatgen.core import Lattice, Structure

from crystal_graph.cgcnn import CGCNN, count_params
from crystal_graph.graph import build_graph


def example_structures():
    nacl = Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    si = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])
    al2o3 = Structure.from_spacegroup("R-3c", Lattice.hexagonal(4.76, 12.99), ["Al", "O"],
                                      [[0, 0, 0.352], [0.306, 0, 0.25]])
    return {"NaCl": nacl.get_primitive_structure(), "Si": si.get_primitive_structure(),
            "Al2O3": al2o3.get_primitive_structure()}


def collate(graphs):
    """Concatenate graphs into one batch; node indices of later graphs are offset."""
    z, src, dst, dist, batch = [], [], [], [], []
    offset = 0
    for k, g in enumerate(graphs):
        z.append(g.z); src.append(g.edge_src + offset); dst.append(g.edge_dst + offset); dist.append(g.edge_dist)
        batch.append(np.full(g.n_nodes, k, dtype=np.int64))
        offset += g.n_nodes
    return (torch.as_tensor(np.concatenate(z), dtype=torch.long),
            torch.as_tensor(np.concatenate(src), dtype=torch.long),
            torch.as_tensor(np.concatenate(dst), dtype=torch.long),
            torch.as_tensor(np.concatenate(dist), dtype=torch.float32),
            torch.as_tensor(np.concatenate(batch), dtype=torch.long), len(graphs))


def main():
    torch.manual_seed(0)
    structures = example_structures()
    graphs = [build_graph(s, cutoff=6.0, max_neighbors=12, with_rbf=False) for s in structures.values()]
    for name, g in zip(structures, graphs):
        print(f"{name:6s} nodes {g.n_nodes:3d} edges {g.n_edges:4d}")
    z, src, dst, dist, batch, n = collate(graphs)
    model = CGCNN()
    print(f"CGCNN parameters: {count_params(model):,}")
    y_metal = torch.tensor([0.0, 0.0, 1.0])
    y_gap = torch.log1p(torch.tensor([5.0, 1.1, 0.0]))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for step in range(2):
        opt.zero_grad()
        logit, gap = model(z, src, dst, dist, batch, n)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y_metal) + ((gap - y_gap) ** 2).mean()
        loss.backward()
        opt.step()
        print(f"step {step + 1}: loss {loss.item():.4f}")
    print("ok")


if __name__ == "__main__":
    main()
