"""Crystal structure -> graph (framework-agnostic numpy).

A CrystalGraph holds:
  z          (n_nodes,)        atomic numbers
  edge_src   (n_edges,)        index of the center atom
  edge_dst   (n_edges,)        index of the neighbor atom (may repeat: periodic images)
  edge_dist  (n_edges,)        distance in Angstrom
  edge_rbf   (n_edges, n_rbf)  Gaussian expansion of the distance (CGCNN style)

Two knobs decide what the GNN can see: `cutoff` (radius in A) and `max_neighbors`
(keep only the k closest). Both are explored on NaCl and CsPbI3 in __main__.

Run:  python -m crystal_graph.graph
"""
from dataclasses import dataclass

import numpy as np
from pymatgen.core import Structure


@dataclass
class CrystalGraph:
    z: np.ndarray
    edge_src: np.ndarray
    edge_dst: np.ndarray
    edge_dist: np.ndarray
    edge_rbf: np.ndarray

    @property
    def n_nodes(self): return len(self.z)
    @property
    def n_edges(self): return len(self.edge_src)

    def degree(self):
        return np.bincount(self.edge_src, minlength=self.n_nodes)


def gaussian_rbf(d: np.ndarray, dmin=0.0, dmax=8.0, step=0.2, gamma=None) -> np.ndarray:
    centers = np.arange(dmin, dmax + step, step)
    gamma = gamma or 1.0 / step ** 2
    return np.exp(-gamma * (d[:, None] - centers[None, :]) ** 2)


def build_graph(s: Structure, cutoff: float = 6.0, max_neighbors: int = 12, rbf_max: float = 8.0,
                with_rbf: bool = True, reverse_ties: bool = False) -> CrystalGraph:
    """Neighbors within `cutoff`, truncated to the `max_neighbors` closest per atom.
    with_rbf=False skips the Gaussian expansion (the model computes it on the fly; saves cache space).
    reverse_ties=True reverses the neighbour list before the (stable) sort, so atoms tied at the cap
    are kept in the opposite order: a diagnostic for the sensitivity of a prediction to the order in
    which tied neighbours are truncated, never meant for a reported prediction. Default behaviour is
    unchanged."""
    src, dst, dist = [], [], []
    all_nb = s.get_all_neighbors(cutoff)  # periodic images included
    for i, nbs in enumerate(all_nb):
        if reverse_ties:
            nbs = list(reversed(nbs))
        nbs = sorted(nbs, key=lambda n: n.nn_distance)[:max_neighbors]
        for n in nbs:
            src.append(i); dst.append(n.index); dist.append(n.nn_distance)
    src, dst, dist = (np.asarray(src, dtype=np.int32), np.asarray(dst, dtype=np.int32), np.asarray(dist, dtype=np.float32))
    z = np.array([site.specie.Z for site in s], dtype=np.int16)
    rbf = gaussian_rbf(dist, dmax=rbf_max) if with_rbf else np.zeros((len(dist), 0), dtype=np.float32)
    return CrystalGraph(z, src, dst, dist, rbf)


def isolated_nodes(g: CrystalGraph) -> int:
    return int((g.degree() == 0).sum())


if __name__ == "__main__":
    from pymatgen.core import Lattice
    nacl = Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    cspbi3 = Structure.from_spacegroup("Pm-3m", Lattice.cubic(6.29), ["Cs", "Pb", "I"], [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]])

    print(f"{'structure':8s} {'cutoff':>6s} {'max_nb':>6s} {'nodes':>5s} {'edges':>5s} {'isolated':>8s}  degree per node")
    for name, s in [("NaCl", nacl), ("CsPbI3", cspbi3)]:
        for cutoff, k in [(3.5, 12), (4.0, 12), (5.0, 12), (6.0, 12), (8.0, 12), (8.0, 6), (8.0, 24)]:
            g = build_graph(s, cutoff=cutoff, max_neighbors=k)
            print(f"{name:8s} {cutoff:6.1f} {k:6d} {g.n_nodes:5d} {g.n_edges:5d} {isolated_nodes(g):8d}  {g.degree().tolist()}")

    g = build_graph(cspbi3, cutoff=6.0, max_neighbors=12)
    print("\nCsPbI3 cutoff 6.0: node 0 (Cs) neighbors and distances:")
    m = g.edge_src == 0
    print("  Z:", g.edge_dst[m].tolist(), " d:", g.edge_dist[m].round(2).tolist())
    print("  rbf shape per edge:", g.edge_rbf.shape[1], "gaussians; first edge, 5 strongest centers:",
          np.argsort(g.edge_rbf[0])[-5:][::-1].tolist())
