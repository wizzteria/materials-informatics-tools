# crystal_graph

Turns a pymatgen `Structure` into a plain numpy graph and trains a CGCNN-style
graph network on it with nothing but PyTorch (no torch_geometric, no DGL).

## What it does

- `graph.py`: `build_graph(structure, cutoff, max_neighbors)` returns a
  `CrystalGraph` (atomic numbers, edge source/destination, distances and an
  optional Gaussian expansion of the distances). Periodic images are included;
  each atom keeps its `max_neighbors` closest neighbours within `cutoff`.
- `cgcnn.py`: `CGCNN` with an element embedding, a radial basis on the bond
  length, gated convolutions (`CGConv`), mean pooling per crystal and two heads
  (a metal/non-metal logit and a regression head, meant for log1p of the band gap).
  The two node-side linear maps of each convolution are applied once per atom and
  gathered per edge, which keeps CPU training affordable.

## How to use it

```python
from pymatgen.core import Lattice, Structure
from crystal_graph.graph import build_graph
from crystal_graph.cgcnn import CGCNN

s = Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
g = build_graph(s, cutoff=6.0, max_neighbors=12, with_rbf=False)
model = CGCNN()
# batch several graphs by offsetting node indices; see example_train.py (collate)
```

Run the example from the repository root:

```
python -m crystal_graph.example_train
```

It builds NaCl, Si and Al2O3 primitive cells in code, batches them and runs two
optimizer steps on made-up targets.

## Where the method comes from

T. Xie and J. C. Grossman, "Crystal Graph Convolutional Neural Networks for an
Accurate and Interpretable Prediction of Material Properties", Phys. Rev. Lett.
120, 145301 (2018). The gated message, the Gaussian distance expansion and the
mean pooling follow that paper; the batching and the two-head output are ours.
