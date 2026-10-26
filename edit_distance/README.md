# edit_distance

How far is a test crystal from the nearest crystal a model was trained on? This
package answers with two pieces: a **prototype key** that says when two crystals
have the same structure type, and a **substitution distance** that counts the
element swaps between two formulas within a prototype.

## What it does

- `symmetry_key.py`: `prototype_parts(structure)` returns
  `(space group number, sites in the primitive cell, orbit string)`, where the
  orbit string is the sorted multiset of crystallographic orbits of the primitive
  cell with their site-symmetry symbols, e.g. rocksalt `1[m-3m]1[m-3m]`. It does
  not depend on the origin, the setting or the size of the stored cell.
- `edit_distance.py`: the prototype is (space group, anonymised formula,
  primitive sites, orbit string). `substitution_distance` is the smallest number
  of elements to swap, each swap only between elements with the same
  stoichiometric amount (CsPbI3 -> CsPbBr3 is 1, NaCl -> KBr is 2).
  `nearest_training_distance` gives, per test row, the minimum over the training
  rows of the same prototype, capped, or NaN when no training row shares it.

## How to use it

```python
from edit_distance.edit_distance import Prototypes, nearest_training_distance, symmetry_table
sym = symmetry_table(list_of_pymatgen_structures)      # columns: formula, spacegroup, nsites_prim, sitesym
P = Prototypes(sym)
d = nearest_training_distance(P, test_rows, train_rows)  # row indices into sym
```

A precomputed table (parquet or CSV with those four columns) is read with
`load_symmetry(path)`; the command line pools five K-folds and prints the counts
per distance bin:

```
python -m edit_distance.edit_distance --symmetry-table table.parquet --out distances.parquet
python -m edit_distance.example_keys
```

The example builds five crystals in code and prints their keys and distances.

## Where the method comes from

The orbit string is built from spglib's symmetry dataset (A. Togo, K. Shinohara,
I. Tanaka, "Spglib: a software library for crystal symmetry search", 2018).
The idea of binning test crystals by their distance to the training set follows
the leakage analyses of materials benchmarks (e.g. MD-HIT, Li et al. 2023);
the substitution distance is a plain formula-level edit distance restricted to a
prototype.
