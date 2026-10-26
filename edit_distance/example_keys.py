"""Compute the prototype key for a few crystals built in code, then the nearest-training distance of
each one against the others.

Run (from the repository root):  python -m edit_distance.example_keys
"""
from pymatgen.core import Lattice, Structure

from edit_distance.edit_distance import Prototypes, nearest_training_distance, roles, self_test, symmetry_table


def main():
    self_test()
    structures = {
        "NaCl": Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
        "KBr": Structure.from_spacegroup("Fm-3m", Lattice.cubic(6.60), ["K", "Br"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
        "SrTiO3": Structure.from_spacegroup("Pm-3m", Lattice.cubic(3.905), ["Sr", "Ti", "O"],
                                            [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]]),
        "BaTiO3": Structure.from_spacegroup("Pm-3m", Lattice.cubic(4.00), ["Ba", "Ti", "O"],
                                            [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]]),
        "TiO2 rutile": Structure.from_spacegroup(136, Lattice.tetragonal(4.59, 2.96), ["Ti", "O"],
                                                 [[0, 0, 0], [0.305, 0.305, 0]]),
    }
    sym = symmetry_table(list(structures.values()))
    print("\n  symmetry table (the input every analysis needs):")
    print(sym.to_string(index=False))
    P = Prototypes(sym)
    print("\n  prototype key = (space group, anonymised formula, primitive sites, orbit string):")
    for name, key in zip(structures, P.key):
        print(f"    {name:12s} {key}  roles {roles(sym.formula[list(structures).index(name)])[1]}")
    rows = list(range(len(sym)))
    print("\n  nearest-training distance of each crystal against all the others:")
    for i, name in enumerate(structures):
        d = nearest_training_distance(P, [i], [r for r in rows if r != i])[0]
        print(f"    {name:12s} {'none (no training crystal shares the prototype)' if d != d else int(d)}")


if __name__ == "__main__":
    main()
