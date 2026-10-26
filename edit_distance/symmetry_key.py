"""One symmetry routine for every symmetry table built with this package.

Two encodings of a crystal's symmetry live here.

prototype_parts(structure)  ->  (space group number, sites in the primitive cell, orbit string)
    The CANONICAL encoding. The orbit string is the multiset of crystallographic orbits of the
    PRIMITIVE cell, each written as "<multiplicity>[<site-symmetry symbol>]" and sorted, e.g.
    rocksalt "1[m-3m]1[m-3m]", rutile "2[m.mm]4[m.2m]". Site-symmetry symbols and orbit
    multiplicities do not depend on the origin or the setting spglib happens to choose, and the
    primitive cell removes the dependence on how large a cell was stored. Both replace an earlier
    Wyckoff-letter multiset, which re-keyed a sizeable share of identical crystals under a pure
    origin shift and split one compound stored in two cell sizes into two prototypes.

space_group_and_wyckoff(structure)  ->  (space group number, Wyckoff-letter multiset of the given cell)
    The earlier encoding, kept so that old tables can be reproduced and compared.

Both use symprec 0.1 and angle tolerance 5 degrees. A structure whose symmetry cannot be
determined returns (-1, 0, "") or (-1, "").
"""
from collections import Counter, defaultdict

import numpy as np
import spglib
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

SYMPREC = 0.1
ANGLE_TOL = 5.0
KEY_VERSION = "sitesym-primitive"


def _dataset(structure, symprec=SYMPREC):
    cell = (structure.lattice.matrix, structure.frac_coords, [s.specie.Z for s in structure])
    ds = spglib.get_symmetry_dataset(cell, symprec=symprec, angle_tolerance=ANGLE_TOL)
    if ds is None:
        return None
    if isinstance(ds, dict):                       # older spglib
        class _D:                                  # noqa: D401
            pass
        o = _D()
        for k, v in ds.items():
            setattr(o, k, v)
        return o
    return ds


def prototype_parts(structure, symprec=SYMPREC):
    """(space group number, number of sites in the primitive cell, sorted orbit string) or (-1, 0, "")."""
    try:
        ds = _dataset(structure, symprec)
        if ds is None:
            return -1, 0, ""
        number = int(ds.number)
        orbits = getattr(ds, "crystallographic_orbits", None)
        if orbits is None:
            orbits = ds.equivalent_atoms
        orbits = np.asarray(orbits)
        to_prim = np.asarray(ds.mapping_to_primitive)
        symbols = list(ds.site_symmetry_symbols)
        prim_atoms = defaultdict(set)
        symbol_of = {}
        for i, (orb, p) in enumerate(zip(orbits, to_prim)):
            prim_atoms[int(orb)].add(int(p))
            symbol_of.setdefault(int(orb), symbols[i])
        n_prim = len(set(int(p) for p in to_prim))
        parts = sorted((len(prim_atoms[o]), symbol_of[o]) for o in prim_atoms)
        seq = "".join(f"{m}[{sym}]" for m, sym in parts)
        if number <= 0 or not seq or n_prim <= 0:
            return -1, 0, ""
        return number, n_prim, seq
    except Exception:                              # noqa: BLE001
        return -1, 0, ""


def space_group_and_wyckoff(structure, symprec=SYMPREC):
    """The earlier encoding: (space group number, Wyckoff-letter multiset of the given cell)."""
    try:
        ds = SpacegroupAnalyzer(structure, symprec=symprec).get_symmetry_dataset()
        number = getattr(ds, "number", None)
        letters = getattr(ds, "wyckoffs", None)
        if number is None:                       # older pymatgen returns a dict
            number, letters = ds["number"], ds["wyckoffs"]
        seq = "".join(f"{n}{w}" for w, n in sorted(Counter(letters).items()))
        return int(number), seq
    except Exception:                            # noqa: BLE001
        return -1, ""


def self_test():
    """Rocksalt in three cells and two origins must give one key; a distortion must not."""
    from pymatgen.core import Lattice, Structure
    prim = Structure(Lattice.from_parameters(4.0, 4.0, 4.0, 60, 60, 60), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    conv = Structure(Lattice.cubic(5.64), ["Na"] * 4 + ["Cl"] * 4,
                     [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
                      [0.5, 0.5, 0.5], [0, 0, 0.5], [0, 0.5, 0], [0.5, 0, 0]])
    sup = conv.copy()
    sup.make_supercell([2, 1, 1])
    shifted = conv.copy()
    shifted.translate_sites(list(range(len(shifted))), [0.13, 0.27, 0.41], frac_coords=True)
    keys = [prototype_parts(s) for s in (prim, conv, sup, shifted)]
    print("  rocksalt primitive / conventional / 2x1x1 / origin-shifted:", keys)
    assert len(set(keys)) == 1 and keys[0][0] == 225 and keys[0][1] == 2, keys
    old = [space_group_and_wyckoff(s) for s in (conv, sup)]
    print("  old letter encoding, conventional vs 2x1x1 (differs, by cell size):", old)
    rut = Structure.from_spacegroup(136, Lattice.tetragonal(4.59, 2.96), ["Ti", "O"], [[0, 0, 0], [0.305, 0.305, 0]])
    print("  rutile:", prototype_parts(rut), "| letters:", space_group_and_wyckoff(rut))
    dist = conv.copy()
    dist.translate_sites([0], [0.03, 0, 0], frac_coords=True)
    print("  rocksalt with one atom moved 0.17 A (should NOT be 225):", prototype_parts(dist))
    print("  symmetry_key self-test passed")


if __name__ == "__main__":
    self_test()
