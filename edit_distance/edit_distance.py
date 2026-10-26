"""Edit distance from a crystal to the nearest crystal a model was trained on.

DEFINITION

Two crystals are in the same PROTOTYPE when they share
    - the space group number,
    - the anonymised reduced formula (NaCl and KBr are both "AB"; CsPbI3 and SrTiO3 are both "ABC3"),
    - the number of sites in the PRIMITIVE cell,
    - the multiset of crystallographic orbits of the primitive cell with their site-symmetry symbols
      (symmetry_key.prototype_parts), e.g. rocksalt "1[m-3m]1[m-3m]".
The last two replaced "sites in the stored cell" and a Wyckoff-letter multiset: the letters depended on
the origin spglib chose (a sizeable share of identical crystals were re-keyed under an origin shift) and
the stored cell size split one compound into several prototypes. The concept is unchanged; only its
encoding was made canonical. It is still a proxy for "same structure type"; a StructureMatcher check on
a sample measures how often it is wrong.

Within a prototype, the SUBSTITUTION DISTANCE between two crystals is the smallest number of elements
that must be swapped to turn one formula into the other, where an element may only take the place of an
element with the same stoichiometric amount:

    CsPbI3 -> CsPbBr3   1      (I -> Br)
    CsPbI3 -> KSnBr3    3
    NaCl   -> KBr       2
    SrTiO3 -> BaTiO3    1
    same formula        0      (a polymorph or a duplicate in another cell)

For amount classes with several elements (e.g. the two "1"s in ABC3), the best pairing is used, which is
the class size minus the number of elements the two formulas share in that class.

The NEAREST-TRAINING DISTANCE of a test crystal is the minimum substitution distance to any crystal in
its fold's training rows with the same prototype, or "none" when no training crystal shares the prototype.

This file only counts crystals per distance. It does not read any prediction or error, on purpose: the
definition is written first, and the counts only tell whether each distance bin has enough crystals.

Input: a SYMMETRY TABLE, one row per crystal, with the columns
    formula       reduced or full formula (string)
    spacegroup    space group number; -1 (or 0) when the symmetry could not be determined
    nsites_prim   number of sites in the primitive cell
    sitesym       the orbit string of symmetry_key.prototype_parts
symmetry_table() builds one from pymatgen structures; load_symmetry() reads one from parquet or CSV.

Run:  python -m edit_distance.edit_distance --symmetry-table TABLE.parquet [--out OUT.parquet] [--gap-column gap]
      python -m edit_distance.edit_distance --self-test
"""
import os
import time
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from sklearn.model_selection import KFold

from edit_distance.symmetry_key import prototype_parts

KEY_COLUMNS = ("formula", "spacegroup", "nsites_prim", "sitesym")


def roles(formula):
    """(anonymised formula, {amount: frozenset(elements)}) for a reduced formula."""
    c = Composition(formula).reduced_composition
    by_amt = defaultdict(set)
    for el, amt in c.items():
        by_amt[round(float(amt), 6)].add(el.symbol)
    return c.anonymized_formula, {a: frozenset(s) for a, s in by_amt.items()}


def substitution_distance(ra, rb):
    """Minimum element swaps between two role maps with the same anonymised formula."""
    if ra.keys() != rb.keys():
        return None
    return sum(len(ra[a]) - len(ra[a] & rb[a]) for a in ra)


def self_test():
    cases = [("CsPbI3", "CsPbBr3", 1), ("CsPbI3", "KSnBr3", 3), ("NaCl", "KBr", 2),
             ("SrTiO3", "BaTiO3", 1), ("Al2O3", "Fe2O3", 1), ("Fe2O3", "Fe2O3", 0),
             ("BaTiO3", "TiBaO3", 0), ("CsPbI3", "PbCsI3", 0), ("KNO3", "NaNO3", 1)]
    for a, b, want in cases:
        (fa, ra), (fb, rb) = roles(a), roles(b)
        got = substitution_distance(ra, rb) if fa == fb else None
        assert got == want, f"{a} vs {b}: got {got}, want {want}"
        fast = _dist(tuple(ra[k] for k in sorted(ra)), tuple(rb[k] for k in sorted(rb)))
        assert fast == want, f"fast path {a} vs {b}: got {fast}, want {want}"
    # different anonymised formulas are never in the same prototype
    assert roles("NaCl")[0] != roles("CaF2")[0]
    print(f"  self-test passed: {len(cases)} known pairs give the expected distance")


def symmetry_table(structures, formulas=None):
    """Build the symmetry table from pymatgen structures: one row per structure with the key columns.
    `formulas` defaults to each structure's reduced formula."""
    rows = []
    for i, s in enumerate(structures):
        g, n, ss = prototype_parts(s)
        f = formulas[i] if formulas is not None else s.composition.reduced_formula
        rows.append(dict(formula=f, spacegroup=g, nsites_prim=n, sitesym=ss))
    return pd.DataFrame(rows, columns=list(KEY_COLUMNS))


def load_symmetry(path):
    """Read a symmetry table (parquet or CSV) and check that the canonical key columns are present.
    Refuses a table that lacks them, so no analysis can silently fall back to a weaker key."""
    sym = pd.read_csv(path) if str(path).lower().endswith(".csv") else pd.read_parquet(path)
    for col in KEY_COLUMNS:
        if col not in sym.columns:
            raise SystemExit(f"{path} has no {col} column: the key needs {', '.join(KEY_COLUMNS)}")
    return sym


class Prototypes:
    """Everything that does not depend on the fold, computed once for the whole dataset."""

    def __init__(self, sym):
        if "sitesym" not in sym.columns or "nsites_prim" not in sym.columns:
            raise ValueError("the prototype key needs nsites_prim and sitesym columns; use load_symmetry() or symmetry_table()")
        forms = sym.formula.to_numpy()
        cache = {f: roles(f) for f in pd.unique(forms)}
        self.formula = forms
        # role map as a tuple of frozensets ordered by amount: two formulas with the same
        # anonymised formula have the same amounts, so the tuples line up position by position
        self.rolemap = {f: tuple(r[1][a] for a in sorted(r[1])) for f, r in cache.items()}
        # a failed symmetry determination is stored as -1 (or an empty orbit string); such
        # crystals get no prototype, so they are never matched to each other
        self.key = [(int(g), cache[f][0], int(n), str(ss)) if (int(g) > 0 and str(ss) and int(n) > 0) else None
                    for g, f, n, ss in zip(sym.spacegroup.to_numpy(), forms, sym.nsites_prim.to_numpy(),
                                           sym.sitesym.to_numpy())]


def _dist(ra, rb):
    return sum(len(x) - len(x & y) for x, y in zip(ra, rb))


def nearest_training_distance(P, test_rows, train_rows, cap=4):
    """For each test row: min substitution distance to a same-prototype training row, capped.

    Returns a float array: 0..cap (cap means 'cap or more'), NaN when no training crystal shares
    the prototype.
    """
    train_forms = defaultdict(set)
    for r in train_rows:
        if P.key[r] is not None:
            train_forms[P.key[r]].add(P.formula[r])
    train_forms = {k: (v, [P.rolemap[f] for f in v]) for k, v in train_forms.items()}
    memo = {}
    out = np.full(len(test_rows), np.nan)
    for i, r in enumerate(test_rows):
        k, f = P.key[r], P.formula[r]
        if k not in train_forms:
            continue
        if (k, f) in memo:
            out[i] = memo[(k, f)]
            continue
        forms, maps = train_forms[k]
        if f in forms:
            best = 0
        else:
            rt, best = P.rolemap[f], cap
            for rm in maps:
                d = _dist(rt, rm)
                if d < best:
                    best = d
                    if best == 1:          # 0 is impossible here: the formula is not in the set
                        break
        memo[(k, f)] = best
        out[i] = best
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--symmetry-table", help="parquet or CSV with the columns formula, spacegroup, nsites_prim, sitesym")
    ap.add_argument("--out", help="parquet to write (row, fold, nearest_train_distance)")
    ap.add_argument("--gap-column", help="optional band-gap column: also split the counts by metallicity")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    self_test()
    if a.self_test or not a.symmetry_table:
        return
    sym = load_symmetry(a.symmetry_table)
    n = len(sym)
    print(f"\n  {n} crystals, prototype = (space group, anonymised formula, primitive-cell sites, orbit string)")
    splits = list(KFold(n_splits=5, shuffle=True, random_state=0).split(np.zeros(n)))
    rows = []
    t0 = time.time()
    P = Prototypes(sym)
    print(f"  {len(P.rolemap):,} distinct formulas, {len(set(P.key)):,} prototypes "
          f"({time.time() - t0:.0f}s)", flush=True)
    for k, (tr, te) in enumerate(splits):
        d = nearest_training_distance(P, te, tr)
        rows.append(d)
        print(f"  fold {k} done ({time.time() - t0:.0f}s)", flush=True)
    d = np.concatenate(rows)

    labels = ["0 (same formula)", "1", "2", "3", "4 or more", "no prototype neighbour"]
    counts = [(d == 0).sum(), (d == 1).sum(), (d == 2).sum(), (d == 3).sum(), (d >= 4).sum(),
              np.isnan(d).sum()]
    print("\n  TEST CRYSTALS BY DISTANCE TO THE NEAREST TRAINING CRYSTAL (all 5 folds pooled)")
    print("  counts only; no prediction or error has been read\n")
    for lab, c in zip(labels, counts):
        print(f"    {lab:<24}{c:>8,}  {100 * c / n:5.1f} %")
    if a.gap_column:
        gap = sym[a.gap_column].values[np.concatenate([s[1] for s in splits])]
        metal = gap < 0.01
        print("\n  split by metallicity (band gap must be analysed separately for metals, see MD-HIT)")
        print(f"    {'distance':<24}{'metals':>8}{'non-metals':>12}")
        for lab, m in zip(labels, [d == 0, d == 1, d == 2, d == 3, d >= 4, np.isnan(d)]):
            print(f"    {lab:<24}{(m & metal).sum():>8,}{(m & ~metal).sum():>12,}")

    if a.out:
        out = pd.DataFrame({"row": np.concatenate([s[1] for s in splits]),
                            "fold": np.concatenate([np.full(len(s[1]), k) for k, s in enumerate(splits)]),
                            "nearest_train_distance": d})
        out.sort_values("row").to_parquet(a.out)
        print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
