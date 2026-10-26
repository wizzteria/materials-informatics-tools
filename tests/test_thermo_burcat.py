"""Checks of the Burcat parser against known values and internal consistency.

Needs BURCAT.THR (see thermo/burcat.py for the download). Set BURCAT_THR to its path or place
it at data/external/BURCAT.THR; otherwise the whole module is skipped.

Run:  python -m unittest tests.test_thermo_burcat -v
"""
import os
import unittest
from pathlib import Path

import numpy as np

from thermo.burcat import (BurcatDatabase, BurcatJoinError, BurcatRangeError,
                           load_defects_csv, parse_formula)

ROOT = Path(__file__).resolve().parents[1]
BURCAT_PATH = Path(os.environ.get('BURCAT_THR') or ROOT / 'data' / 'external' / 'BURCAT.THR')
DEFECTS = ROOT / 'data' / 'burcat_known_defects.csv'

# The only blocks rejected for their coefficients: 'N/A' where a number belongs, and Li3+ with the
# numbers shifted one column. If another one appears, the parser lost a valid block.
UNREADABLE = {9801, 19595, 23727, 24856, 29507, 33353}


def raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    except Exception:  # noqa: BLE001
        return False
    return False


def trapz(y, x):
    """Manual trapezoidal integration, to avoid depending on np.trapz/np.trapezoid."""
    return float(np.sum((y[:-1] + y[1:]) / 2.0 * np.diff(x)))


class TestFormulaParser(unittest.TestCase):
    """Needs no database file."""

    def test_nested_groups_and_case(self):
        self.assertEqual(parse_formula('Ca(HO)2'), {'Ca': 1.0, 'H': 2.0, 'O': 2.0})
        self.assertEqual(parse_formula('Ge(OH)2'), {'Ge': 1.0, 'O': 2.0, 'H': 2.0})
        self.assertEqual(parse_formula('Co'), {'Co': 1.0})               # cobalt
        self.assertEqual(parse_formula('CO'), {'C': 1.0, 'O': 1.0})      # carbon monoxide
        self.assertEqual(parse_formula('Fe0.947O'), {'Fe': 0.947, 'O': 1.0})

    def test_invalid_formulas_return_none(self):
        self.assertIsNone(parse_formula('Xx2'))
        self.assertIsNone(parse_formula('Al2O3)'))


@unittest.skipUnless(BURCAT_PATH.exists(), f'BURCAT.THR not found at {BURCAT_PATH}; set BURCAT_THR')
class TestBurcat(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.db = BurcatDatabase(BURCAT_PATH, defects=DEFECTS)
        cls.by_line = {x.source_line: x for x in cls.db.blocks}

    # ---- block count and the few unreadable records
    def test_rejected_for_coefficients_are_exactly_the_known_ones(self):
        coef_rej = {ln for ln, why in self.db.rejected if why.startswith(('unreadable coefficient', 'both sets'))}
        self.assertEqual(coef_rej, UNREADABLE)
        self.assertTrue(any(b.name.startswith('HT ') for b in self.db.blocks),
                        'HT (tritium hydride) is accepted: its record uses "0  0." as filler')
        nacl = self.by_line.get(31583)
        self.assertIsNotNone(nacl)
        # NaCL(cr) 200-500 K: the unused set is written "0.0       E 00"; H(298.15) and Cp as NBS
        self.assertAlmostEqual(nacl.H(298.15) / 1000, -411.153, delta=0.2)
        self.assertAlmostEqual(nacl.cp(298.15), 50.50, delta=0.1)

    # ---- AL2O3(a)
    def test_alumina_block(self):
        b = self.db.get('AL2O3(a)')[0]
        self.assertAlmostEqual(b.dHf_298 / 1000, -1675.7, delta=1.0)
        self.assertAlmostEqual(b.H(298.15) / 1000, b.dHf_298 / 1000, delta=0.01)
        self.assertAlmostEqual(b.T_mid, 700, delta=5)
        self.assertTrue(b.join_ok)
        self.assertTrue(raises(lambda: b.cp(2327), BurcatRangeError))
        self.assertTrue(np.isfinite(b.cp(2327, strict=False)))      # deliberate extrapolation
        b2 = self.db.pick('Al2O3', 2000, 'condensed')
        self.assertEqual(b2.name, 'AL2O3(b)')

    # ---- H2O, isotopologues and ions
    def test_water_lookup_by_formula(self):
        g = self.db.get_by_formula('H2O', 'G')
        self.assertEqual([x.name for x in g], ['H2O'])              # no isotopologues, no H2O+
        for x in g:
            self.assertAlmostEqual(x.dHf_298 / 1000, -241.826, delta=0.5)
            self.assertAlmostEqual(x.T_mid, 1000.0, delta=1.0)
        self.assertEqual([x.name for x in self.db.get_by_formula('D2O', 'G')], ['D2O'])
        self.assertEqual([x.name for x in self.db.get_by_formula('H2O', 'G', charge=1.0)], ['H2O+'])
        self.assertEqual(len(self.db.get_by_formula('H2O', 'condensed')), 2)

    # ---- Bi(L): no join, with a recorded choice
    def test_bismuth_liquid_uses_recorded_set(self):
        bl = self.db.get_by_formula('Bi', 'L')
        self.assertEqual(len(bl), 2)
        self.assertFalse(bl[0].join_ok)
        self.assertEqual(bl[0].forced_set, 'high')
        self.assertTrue(bl[0].forced_reason)
        self.assertIn((bl[0].source_line, 'use_set_high'), self.db.defects_applied)

    # ---- H against the integral of Cp inside a block
    def test_enthalpy_is_integral_of_cp(self):
        a = self.db.get('AL2O3(a)')[0]
        Ts = np.linspace(298.15, 1000, 20000)
        H_int = trapz(np.array([a.cp(T) for T in Ts]), Ts)
        H_pol = a.H(1000) - a.H(298.15)
        self.assertLess(abs(H_int - H_pol) / 1000, 0.01)

    # ---- continuity and transition enthalpies between blocks
    def test_transitions_between_blocks(self):
        a, ab, liq = self.db.get('AL2O3(a)')[0], self.db.get('AL2O3(b)')[0], self.db.get('AL2O3(L)')[0]
        self.assertLess(abs(ab.H(1200) - a.H(1200)), 100.0)
        fus = (liq.H(2327) - ab.H(2327)) / 1000
        self.assertTrue(100 < fus < 125, f'Al2O3 fusion at 2327 K = {fus:.2f} kJ/mol (about 110)')
        bl = self.db.get_by_formula('Bi', 'L')
        bs = self.db.get_by_formula('Bi', 'S')
        self.assertEqual([x.name for x in bs], ['Bi(S) REF ELEME'])
        fus_bi = (bl[0].H(544.52) - bs[0].H(544.52)) / 1000
        self.assertAlmostEqual(fus_bi, 11.28, delta=0.5)             # measured about 11.3
        fus_bi_nostrict = (bl[0].H(544.52, strict=False) - bs[0].H(544.52, strict=False)) / 1000
        self.assertAlmostEqual(fus_bi_nostrict, fus_bi, delta=1e-9)   # strict=False does not choose blindly
        self.assertLess(abs(bl[1].H(1200) - bl[0].H(1200)), 50.0)
        al = self.db.get_by_formula('Al', 'S')
        self.assertEqual([x.name for x in al], ['AL(cr) REF ELEME'])
        al = al[0]
        self.assertAlmostEqual(al.S(298.15), 28.30, delta=0.1)
        self.assertAlmostEqual(al.cp(298.15), 24.20, delta=0.1)
        self.assertAlmostEqual(al.cp(900), 33.25, delta=0.3)
        self.assertLess(abs(al.H(298.15)), 1.0)

    # ---- two-set blocks without a clean join
    def test_blocks_without_join_refuse_to_compute(self):
        bad = [x for x in self.db.blocks if not x.join_ok]
        free = next(x for x in bad if x.forced_set is None)
        self.assertTrue(raises(lambda: free.H(1000), BurcatJoinError))
        self.assertTrue(raises(lambda: free.H(1000, strict=False), BurcatJoinError))
        self.assertAlmostEqual(free.H(1000, use_set='high'), free.H_high(1000), delta=1e-9)
        # two CHO2 isomers do not join; the minimum without them would be a blind choice
        self.assertTrue(raises(lambda: self.db.pick('CHO2', 300, 'G'), BurcatJoinError))

    # ---- coefficient 15 against the polynomial in every block containing 298.15 K
    def test_polynomial_matches_field_15_except_registered_defects(self):
        checks = [(x, x.dHf_check()) for x in self.db.blocks]
        checks = [(x, d) for x, d in checks if d is not None]
        off = {x.source_line for x, d in checks if abs(d) > 0.5}
        registered = {int(r['burcat_line']) for r in load_defects_csv(DEFECTS)
                      if r['defect_type'] == 'internal_inconsistency' and r['burcat_line']
                      and self.by_line[int(r['burcat_line'])].contains(298.15)}
        self.assertEqual(off, registered)

    # ---- coverage of a list of metals and their oxides, by phase (no ions)
    def test_coverage_of_metals_and_oxides(self):
        species = ['Al', 'AlO', 'Al2O3', 'O2', 'H2O', 'H2', 'OH',
                   'Bi', 'BiO', 'Bi2O3', 'I', 'I2', 'IO', 'I2O5',
                   'Cu', 'CuO', 'Fe', 'Fe2O3', 'Mo', 'MoO3',
                   'W', 'WO3', 'Co', 'CoO', 'Co3O4', 'Ni', 'NiO',
                   'Mn', 'MnO2', 'Mn2O3', 'Cr', 'Cr2O3',
                   'Sn', 'SnO2', 'Sb', 'Sb2O3', 'Zr', 'ZrO2',
                   'Ti', 'TiO2', 'Ru', 'RuO2']
        gas = {c for c in species if self.db.get_by_formula(c, 'gas')}
        cond = {c for c in species if self.db.get_by_formula(c, 'condensed')}
        # IO leaves the formula index through the register (exclude_from_formula_index, unit error)
        self.assertEqual((len(gas), len(cond)), (26, 20))
        self.assertNotIn('IO', gas)
        nio = self.db.get_by_formula('NiO', 'condensed')
        self.assertEqual(len(nio), 4)                                 # includes the C-phase records

    # ---- defect register
    def test_defect_register(self):
        raw = BurcatDatabase(BURCAT_PATH, defects=None)
        rb = next(x for x in raw.get_by_formula('Bi', 'L') if x.T_low == 544.52)
        self.assertIsNone(rb.forced_set)
        self.assertTrue(raises(lambda: rb.H(700), BurcatJoinError))   # the choice lives in the CSV
        base = {'source': 'Burcat', 'species': 'KNO3', 'override_rule': 'flag_only', 'burcat_line': '29129', '_row': 0}
        self.assertTrue(raises(lambda: raw.validate_defects([{**base, 'burcat_line': '29130'}]), ValueError))
        self.assertTrue(raises(lambda: raw.validate_defects([{**base, 'species': 'KN'}]), ValueError))
        self.assertTrue(raises(lambda: raw.validate_defects([{**base, 'override_rule': 'use_nbs'}]), ValueError))
        rows = load_defects_csv(DEFECTS)
        self.assertEqual(len(raw.validate_defects(rows)),
                         sum(1 for r in rows if r['source'] == 'Burcat' and r['burcat_line']))
        before = raw.pick('Li', 300, 'gas')
        raw.apply_defects([r for r in rows if r['burcat_line'] in ('29213', '32766')])
        after = raw.pick('Li', 300, 'gas')
        # set_phase_S: Li(cr) is flagged G in the file; after the rule it leaves the gas lookup
        self.assertEqual(before.source_line, 29213)
        self.assertIsNotNone(after)
        self.assertEqual(after.phase, 'G')
        self.assertNotEqual(after.source_line, 29213)
        # exclude_from_formula_index: PT (tritium phosphide) leaves the PH index, stays reachable by name
        ph = raw.get_by_formula('PH', 'gas')
        self.assertTrue(ph and all(x.source_line != 32766 for x in ph))
        name = next(b.name for b in raw.blocks if b.source_line == 32766)
        self.assertTrue(any(x.source_line == 32766 for x in raw.get(name)))

    # ---- references (text between braces is an alternative source, not the adopted one)
    def test_references_from_comments(self):
        self.assertFalse(any('{' in x.ref or '}' in x.ref for x in self.db.blocks))
        al_g = self.by_line[195]
        self.assertEqual((al_g.ref, al_g.ref_hf298), ('Cox CODATA 1989', 'Cox CODATA 1989'))
        oh = self.by_line[28247]
        self.assertTrue(oh.ref_hf298.startswith('Ruscic et al JPC 106'))
        alo = self.by_line[377]
        self.assertEqual((alo.ref, alo.ref_code), ('-', 'tpis96'))   # header code (Gurvich 1996), no REF= outside braces


if __name__ == '__main__':
    unittest.main(verbosity=2)
