"""Tests of thermo/shomate_reader.py (NIST Shomate equations). Needs only data/nist_shomate.csv.

Run:  python -m unittest tests.test_thermo_shomate_reader -v
"""
from __future__ import annotations

import unittest
from pathlib import Path

from thermo.burcat import BurcatRangeError
from thermo.shomate_reader import ShomateDatabase, ShomateRangeError

ROOT = Path(__file__).resolve().parents[1]
DB = ShomateDatabase(ROOT / "data" / "nist_shomate.csv")


class TestShomate(unittest.TestCase):

    def test_blocks(self):
        self.assertEqual(sorted(b.name for b in DB.blocks), ["Co3O4(solid)", "CoO(solid)"])
        coo = DB.chain("CoO", "condensed")[0]
        self.assertEqual((coo.T_low, coo.T_high, len(coo.segments)), (298.0, 3000.0, 2))
        self.assertEqual(DB.chain("CoO", "gas"), [])

    def test_fit_reproduces_its_own_298_values(self):
        # miscopied F and G would show here: JANAF tabulates dHf -910.02 and S 114.44 for Co3O4
        b = DB.chain("Co3O4", "condensed")[0]
        self.assertAlmostEqual(b.H(298.15) / 1000.0, -910.02, delta=0.1)
        self.assertAlmostEqual(b.S(298.15), 114.44, delta=0.05)
        self.assertAlmostEqual(b.dHf_298, -910020.0, delta=0.01)

    def test_equations_are_consistent(self):
        # dH/dT = Cp and dS/dT = Cp/T: tests the formulas, not the coefficients
        for b in DB.blocks:
            for T in (400.0, 900.0, 1400.0):
                h = 1e-3
                self.assertAlmostEqual((b.H(T + h) - b.H(T - h)) / (2 * h), b.cp(T), delta=1e-3)
                self.assertAlmostEqual((b.S(T + h) - b.S(T - h)) / (2 * h), b.cp(T) / T, delta=1e-6)

    def test_segments_join(self):
        b = DB.chain("CoO", "condensed")[0]
        lo, hi = b.segments[0][2], b.segments[1][2]
        T, t = 1600.0, 1.6
        H = lambda c: c[0]*t + c[1]*t**2/2 + c[2]*t**3/3 + c[3]*t**4/4 - c[4]/t + c[5]
        self.assertAlmostEqual(H(lo), H(hi), delta=0.02)        # kJ/mol

    def test_range_is_enforced(self):
        b = DB.chain("Co3O4", "condensed")[0]
        with self.assertRaises(ShomateRangeError):
            b.cp(1600.0)
        self.assertTrue(issubclass(ShomateRangeError, BurcatRangeError))


if __name__ == "__main__":
    unittest.main(verbosity=2)
