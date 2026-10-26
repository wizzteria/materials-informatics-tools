"""Tests of the NASA CEA NASA-9 reader (thermo/cea_reader.py).

Needs the converted YAML (see thermo/cea_reader.py and thermo/cea_convert.py). Set CEA_NASA9_YAML
to its path or place it at data/external/cea_nasa9.yaml; otherwise the module is skipped.

Run:  python -m unittest tests.test_thermo_cea_reader -v
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from thermo.burcat import BurcatRangeError
from thermo.cea_reader import CEADatabase, CEARangeError

ROOT = Path(__file__).resolve().parents[1]
YAML = Path(os.environ.get("CEA_NASA9_YAML") or ROOT / "data" / "external" / "cea_nasa9.yaml")


@unittest.skipUnless(YAML.exists(), f"CEA YAML not found at {YAML}; set CEA_NASA9_YAML")
class TestCEAReader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.db = CEADatabase(YAML)

    def test_all_species_and_phases(self):
        self.assertEqual(len(self.db.blocks), 1831)
        self.assertEqual(sum(b.phase == "C" for b in self.db.blocks), 780)

    def test_NO_is_a_name_not_a_boolean(self):
        # YAML 1.1 reads NO as False; the reader must not
        no = self.db.get("NO")
        self.assertEqual([(b.name, b.phase) for b in no], [("NO", "G")])
        self.assertTrue(all(isinstance(b.name, str) for b in self.db.blocks))

    def test_values_match_cantera(self):
        try:
            import cantera as ct
        except ImportError:
            self.skipTest("cantera is not installed")
        ref = {s.name: s.thermo for s in ct.Species.list_from_file(str(YAML))}
        worst = 0.0
        for b in self.db.blocks:
            th = ref[b.name]
            for lo, hi in zip(b.T_ranges[:-1], b.T_ranges[1:]):
                T = (lo + hi) / 2                      # the middle of each range, no boundary ambiguity
                for mine, theirs in ((b.cp(T), th.cp(T) / 1000), (b.H(T), th.h(T) / 1000), (b.S(T), th.s(T) / 1000)):
                    worst = max(worst, abs(mine - theirs) / max(1.0, abs(theirs)))
        self.assertLess(worst, 1e-9, f"largest relative difference from Cantera: {worst:.2e}")

    def test_missing_phase_file_fails(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            shutil.copy(YAML, tmp / YAML.name)
            with self.assertRaises(FileNotFoundError):
                CEADatabase(tmp / YAML.name)
        finally:
            shutil.rmtree(tmp)

    def test_chain_by_composition(self):
        self.assertEqual([b.name for b in self.db.chain("TiO2", "condensed")], ["TiO2(cr)", "TiO2(L)"])
        self.assertEqual([b.name for b in self.db.chain("Fe", "condensed")],
                         ["Fe(a)", "Fe(a)_2", "Fe(c)", "Fe(d)", "Fe(L)"])
        ge = self.db.chain("Ge", "condensed")
        self.assertEqual([b.name for b in ge], ["Ge(cr)", "Ge(L)"])
        self.assertEqual(ge[0].T_high, ge[1].T_low)           # no 0.4 K gap as in Burcat

    def test_chain_boundaries_are_physical(self):
        for f in ("Fe", "WO3", "Cr2O3", "Ge", "Al2O3", "TiO2", "H2O"):
            ch = self.db.chain(f, "condensed")
            for a, b in zip(ch[:-1], ch[1:]):
                T = b.T_low
                dG = (b.G(T) - a.G(T)) / 1000
                dH = (b.H(T) - a.H(T)) / 1000
                self.assertLess(abs(dG), 0.01, f"{f}: {a.name} -> {b.name} at {T} K, dG {dG:+.4f} kJ/mol")
                self.assertGreater(dH, -0.1, f"{f}: {a.name} -> {b.name} at {T} K, dH {dH:+.3f} kJ/mol")

    def test_out_of_range_is_a_burcat_range_error(self):
        b = self.db.get("TiO2(cr)")[0]
        with self.assertRaises(CEARangeError):
            b.H(3000.0)
        self.assertTrue(issubclass(CEARangeError, BurcatRangeError))
        self.assertTrue(np.isfinite(b.H(3000.0, strict=False)))   # strict=False extrapolates on purpose

    def test_dHf_298(self):
        self.assertAlmostEqual(self.db.get("TiO2(cr)")[0].dHf_298 / 1000, -943.99, delta=0.05)
        self.assertIsNone(self.db.get("TiO2(L)")[0].dHf_298)       # the liquid starts at 2185 K


if __name__ == "__main__":
    unittest.main(verbosity=2)
