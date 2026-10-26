"""Tests of safety_gate/safety_gate.py, one positive and one negative case per rule S1 to S12. The rules are labels:
nothing is removed.
Run (from the repository root):  python -m unittest discover -s tests -p "test_safety_gate*.py" -v
"""
import datetime
import json
import os
import tempfile
import unittest

from pymatgen.core import Lattice, Structure

from safety_gate import safety_gate as sg


def codes(text):
    import re
    return re.findall(r"\[(S\d+)\]", text or "")


class TestS1Elements(unittest.TestCase):

    def test_s1_warning_for_group_1_carcinogen_and_none_for_benign(self):
        rules = sg.element_rules("CdO", {"Cd": [2]})
        self.assertTrue(any(lv == "WARNING" and "[S1] Cd" in why for lv, why in rules))
        self.assertEqual([r for r in sg.element_rules("MgO", {"Mg": [2]}) if "[S1]" in r[1]], [])

    def test_s1_caution_for_group_2(self):
        rules = sg.element_rules("Sb2O3", {"Sb": [3]})
        self.assertTrue(any(lv == "CAUTION" and "[S1] Sb" in why for lv, why in rules))
        self.assertFalse(any(lv == "WARNING" and "[S1] Sb" in why for lv, why in rules))

    def test_rows_added_after_the_clp_validation(self):
        """Co WARNING (Annex VI Carc. 1B / Repr. 1B salts), Mn CAUTION, Se WARNING; Sn unchanged (no row)."""
        self.assertTrue(any(lv == "WARNING" and "[S1] Co" in why and "027-001-00-9" in why for lv, why in sg.element_rules("Co(OH)3", {"Co": [3]})))
        self.assertTrue(any(lv == "CAUTION" and "[S1] Mn" in why for lv, why in sg.element_rules("Mn(OH)3", {"Mn": [3]})))
        self.assertFalse(any(lv == "WARNING" and "[S1] Mn" in why for lv, why in sg.element_rules("MnO2", {"Mn": [4]})))
        self.assertTrue(any(lv == "WARNING" and "[S1] Se" in why for lv, why in sg.element_rules("Na2SeO3", None)))
        self.assertEqual([r["element"] for r in sg.rules() if r["element"] in ("Sn",)], [])

    def test_anion_clp_note_is_a_note_not_a_level(self):
        self.assertIn("Carc. 1B", sg.anion_clp_note("KBrO3"))
        self.assertIn("Acute Tox. 3", sg.anion_clp_note("KClO3"))
        self.assertIn("nitrite", sg.anion_clp_note("NaNO2"))
        self.assertEqual(sg.anion_clp_note("Fe2O3"), "")
        self.assertEqual(sg.anion_clp_note("KBr"), "")
        self.assertNotIn("anion", sg.gate("KBrO3")["hazard_warning"] + sg.gate("KBrO3")["hazard_caution"])

    def test_s1_state_specific_rule(self):
        cr6 = sg.element_rules("CrO3", {"Cr": [6]})
        cr3 = sg.element_rules("Cr(OH)3", {"Cr": [3]})
        self.assertTrue(any(lv == "WARNING" and "Cr(6)" in why for lv, why in cr6))
        self.assertFalse(any("Cr(6)" in why for lv, why in cr3))          # the Cr(VI) rule does not fire at +3
        self.assertTrue(any("[S2] Cr" in why for lv, why in cr3))          # but the any-state Cr caution does
        no_state = sg.element_rules("CrO3", None)                           # no state information: applies (conservative)
        self.assertTrue(any("Cr(6)" in why for lv, why in no_state))

    def test_s1_radioactive(self):
        self.assertTrue(any("radioactive" in why for lv, why in sg.element_rules("ThO2", {"Th": [4]})))
        self.assertFalse(any("radioactive" in why for lv, why in sg.element_rules("ZrO2", {"Zr": [4]})))

    def test_applies_to_conditions(self):
        self.assertTrue(sg._applies("hydroxide", "Cu(OH)2"))
        self.assertFalse(sg._applies("hydroxide", "CuO"))
        self.assertTrue(sg._applies("formula:SiO2", "SiO2"))
        self.assertFalse(sg._applies("formula:SiO2", "MgSiO3"))
        self.assertTrue(sg._applies("with:Mg,Fe", "Mg3Si2O5(OH)4"))
        self.assertFalse(sg._applies("with:Mg,Fe", "CaSiO3"))
        self.assertTrue(sg._applies("hydroxide&with:Mg", "Mg3Si2O5(OH)4"))

    def test_table_levels_are_labels(self):
        self.assertEqual(sg._level("EXCLUDE"), "WARNING")
        self.assertEqual(sg._level("CAUTION"), "CAUTION")
        self.assertEqual(len(sg.rules()), 31)
        self.assertTrue(all(r["level"] in ("EXCLUDE", "CAUTION") for r in sg.rules()))
        self.assertTrue(all(r["verified"].startswith("confirmed") for r in sg.rules()))   # one row: 'confirmed (a minor correction ...)'
        self.assertTrue(all(r["source_url"].startswith("http") for r in sg.rules()))


class TestS2Volatile(unittest.TestCase):

    def test_s2_warning_for_osmium_none_for_iron(self):
        self.assertTrue(any(lv == "WARNING" and "[S2] Os" in why for lv, why in sg.element_rules("OsO2", {"Os": [4]})))
        self.assertEqual([r for r in sg.element_rules("Fe2O3", {"Fe": [3]}) if "[S2]" in r[1]], [])

    def test_s2_caution_for_rhenium(self):
        self.assertTrue(any(lv == "CAUTION" and "[S2] Re" in why for lv, why in sg.element_rules("Re2O5", {"Re": [5]})))


class TestS3Scope(unittest.TestCase):

    def test_scope(self):
        self.assertEqual(sg.scope_check("KClO4")[0], "WARNING")                # Cl is outside the oxide/hydroxide scope
        self.assertEqual(sg.scope_check("Fe(OH)3")[0], "NONE")
        self.assertEqual(sg.scope_check("Sb2O5")[0], "NONE")                   # scope metalloid

    def test_oo_bond(self):
        per = Structure(Lattice.cubic(6.0), ["Ba", "O", "O"], [[0, 0, 0], [0.5, 0.5, 0.3], [0.5, 0.5, 0.3 + 1.49 / 6.0]])
        lv, why, d = sg.oo_bond_check(per)
        self.assertEqual(lv, "WARNING")
        self.assertIn("peroxide", why)
        sup = Structure(Lattice.cubic(6.0), ["K", "O", "O"], [[0, 0, 0], [0.5, 0.5, 0.3], [0.5, 0.5, 0.3 + 1.33 / 6.0]])
        self.assertIn("superoxide", sg.oo_bond_check(sup)[1])
        mgo = Structure.from_spacegroup("Fm-3m", Lattice.cubic(4.21), ["Mg", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        lv, why, d = sg.oo_bond_check(mgo)
        self.assertEqual(lv, "NONE")
        self.assertGreater(d, 2.5)
        self.assertEqual(sg.oo_bond_check(None)[0], "NONE")


class TestS4Thermal(unittest.TestCase):

    def test_thresholds(self):
        self.assertAlmostEqual(sg.T_WARN_K, 743.0)
        self.assertAlmostEqual(sg.T_CAUTION_K, 903.0)
        self.assertEqual(sg.thermal_check(655.0)[0], "WARNING")
        self.assertEqual(sg.thermal_check(743.0)[0], "WARNING")               # at the threshold
        self.assertEqual(sg.thermal_check(744.0)[0], "CAUTION")
        self.assertEqual(sg.thermal_check(903.0)[0], "CAUTION")
        self.assertEqual(sg.thermal_check(904.0)[0], "NONE")
        self.assertEqual(sg.thermal_check(float("nan"))[0], "NONE")           # not evaluated: no label from S4


class TestS5S6Gate(unittest.TestCase):
    """S6 reads LIT_FILE; the test points it at a temporary review file with one reviewed composition."""

    def setUp(self):
        self._old = sg.LIT_FILE
        fd, self.path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"rows": [{"formula": "Mn(OH)3 (from a review)"}]}, fh)
        sg.LIT_FILE = self.path

    def tearDown(self):
        sg.LIT_FILE = self._old
        os.remove(self.path)

    def test_s5_label_for_a_new_composition_and_s6_without_review(self):
        self.assertTrue(sg.literature_reviewed("Mn(OH)3"))
        g = sg.gate("Mn(OH)3", None, {"Mn": [3]}, float("nan"), known_compound=False)
        self.assertNotIn("S6", codes(g["hazard_warning"]))
        self.assertTrue(g["lab_use"].startswith("[S5]"))                       # a new composition: not characterized
        self.assertFalse(sg.literature_reviewed("Hf(OH)4"))
        g3 = sg.gate("Hf(OH)4", None, {"Hf": [4]}, float("nan"), known_compound=False)
        self.assertIn("[S6]", g3["hazard_warning"])                             # no review for an invented composition
        self.assertEqual(g3["hazard_level"], "WARNING")
        self.assertEqual(g3["lab_use"], sg.NOT_FOR_LAB)                        # every WARNING carries the exclusion statement

    def test_s6_without_any_review_file(self):
        sg.LIT_FILE = self.path + ".missing"
        self.assertFalse(sg.literature_reviewed("Mn(OH)3"))
        self.assertIn("[S6]", sg.gate("MgO", known_compound=True)["hazard_warning"])

    def test_lab_use_statements(self):
        self.assertEqual(sg.current_lab_use(sg.NOT_FOR_LAB_LEGACY), sg.NOT_FOR_LAB)
        self.assertEqual(sg.current_lab_use(sg.KNOWN_LAB_USE), sg.KNOWN_LAB_USE)
        self.assertNotIn("[S", sg.NOT_FOR_LAB)                                 # report cards parse [S#] tags; none here
        self.assertIn("does not imply", sg.COMPUTATIONAL_ONLY)


class TestS7S8(unittest.TestCase):

    def test_s7_caution_for_element_without_evidence(self):
        self.assertNotIn("Hf", sg.EVIDENCE_ELEMENTS)
        self.assertTrue(any("[S7] Hf" in why and lv == "CAUTION" for lv, why in sg.element_rules("HfO2", {"Hf": [4]})))
        self.assertFalse(any("[S7]" in why for lv, why in sg.element_rules("Fe2O3", {"Fe": [3]})))

    def test_s8_recheck_dates(self):
        self.assertEqual(sg.recheck_due(datetime.date(2026, 10, 10)), [])
        due = sg.recheck_due(datetime.date(2027, 11, 1))
        self.assertEqual(len(due), 3)
        self.assertTrue(any("IARC" in why for d, why in due))


class TestS10toS12(unittest.TestCase):

    def test_s10_s11_statements(self):
        self.assertIn("GHS 2.14", sg.S10)
        self.assertIn("reducing agents", sg.S11)

    def test_s12_dust_note(self):
        fully = sg.dust_note("TiO2", {"Ti": [4]})
        self.assertIn("completely oxidized", fully)
        partly = sg.dust_note("Mn(OH)3", {"Mn": [3]})
        self.assertIn("could oxidize further", partly)
        unresolved = sg.dust_note("MnO2", None)
        self.assertIn("not resolved", unresolved)
        self.assertIn("combustible metal dust", fully)                         # the metal-powder note is always present


if __name__ == "__main__":
    unittest.main()
