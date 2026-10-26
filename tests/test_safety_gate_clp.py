"""Tests of safety_gate/clp_validation.py, written before any hazard column of the Annex VI table was read: the name
parser on named cases, the code-to-level mapping, the criterion. The one test that reads the ECHA workbook skips
unless CLP_ANNEX_VI_XLSX points at a copy of it (download page: https://echa.europa.eu/information-on-chemicals/annex-vi-to-clp).
Run (from the repository root):  python -m unittest discover -s tests -p "test_safety_gate*.py" -v
"""
import os
import unittest

from safety_gate import clp_validation as v


class TestParser(unittest.TestCase):

    def formula(self, name):
        kind, f, note, states = v.parse_name(name)
        return kind, f

    def test_oxides_and_hydroxides(self):
        self.assertEqual(self.formula("beryllium oxide"), ("compound", "BeO"))
        self.assertEqual(self.formula("diboron trioxide"), ("compound", "B2O3"))
        self.assertEqual(self.formula("manganese dioxide"), ("compound", "MnO2"))
        self.assertEqual(self.formula("dinickel trioxide"), ("compound", "Ni2O3"))
        self.assertEqual(self.formula("nickel monoxide [1]\nnickel oxide [2]\nbunsenite [3]"), ("compound", "NiO"))
        self.assertEqual(self.formula("chromium (VI) trioxide"), ("compound", "CrO3"))
        self.assertEqual(self.formula("nickel dihydroxide [1]\nnickel hydroxide [2]"), ("compound", "Ni(HO)2"))
        self.assertEqual(self.formula("sodium hydroxide; caustic soda"), ("compound", "NaHO"))
        self.assertEqual(self.formula("divanadium pentaoxide; vanadium pentoxide"), ("compound", "V2O5"))
        self.assertEqual(self.formula("osmium tetroxide"), ("compound", "OsO4"))

    def test_oxyanion_salts(self):
        self.assertEqual(self.formula("potassium dichromate"), ("compound", "K2Cr2O7"))
        self.assertEqual(self.formula("potassium perchlorate"), ("compound", "KClO4"))
        self.assertEqual(self.formula("barium chlorate"), ("compound", "Ba(ClO3)2"))
        self.assertEqual(self.formula("cobalt dinitrate"), ("compound", "Co(NO3)2"))
        self.assertEqual(self.formula("iron (II) sulfate"), ("compound", "FeSO4"))
        self.assertEqual(self.formula("sodium hydrogensulphate"), ("compound", "NaHSO4"))
        self.assertEqual(self.formula("nickel dipotassium bis(sulfate) [1]"), ("compound", "K2Ni(SO4)2"))
        self.assertEqual(self.formula("potassium permanganate"), ("compound", "KMnO4"))

    def test_groups_formulas_and_out_of_scope(self):
        kind, f, note, states = v.parse_name("Chromium (VI) compounds, with the exception of barium chromate")
        self.assertEqual((kind, f, states), ("group", "Cr", {"Cr": [6]}))
        self.assertEqual(self.formula("lead compounds with the exception of those specified elsewhere in this Annex"), ("group", "Pb"))
        self.assertEqual(self.formula("potassium titanium oxide (K2Ti6O13)"), ("compound", "K2Ti6O13"))
        self.assertEqual(self.formula("sodium peroxide")[0], "peroxide")
        self.assertEqual(self.formula("cobalt di(acetate)")[0], "out of scope")
        self.assertEqual(self.formula("nickel dichloride")[0], "out of scope")
        self.assertEqual(self.formula("dichlorvos (ISO); 2,2-dichlorovinyl dimethyl phosphate")[0], "out of scope")
        self.assertEqual(self.formula("calcium hypochlorite")[0], "out of scope")

    def test_cases_from_the_independent_verification(self):
        """Names that an independent verification of a first run showed were parsed wrongly or dropped."""
        self.assertEqual(self.formula("antimony trioxide"), ("compound", "Sb2O3"))
        self.assertEqual(self.formula("arsenic pentoxide"), ("compound", "As2O5"))
        self.assertEqual(self.formula("disodium octaborate"), ("compound", "Na2B8O13"))
        self.assertEqual(self.formula("cadmium (non-pyrophoric) [1]" + chr(10) + "cadmium oxide (non-pyrophoric) [2]"), ("compound", "CdO"))
        self.assertEqual(self.formula("nickel triuranium decaoxide"), ("compound", "U3NiO10"))
        self.assertEqual(self.formula("potassium ferrite"), ("compound", "KFeO2"))
        self.assertEqual(self.formula("zinc sulphate hydrous"), ("compound", "ZnSO4"))
        self.assertEqual(self.formula("gadolinium(III) sulfite"), ("compound", "Gd2(SO3)3"))
        self.assertEqual(self.formula("organic compounds of mercury with the exception of those specified elsewhere"), ("out of scope", ""))
        self.assertEqual(self.formula("inorganic compounds of mercury with the exception of mercuric sulphide"), ("group", "Hg"))
        self.assertEqual(self.formula("arsenic acid and its salts"), ("group", "As"))
        self.assertEqual(self.formula("barium salts, with the exception of barium sulphate"), ("group", "Ba"))
        self.assertEqual(self.formula("product-by-process definition polyazodyestuff compounds sodium")[0], "unparsed")
        self.assertEqual(self.formula("sodium peroxometaborate")[0], "peroxide")
        self.assertEqual(self.formula("perboric acid (H3BO2(O2)), monosodium salt")[0], "peroxide")
        self.assertEqual(v.gate_level("", "peroxide")[0], "NONE")

    def test_states_are_recorded(self):
        self.assertEqual(v.parse_name("chromium (VI) trioxide")[3], {"Cr": [6]})
        self.assertEqual(v.parse_name("manganese dioxide")[3], {"Mn": [4]})
        self.assertEqual(v.parse_name("cobalt oxide")[3], {"Co": [2]})


class TestLevels(unittest.TestCase):

    def test_code_mapping(self):
        self.assertEqual(v.level_from_codes("Carc. 1B\nMuta. 2\nAcute Tox. 3"), "WARNING")
        self.assertEqual(v.level_from_codes("Acute Tox. 2 *\nSkin Corr. 1B"), "WARNING")
        self.assertEqual(v.level_from_codes("Repr. 1A"), "WARNING")
        self.assertEqual(v.level_from_codes("Acute Tox. 3\nSTOT RE 2"), "CAUTION")
        self.assertEqual(v.level_from_codes("Carc. 2"), "CAUTION")
        self.assertEqual(v.level_from_codes("Ox. Sol. 2\nSkin Corr. 1A\nAquatic Acute 1"), "NONE")
        self.assertEqual(v.level_from_codes("Acute Tox. 4"), "NONE")
        self.assertEqual(v.level_from_codes(""), "NONE")

    def test_gate_level_from_rules(self):
        self.assertEqual(v.gate_level("CdO", "compound")[0], "WARNING")
        self.assertEqual(v.gate_level("Co(NO3)2", "compound")[0], "WARNING")
        self.assertEqual(v.gate_level("MnSO4", "compound")[0], "CAUTION")
        self.assertEqual(v.gate_level("Cr", "group", {"Cr": [6]})[0], "WARNING")
        self.assertEqual(v.gate_level("MgO", "compound")[0], "NONE")

    def test_criterion_thresholds(self):
        self.assertEqual((v.SENS_WARN, v.SENS_CAUTION), (0.95, 0.90))


class TestWorkbook(unittest.TestCase):

    @unittest.skipUnless(os.path.isfile(os.environ.get("CLP_ANNEX_VI_XLSX", "")),
                         "set CLP_ANNEX_VI_XLSX to the ECHA Annex VI Excel export to run this test")
    def test_workbook_reads_and_parses(self):
        sheets, rows = v.read_sheet(os.environ["CLP_ANNEX_VI_XLSX"], names_only=True)
        self.assertGreater(len(rows), 1000)
        self.assertTrue(all("name" in r and "index" in r for r in rows))
        kinds = {v.parse_name(r["name"])[0] for r in rows[:200]}
        self.assertTrue(kinds & {"compound", "group", "out of scope"})


if __name__ == "__main__":
    unittest.main()
