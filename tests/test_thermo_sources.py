"""Tests of thermo/thermo_sources.py: NBS anchor, Burcat chain, boundaries, CEA as the tertiary
source, Shomate as the fourth, and the exclusions of the defect register.

These cases live in the unified layer, not in the parsers:

  - the NBS anchor respects the phase (H2O gas, not liquid; Bi2O3 and CuO gas)
  - H(298.15) comes from the anchor without a chain, for species only needed at 298 K
  - the Burcat chain is validated at every boundary (KNO3 at 607.7 K)
  - a chain never mixes blocks of two sources
  - species excluded by the register (IO) are rejected
  - every Property declares its source and its type
  - TiO2 comes from CEA with its fusion; Al2O3 from Burcat with the internal join at 699.5 K

Needs BURCAT.THR (BURCAT_THR); the TiO2 and WO3 cases also need the CEA YAML (CEA_NASA9_YAML).
Without the files the tests skip.

Run:  python -m unittest tests.test_thermo_sources -v
"""
from __future__ import annotations

import unittest

from thermo import thermo_sources
from thermo.thermo_sources import (
    get_species, Property,
    SpeciesNotFound, SpeciesPropertyMissing, FrontierInconsistency,
    SourceType,
)

HAVE_BURCAT = thermo_sources.BURCAT_PATH.exists()
HAVE_CEA = thermo_sources.CEA_YAML.exists()
NEEDS_CEA = unittest.skipUnless(HAVE_CEA, f"CEA YAML not found at {thermo_sources.CEA_YAML}; set CEA_NASA9_YAML")

# Tolerances
ANCHOR_TOL_J = 200.0        # J/mol; NBS values are kJ/mol with 1-2 decimals
CONTINUITY_TOL_J = 500.0    # J/mol; at the internal join of a Burcat block
PHYSICAL_TOL_J = 100.0      # J/mol; for monotonicity and fusion checks


@unittest.skipUnless(HAVE_BURCAT, f"BURCAT.THR not found at {thermo_sources.BURCAT_PATH}; set BURCAT_THR")
class Base(unittest.TestCase):
    """Shared helpers."""

    def H(self, formula, phase, T):
        sp = get_species(formula, phase)
        return sp.H(T), sp

    def anchor(self, formula, phase, expected_J):
        H, sp = self.H(formula, phase, 298.15)
        self.assertAlmostEqual(
            H, expected_J, delta=ANCHOR_TOL_J,
            msg=f"{formula} ({phase}) H(298.15) = {H:.1f} J/mol, "
                f"expected ~{expected_J:.1f}; declared source "
                f"{sp.dHf_298.source_ref}",
        )
        return sp


# --------------------------------------------------------------- NBS anchors

class TestAnchorsNBS(Base):
    """H(298.15) matches the NBS value of the right phase."""

    def test_alumina_solid(self):
        self.anchor("Al2O3", "solid", -1675700.0)

    def test_water_gas_uses_gas_not_liquid(self):
        # NBS: H2O(g) = -241.818 kJ/mol, H2O(l) = -285.83.
        # If the anchor ignored the phase the result would be -285830 and fail here.
        sp = self.anchor("H2O", "gas", -241818.0)
        self.assertEqual(sp.dHf_298.source_ref, "NBS")
        self.assertEqual(sp.phase, "gas")

    def test_bi2o3_gas_anchor_not_solid(self):
        # The solid is -573.88; the gas is +376.8 kJ/mol in NBS.
        H, sp = self.H("Bi2O3", "gas", 298.15)
        self.assertGreater(H, 200_000.0,
                           f"Bi2O3(g) H(298.15) = {H:.0f} J/mol; "
                           f"should be positive (~376 kJ/mol). Anchored to the solid?")

    def test_cuo_gas_anchor_not_solid(self):
        H, _ = self.H("CuO", "gas", 298.15)
        self.assertGreater(H, 100_000.0,
                           f"CuO(g) H(298.15) = {H:.0f} J/mol; should be positive (~306 kJ/mol)")

    def test_zro2_gas_anchor_not_solid(self):
        H, _ = self.H("ZrO2", "gas", 298.15)
        self.assertGreater(H, -500_000.0,
                           f"ZrO2(g) H(298.15) = {H:.0f} J/mol; should be less negative than the solid")


# --------------------------------------------------------------- internal join

class TestJoinInsideBlock(Base):
    """The join between the two coefficient sets of a Burcat block."""

    def test_alumina_join_continuity(self):
        """H with the low set and with the high set agree at T_mid."""
        sp = get_species("Al2O3", "solid")
        blocks = [b for b in sp._chain if not b.single_range]
        self.assertTrue(blocks, "Al2O3 should have at least one block with two sets")
        b = blocks[0]
        self.assertIsNotNone(b.T_mid)
        self.assertAlmostEqual(b.T_mid, 699.5, delta=15.0,
                               msg=f"T_mid = {b.T_mid:.1f}, expected ~699.5")
        dH = b.H_high(b.T_mid) - b.H_low(b.T_mid)
        self.assertLess(abs(dH), CONTINUITY_TOL_J, f"jump at the join: {dH:.1f} J/mol")


# --------------------------------------------------------------- monotonicity

class TestChainMonotonicity(Base):
    """H(T) increases; Cp(T) is positive."""

    def test_alumina_H_monotonic(self):
        sp = get_species("Al2O3", "solid")
        H_prev = None
        for T in (298.15, 500, 700, 800, 1000, 1200, 1500, 1800, 2000):
            H = sp.H(T)
            if H_prev is not None:
                self.assertGreater(H, H_prev - PHYSICAL_TOL_J,
                                   f"H({T}) = {H:.1f} J/mol < H(prev) = {H_prev:.1f}")
            H_prev = H

    def test_alumina_cp_positive(self):
        sp = get_species("Al2O3", "solid")
        for T in (298.15, 500, 1000, 1500, 2000):
            Cp = sp.Cp(T)
            self.assertGreater(Cp, 0.0, f"Cp({T}) = {Cp:.2f} J/(mol·K) cannot be <= 0")


# --------------------------------------------------------------- fusions

class TestFusionsPhysical(Base):
    """Transitions between blocks are endothermic and of reasonable size."""

    @NEEDS_CEA
    def test_tio2_fusion(self):
        """TiO2: solid to liquid at 2185 K, from the CEA chain."""
        sp = get_species("TiO2", "solid")
        chain = sp._chain
        if len(chain) < 2:
            self.skipTest("TiO2 has no two-block chain")
        a, b = chain[0], chain[1]
        Tf = b.T_low
        dHfus = b.H(Tf) - a.H(Tf)
        self.assertGreater(dHfus, 10_000.0, f"TiO2 fusion at {Tf} K: {dHfus/1000:.1f} kJ/mol; expected > 10")
        self.assertLess(dHfus, 200_000.0, f"TiO2 fusion: {dHfus/1000:.1f} kJ/mol; too large")

    @NEEDS_CEA
    def test_no_exothermic_transitions_in_chain(self):
        """Every boundary between consecutive blocks of Al2O3 or TiO2 has dH >= 0."""
        for formula in ("Al2O3", "TiO2"):
            sp = get_species(formula, "solid")
            for a, b in zip(sp._chain[:-1], sp._chain[1:]):
                Tf = b.T_low
                dH = b.H(Tf, strict=False) - a.H(Tf, strict=False)
                self.assertGreaterEqual(dH, -PHYSICAL_TOL_J,
                                        f"{formula}: boundary at {Tf} K with dH = {dH:.1f} J/mol "
                                        f"({a.name} -> {b.name})")


# --------------------------------------------------------------- boundaries

class TestFrontierProtection(Base):
    """Broken boundaries are detected and stop the calculation."""

    def test_kno3_below_frontier_ok(self):
        sp = get_species("KNO3", "solid")
        self.assertLess(sp.H(400.0), sp.H(500.0), "H should grow with T")

    def test_kno3_above_frontier_raises(self):
        """KNO3 above 607.7 K raises FrontierInconsistency (register: no_calculate_above_T)."""
        sp = get_species("KNO3", "solid")
        with self.assertRaises(FrontierInconsistency):
            sp.H(700.0)


# --------------------------------------------------------------- species at 298 K

class TestReactantsAt298WithoutChain(Base):
    """Species with an NBS value but no Burcat chain can still give H(298.15)."""

    def test_moo3_H298_from_anchor(self):
        """MoO3 is not in Burcat as a solid; H(298.15) comes from the NBS anchor."""
        H = get_species("MoO3", "solid").H(298.15)
        self.assertAlmostEqual(H, -745090.0, delta=500.0)

    def test_i2o5_H298_from_anchor(self):
        H = get_species("I2O5", "solid").H(298.15)
        self.assertAlmostEqual(H, -158070.0, delta=500.0)

    def test_bi2o3_solid_H298_from_anchor(self):
        H = get_species("Bi2O3", "solid").H(298.15)
        self.assertAlmostEqual(H, -573880.0, delta=500.0)

    def test_bi2o3_solid_Cp_raises(self):
        """Solid Bi2O3 has Cp(T) in none of the digitised sources."""
        sp = get_species("Bi2O3", "solid")
        with self.assertRaises(SpeciesPropertyMissing):
            sp.Cp(1000.0)


# --------------------------------------------------------------- exclusions

class TestExclusions(Base):
    """Species marked exclude_from_formula_index do not resolve."""

    def test_io_not_found(self):
        """IO has a unit error in the file and is excluded from the index."""
        with self.assertRaises((SpeciesNotFound, SpeciesPropertyMissing)):
            get_species("IO", "gas")


# --------------------------------------------------------------- provenance

class TestProvenance(Base):
    """Every Property declares source and type."""

    def test_all_properties_have_source(self):
        for formula, phase in [
            ("Al2O3", "solid"), ("H2O", "gas"), ("Bi2O3", "solid"),
            ("CuO", "solid"), ("TiO2", "solid"), ("Fe2O3", "solid"),
            ("Bi", "liquid"),
        ]:
            sp = get_species(formula, phase)
            for name, prop in [("dHf_298", sp.dHf_298), ("S_298", sp.S_298), ("Cp_298", sp.Cp_298)]:
                self.assertIsInstance(prop, Property, f"{formula}.{name} is not a Property")
                if prop.value is not None:
                    self.assertTrue(prop.source_ref, f"{formula}.{name} without source_ref")
                    self.assertNotEqual(prop.source_ref, "missing",
                                        f"{formula}.{name} marked missing but has a value")
                    self.assertIsInstance(prop.source_type, SourceType)

    def test_burcat_is_marked_estimated(self):
        """Burcat is a fit to evaluated compilations, not a direct measurement."""
        sp = get_species("Al2O3", "solid")
        if sp.Cp_298.value is not None:
            self.assertEqual(sp.Cp_298.source_type, SourceType.ESTIMATED,
                             f"Cp(298) of Al2O3 comes from {sp.Cp_298.source_ref} and should be ESTIMATED")


# --------------------------------------------------------------- no mixing

class TestNoMixing(Base):
    """A chain never mixes blocks of two sources."""

    def test_alumina_chain_is_all_burcat(self):
        from thermo.burcat import BurcatBlock
        sp = get_species("Al2O3", "solid")
        for b in sp._chain:
            self.assertIsInstance(b, BurcatBlock,
                                  f"Al2O3: block {getattr(b, 'name', b)} is not a BurcatBlock")

    def test_tio2_chain_is_all_same_class(self):
        sp = get_species("TiO2", "solid")
        if not sp._chain:
            self.skipTest("TiO2 without a chain (CEA YAML not installed)")
        types = {type(b).__name__ for b in sp._chain}
        self.assertEqual(len(types), 1, f"TiO2 mixes block types in the chain: {types}")


# --------------------------------------------------------------- chain edge cases

class TestChainEdgeCases(Base):
    """Three cases that a naive chain builder gets wrong."""

    def test_ge_chain_crosses_gap(self):
        # the chain must cross the 0.4 K gap between Ge(cr) and Ge(L) in Burcat
        self.assertGreater(get_species("Ge", "solid").H(2000.0), 0.0)

    def test_registry_rule_is_read(self):
        # no_calculate_above_T is read from the override_rule column
        self.assertEqual(get_species("KNO3", "solid")._no_calc_above_K, 607.7)

    def test_anchor_is_field15_without_nbs(self):
        # the polynomial gives -2491.13 and field 15 declares -2508.20 kJ/mol; field 15 wins
        sp = get_species("MgTi2O5", "solid")
        self.assertAlmostEqual(sp.H(298.15), sp.dHf_298.value, delta=1.0)


# --------------------------------------------------------------- Shomate and the S anchor

class TestShomateAndEntropyAnchor(Base):
    """Fourth source (NIST Shomate) and S° anchored on NBS."""

    def test_co3o4_chain_is_nist(self):
        from thermo.shomate_reader import ShomateBlock
        sp = get_species("Co3O4", "solid")
        self.assertEqual(sp._chain_source, "NIST-JANAF")
        self.assertTrue(all(isinstance(b, ShomateBlock) for b in sp._chain))
        # the whole 298 K point from NBS; JANAF would give -910.02 and 114.44
        self.assertEqual((sp.dHf_298.source_ref, sp.S_298.source_ref), ("NBS", "NBS"))
        self.assertAlmostEqual(sp.H(298.15), -891000.0, delta=1.0)
        self.assertAlmostEqual(sp.S(298.15), 102.5, places=6)

    def test_increment_comes_from_the_chain(self):
        sp = get_species("Co3O4", "solid")
        b = sp._chain[0]
        self.assertAlmostEqual(sp.S(1000.0) - sp.S(298.15), b.S(1000.0) - b.S(298.15), places=9)
        self.assertAlmostEqual(sp.H(1000.0) - sp.H(298.15), b.H(1000.0) - b.H(298.15), places=6)

    def test_without_nbs_the_anchor_is_the_tabulated_value(self):
        sp = get_species("CoO", "solid")                 # CoO is not in SELECT
        self.assertEqual(sp.dHf_298.source_ref, "NIST-JANAF")
        self.assertAlmostEqual(sp.H(298.15), -237735.3, delta=0.5)   # the H coefficient, not the polynomial

    @NEEDS_CEA
    def test_entropy_anchor_moves_wo3(self):
        sp = get_species("WO3", "solid")                 # CEA chain, S(298) = 81.64
        self.assertEqual(sp.S_298.source_ref, "NBS")
        self.assertAlmostEqual(sp.S(298.15), 75.90, places=6)
        self.assertIn("81.64", sp.S_298.note)

    def test_no_anchor_without_nbs_dHf(self):
        sp = get_species("MgTi2O5", "solid")             # Burcat anchor: S is not touched either
        self.assertIsNone(sp._anchor_S298)
        self.assertAlmostEqual(sp.S(298.15), sp._pick(298.15).S(298.15, strict=False), places=9)


# --------------------------------------------------------------- H_with_sources

class TestHWithSources(Base):

    def test_h_sources_declared(self):
        sp = get_species("Al2O3", "solid")
        H, srcs = sp.H_with_sources(1000.0)
        self.assertAlmostEqual(H, sp.H(1000.0), delta=1e-6)
        self.assertTrue(srcs, "empty source list")
        for s in srcs:
            self.assertIn(s, ("NBS", "Burcat", "CEA", "NIST-JANAF"), f"unknown source: {s}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
