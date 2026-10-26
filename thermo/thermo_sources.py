"""One layer over four thermochemical sources, with provenance per property.

Sources, in the order they are consulted:
  - dHf(298), any phase: NBS if it has it (gases are asked for as formula(g)); otherwise the
    dHf of the block stable at 298.15 K in the chain (field 15 in Burcat, H(298.15) in CEA, the
    tabulated H coefficient in Shomate).
  - H(T), S(T) and Cp(T): the Burcat chain from the block stable at 298 K; if Burcat has no
    chain, the CEA chain; if neither, the NIST Shomate chain. A chain never mixes blocks of two
    sources. H(T) = anchor + [H_chain(T) - H_chain(298)].
  - S°(298): NBS if dHf came from NBS and that same row has S°; otherwise the S(298.15) of the
    chain. S(T) = S°(298) + [S_chain(T) - S_chain(298)]. So the 298 K point (dHf and S°) comes
    whole from one source and the chain only supplies the shape in T.

Rules the layer enforces:
  - The NBS anchor respects the phase: gases ask the reader for (g).
  - H(298.15) returns the anchor when there is no chain, for species only needed at 298 K.
  - The chain starts from the block stable at 298.15 K and only follows blocks that touch in T,
    which avoids mixing isomers (singlet O2, OH A state, two Bi2O3 records, etc.). CHAIN_TOL is
    0.5 K because Burcat leaves a 0.4 K gap between Ge(cr) and Ge(L).
  - Boundaries that do not join raise FrontierInconsistency; they are not skipped.
  - Burcat, CEA and Shomate values are classified as ESTIMATED (fits to evaluated compilations),
    with the block's REF in the note; NBS values as EXPERIMENTAL.
  - H() returns the number; H_with_sources() returns the sources as well.
  - Without NBS, the anchor is field 15 of the block stable at 298 K, not the polynomial.
  - no_calculate_above_T from the defect register applies to any phase of the species.
  - SELECT in nbs_tables has a single gas, H2O(g), equal in NBS and Burcat. NBS 1982 is
    superseded for several gases: do not add gases to SELECT without checking them first.

File locations (environment variables, or reassign the module constants before first use):
  BURCAT_THR           path of BURCAT.THR (see burcat.py for the download)
  CEA_NASA9_YAML       path of the converted CEA YAML (see cea_reader.py)
  NIST_SHOMATE_CSV     default data/nist_shomate.csv
  BURCAT_DEFECTS_CSV   default data/burcat_known_defects.csv
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from .burcat import (
    BurcatBlock, BurcatDatabase, BurcatJoinError, BurcatRangeError,
    load_defects_csv,
)
from .cea_reader import CEABlock, CEADatabase
from .shomate_reader import ShomateBlock, ShomateDatabase
from . import nbs_tables

ROOT = Path(__file__).resolve().parents[1]
DEFECTS_CSV = Path(os.environ.get("BURCAT_DEFECTS_CSV") or ROOT / "data" / "burcat_known_defects.csv")
BURCAT_PATH = Path(os.environ.get("BURCAT_THR") or ROOT / "data" / "external" / "BURCAT.THR")
CEA_YAML = Path(os.environ.get("CEA_NASA9_YAML") or ROOT / "data" / "external" / "cea_nasa9.yaml")
NIST_SHOMATE_CSV = Path(os.environ.get("NIST_SHOMATE_CSV") or ROOT / "data" / "nist_shomate.csv")

T_REF = 298.15
FRONTIER_TOL_KJ = 1.0       # maximum |dG| in kJ/mol to accept a boundary
FRONTIER_TOL_DH = -0.1      # minimum dH in kJ/mol (transitions cannot be exothermic)
CHAIN_TOL = 0.5             # tolerance in K for T_high == T_low between blocks (Ge leaves 0.4 K)


class SourceType(Enum):
    """ThermoChimie classification."""
    CONVENTION = "A"      # value by convention (CODATA key values)
    EXPERIMENTAL = "B"    # direct experimental measurement
    ESTIMATED = "C"       # estimated on a scientific basis (fits, NKR)
    INTERNAL = "D"        # computed internally from other values
    UNDEFINED = "E"


class SpeciesPropertyMissing(KeyError):
    """The requested property is in no source."""


class FrontierInconsistency(ValueError):
    """Two consecutive blocks do not join within the declared thresholds."""


class SpeciesNotFound(KeyError):
    """The species appears in no source."""


# ------------------------------------------------------------------ Property

@dataclass(frozen=True)
class Property:
    value: Optional[float]
    units: str
    source_ref: str
    source_type: SourceType
    T_ref: Optional[float] = None
    note: str = ""


# ------------------------------------------------------------------ Species

@dataclass
class Species:
    formula: str
    phase: str
    dHf_298: Property
    S_298: Property
    Cp_298: Property
    _chain: list = field(default_factory=list)          # all of one type: Burcat, CEA or Shomate
    _anchor_H298_J: Optional[float] = None
    _no_calc_above_K: Optional[float] = None
    _chain_source: str = ""                             # 'Burcat', 'CEA' or 'NIST-JANAF'
    _anchor_S298: Optional[float] = None                # NBS S° in J/(mol K), if it anchors the entropy

    def _require_chain(self, what: str) -> None:
        if not self._chain:
            raise SpeciesPropertyMissing(
                f"{self.formula} ({self.phase}): {what} needs a chain (Burcat, CEA or NIST) "
                f"and there is none. Anchor source: {self.dHf_298.source_ref}."
            )

    def _pick(self, T: float) -> BurcatBlock:
        for b in self._chain:
            if b.T_low <= T <= b.T_high:
                return b
        raise BurcatRangeError(
            f"{self.formula}: T={T:.2f} outside the chain "
            f"[{self._chain[0].T_low}, {self._chain[-1].T_high}]"
        )

    def _validate_frontiers(self, T_lo: float, T_hi: float) -> None:
        for a, b in zip(self._chain[:-1], self._chain[1:]):
            Tf = b.T_low
            if Tf < T_lo or Tf > T_hi:
                continue
            try:
                Ha = a.H(Tf, strict=False)
                Hb = b.H(Tf, strict=False)
                Sa = a.S(Tf, strict=False)
                Sb = b.S(Tf, strict=False)
            except BurcatJoinError as e:
                raise FrontierInconsistency(
                    f"{self.formula}: cannot evaluate the boundary at "
                    f"T={Tf:.2f} between {a.name} and {b.name}: {e}"
                ) from e
            dH = Hb - Ha
            dG = dH - Tf * (Sb - Sa)
            if abs(dG) > FRONTIER_TOL_KJ * 1000.0 or dH < FRONTIER_TOL_DH * 1000.0:
                raise FrontierInconsistency(
                    f"{self.formula}: boundary at T={Tf:.2f} K between "
                    f"{a.name} and {b.name} does not join (dH={dH/1000:.3f}, "
                    f"dG={dG/1000:.3f} kJ/mol)"
                )

    def H(self, T: float) -> float:
        if self._no_calc_above_K is not None and T > self._no_calc_above_K:
            raise FrontierInconsistency(
                f"{self.formula}: T={T:.2f} > {self._no_calc_above_K:.2f} K "
                f"forbidden by the defect register"
            )
        # Anchor alone for T = 298.15 without a chain (species only needed at 298 K)
        if abs(T - T_REF) < 1e-6:
            if self._anchor_H298_J is not None:
                return self._anchor_H298_J
            if self._chain:
                return self._pick(T_REF).H(T_REF, strict=False)
            raise SpeciesPropertyMissing(
                f"{self.formula}: H(298.15) without NBS anchor or chain."
            )
        self._require_chain("H(T)")
        self._validate_frontiers(min(T_REF, T), max(T_REF, T))
        if self._anchor_H298_J is not None:
            anchor_H, anchor_T = self._anchor_H298_J, T_REF
        else:
            b298 = self._pick(T_REF)
            anchor_H = b298.H(T_REF, strict=False)
            anchor_T = T_REF
        bT = self._pick(T)
        bA = self._pick(anchor_T)
        return anchor_H + (bT.H(T, strict=False) - bA.H(anchor_T, strict=False))

    def S(self, T: float) -> float:
        self._require_chain("S(T)")
        self._validate_frontiers(min(T_REF, T), max(T_REF, T))
        S_T = self._pick(T).S(T, strict=False)
        if self._anchor_S298 is None:
            return S_T
        return self._anchor_S298 + (S_T - self._pick(T_REF).S(T_REF, strict=False))

    def Cp(self, T: float) -> float:
        self._require_chain("Cp(T)")
        return self._pick(T).cp(T, strict=False)

    def G(self, T: float) -> float:
        return self.H(T) - T * self.S(T)

    def H_with_sources(self, T: float) -> tuple[float, list[str]]:
        """Like H(T), but also returns the sources: (J/mol, ['NBS', 'Burcat']).
        The first is the anchor's (the one declared by dHf_298, not assumed to be NBS); the chain's
        (Burcat, CEA or NIST-JANAF) is added when it supplies the increment H(T) - H(298)."""
        val = self.H(T)
        srcs = [self.dHf_298.source_ref]
        if self._chain and abs(T - T_REF) >= 1e-6:
            srcs.append(self._chain_source)
        return val, list(dict.fromkeys(srcs))


# ------------------------------------------------------------------ chain build

_BURCAT: Optional[BurcatDatabase] = None
_CEA: Optional[CEADatabase] = None
_NIST: Optional[ShomateDatabase] = None
_DEFECTS_ROWS: Optional[list[dict]] = None


def _db() -> BurcatDatabase:
    global _BURCAT
    if _BURCAT is None:
        _BURCAT = BurcatDatabase(BURCAT_PATH, defects=DEFECTS_CSV if Path(DEFECTS_CSV).exists() else None)
    return _BURCAT


def _cea() -> Optional[CEADatabase]:
    """The CEA database, or None when the converted YAML is not installed."""
    global _CEA
    if _CEA is None and Path(CEA_YAML).exists():
        _CEA = CEADatabase(CEA_YAML)
    return _CEA


def _nist() -> Optional[ShomateDatabase]:
    """The Shomate database, or None when the CSV is not present."""
    global _NIST
    if _NIST is None and Path(NIST_SHOMATE_CSV).exists():
        _NIST = ShomateDatabase(NIST_SHOMATE_CSV)
    return _NIST


def _defects() -> list[dict]:
    global _DEFECTS_ROWS
    if _DEFECTS_ROWS is None:
        try:
            _DEFECTS_ROWS = load_defects_csv(DEFECTS_CSV)
        except FileNotFoundError:
            _DEFECTS_ROWS = []
    return _DEFECTS_ROWS


def _applicable_defects(formula: str, phase: str) -> list[dict]:
    return [r for r in _defects()
            if r.get("species") == formula and r.get("phase") == phase]


def _no_calc_above(formula: str) -> Optional[float]:
    """no_calculate_above_T from the register. Applies to any phase of the species: the chain
    is the same for 'solid' and 'liquid'."""
    for r in _defects():
        if r.get("species") != formula:
            continue
        if r.get("override_rule") != "no_calculate_above_T":
            continue
        try:
            return float(r["T_K"])
        except (KeyError, ValueError, TypeError):
            continue
    return None


def _build_chain(db: BurcatDatabase, formula: str, phase: str) -> list[BurcatBlock]:
    """Chain from the block stable at 298 K, extended only to contiguous blocks."""
    bphase = "gas" if phase == "gas" else "condensed"
    blocks = db.chain(formula, bphase)
    if not blocks:
        return []
    # Anchor candidates: blocks that contain 298.15 K and compute without error
    candidates = []
    for b in blocks:
        if not b.contains(T_REF):
            continue
        try:
            candidates.append((b.G(T_REF), b))
        except BurcatJoinError:
            continue
    if not candidates:
        return []
    candidates.sort(key=lambda x: x[0])
    chain = [candidates[0][1]]
    # Extend forward
    cur = chain[-1]
    while True:
        nxt = next((b for b in blocks
                    if b not in chain and abs(b.T_low - cur.T_high) < CHAIN_TOL), None)
        if nxt is None:
            break
        chain.append(nxt)
        cur = nxt
    # Extend backward
    cur = chain[0]
    while True:
        prv = next((b for b in blocks
                    if b not in chain and abs(b.T_high - cur.T_low) < CHAIN_TOL), None)
        if prv is None:
            break
        chain.insert(0, prv)
        cur = prv
    return chain


# ------------------------------------------------------------------ NBS anchor

def _nbs_dHf(formula: str, phase: str) -> Optional[Property]:
    """The curated NBS value (nbs_tables.SELECT) for the formula in the requested phase.
    Confidence describes the selection: 'high' for a single unambiguous entry (cr, l, g),
    'polymorph-choice' when one of several forms was picked, 'amorphous-only' when the tables
    carry only an amorphous form."""
    # gases ask for an explicit (g); condensed phases ask for the bare formula
    lookup = formula + "(g)" if phase == "gas" else formula
    r = nbs_tables.lookup(lookup)
    if r is None or r["dHf_kJ"] is None:
        return None
    conf = "high" if r["state"] in ("cr", "l", "g") else "polymorph-choice"
    if r["state"] == "am":
        conf = "amorphous-only"
    note = f"confidence={conf}" if conf else ""
    return Property(
        value=r["dHf_kJ"] * 1000.0, units="J/mol", source_ref="NBS",
        source_type=SourceType.EXPERIMENTAL, T_ref=T_REF, note=note,
    )


# ------------------------------------------------------------------ public API

def get_species(formula: str, phase: str) -> Species:
    if phase not in ("gas", "solid", "liquid"):
        raise ValueError(f"phase must be 'gas', 'solid' or 'liquid', not {phase!r}")

    db = _db()
    chain = _build_chain(db, formula, phase)
    cphase = "gas" if phase == "gas" else "condensed"
    if not chain and _cea() is not None:            # tertiary source: the whole chain from CEA
        chain = _cea().chain(formula, cphase)
    if not chain and _nist() is not None:           # fourth: the whole chain from Shomate (NIST)
        chain = _nist().chain(formula, cphase)
    src = "" if not chain else ("CEA" if isinstance(chain[0], CEABlock)
                                else "NIST-JANAF" if isinstance(chain[0], ShomateBlock) else "Burcat")
    dHf = _nbs_dHf(formula, phase)

    if not chain and dHf is None:
        raise SpeciesNotFound(
            f"{formula} ({phase}): no Burcat or CEA chain and no dHf° in NBS."
        )

    def where(b) -> str:
        if src == "CEA":
            return f"{b.name}, cea_nasa9.yaml line {b.source_line}, REF={b.ref}"
        if src == "NIST-JANAF":
            return f"{b.name}, nist_shomate.csv row {b.source_line}, {b.ref}"
        return f"line {b.source_line}, REF={b.ref}"

    # Anchor: NBS if it exists; otherwise the dHf of the block containing 298 K (fallback below)
    if dHf is None and chain:
        b298 = next((b for b in chain if b.contains(T_REF)), None)
        if b298 is not None:
            dHf = Property(
                value=b298.dHf_298, units="J/mol", source_ref=src,
                source_type=SourceType.ESTIMATED, T_ref=T_REF, note=where(b298),
            )

    # The anchor is fixed AFTER the fallback: without NBS it is field 15, not the polynomial
    anchor_H298 = dHf.value if dHf is not None else None

    # S(298) and Cp(298) from the block stable at 298 K
    S298 = Cp298 = None
    b298 = next((b for b in chain if b.contains(T_REF)), None)
    if b298 is not None:
        try:
            Cp298 = Property(
                value=b298.cp(T_REF, strict=False), units="J/(mol·K)",
                source_ref=src, source_type=SourceType.ESTIMATED,
                T_ref=T_REF, note=where(b298),
            )
        except (BurcatJoinError, BurcatRangeError):
            pass
        try:
            S298 = Property(
                value=b298.S(T_REF, strict=False), units="J/(mol·K)",
                source_ref=src, source_type=SourceType.ESTIMATED,
                T_ref=T_REF, note=where(b298),
            )
        except (BurcatJoinError, BurcatRangeError):
            pass

    # With a chain and an NBS anchor, S° also comes from NBS if the SAME row has it (its dHf is
    # checked against the anchor). The chain keeps the shape in T: S(T) = S° + dS_chain.
    anchor_S298 = None
    if chain and dHf is not None and dHf.source_ref == "NBS":
        r = nbs_tables.lookup(formula + "(g)" if phase == "gas" else formula)
        if (r is not None and r["S_J_per_molK"] is not None and r["dHf_kJ"] is not None
                and abs(r["dHf_kJ"] * 1000.0 - dHf.value) < 1.0):
            anchor_S298 = r["S_J_per_molK"]
            chain_S = S298.value if S298 is not None else None
            gap = "" if chain_S is None else f"; the chain ({src}) gave {chain_S:.3f}"
            S298 = Property(value=anchor_S298, units="J/(mol·K)", source_ref="NBS",
                            source_type=SourceType.EXPERIMENTAL, T_ref=T_REF,
                            note=f"NBS {r['nbs_formula']} {r['state']}, same row as dHf{gap}")

    # Without a chain, S° and Cp° at 298.15 K come from NBS (the same row as the anchor, via SELECT)
    if not chain:
        r = nbs_tables.lookup(formula + "(g)" if phase == "gas" else formula)
        if r is not None:
            note = f"NBS {r['nbs_formula']} {r['state']} at 298.15 K; no chain"
            if r["S_J_per_molK"] is not None:
                S298 = Property(value=r["S_J_per_molK"], units="J/(mol·K)", source_ref="NBS",
                                source_type=SourceType.EXPERIMENTAL, T_ref=T_REF, note=note)
            if r["Cp_J_per_molK"] is not None:
                Cp298 = Property(value=r["Cp_J_per_molK"], units="J/(mol·K)", source_ref="NBS",
                                 source_type=SourceType.EXPERIMENTAL, T_ref=T_REF, note=note)

    missing = Property(value=None, units="", source_ref="missing",
                       source_type=SourceType.UNDEFINED)

    # no_calculate_above_T from the register (override_rule column, any phase)
    no_above = _no_calc_above(formula)

    return Species(
        formula=formula, phase=phase,
        dHf_298=dHf if dHf is not None else missing,
        S_298=S298 if S298 is not None else missing,
        Cp_298=Cp298 if Cp298 is not None else missing,
        _chain=chain,
        _anchor_H298_J=anchor_H298,
        _no_calc_above_K=no_above,
        _chain_source=src,
        _anchor_S298=anchor_S298,
    )


def standard_298(formula: str, phase: str) -> dict:
    """dHf, S and Cp at 298.15 K as Property, for first-order calculations (dCp = 0).

    First get_species (anchor + chain, or NBS when there is no chain). If the species is not
    there and the phase is condensed, a row of the NBS table outside SELECT is used (state cr,
    then l, then am), marked as such in the note. Gases never come from an uncurated row: NBS
    1982 is superseded for several gases."""
    try:
        sp = get_species(formula, phase)
        return dict(dHf=sp.dHf_298, S=sp.S_298, Cp=sp.Cp_298, origin="get_species")
    except SpeciesNotFound:
        if phase == "gas":
            raise
    for st in ("cr", "l", "am"):
        r = nbs_tables.lookup(formula, st)
        if r is None or r["dHf_kJ"] is None:
            continue
        note = f"NBS {r['nbs_formula']} {st}: uncurated row (not in SELECT)"

        def prop(v, units):
            if v is None:
                return Property(value=None, units="", source_ref="missing", source_type=SourceType.UNDEFINED)
            return Property(value=v, units=units, source_ref="NBS", source_type=SourceType.EXPERIMENTAL,
                            T_ref=T_REF, note=note)
        return dict(dHf=prop(r["dHf_kJ"] * 1000.0, "J/mol"), S=prop(r["S_J_per_molK"], "J/(mol·K)"),
                    Cp=prop(r["Cp_J_per_molK"], "J/(mol·K)"), origin="nbs_raw")
    raise SpeciesNotFound(f"{formula} ({phase}): in no source, not even as an uncurated NBS row")
