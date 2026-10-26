"""A hazard-label gate for computational screening of inorganic compounds: an independent check of every
composition's elements and structure that attaches a hazard label with its cause and source.

Design: hazardous compositions are NOT removed, because removing them would also remove the scientific
information they carry; they are shown to the reader with a WARNING, the reason, and the sources, so that
the reader (and any reviewer) can judge. A model suggestion is never an instruction: everything is
evaluated. So this module labels; it never deletes. Any scientific category is decided elsewhere, on
scientific grounds only. The lab-use statements make two things explicit: a WARNING composition is
excluded from any experimental follow-up unless a separate, formal safety review establishes an
appropriate justification and approved procedures; and computational identification never implies that
synthesis or experimental testing is recommended (NOT_FOR_LAB and COMPUTATIONAL_ONLY carry those two
sentences; current_lab_use() maps an older wording of the first one to the current text).

No computational filter can certify that anything is safe: it can only report what authoritative sources
already classify as hazardous and flag what nobody has characterized.

Rules (criteria declared before the evidence was collected; element table data/safety_rules.csv, built
only from items that an independent verifier confirmed):
  S1  WARNING for an element (or an element in a given oxidation state) whose inorganic compounds, oxides or
      hydroxides carry an IARC Group 1, an EU CLP harmonised Acute Tox. 1-2 or a Carc./Muta./Repr. 1A-1B
      classification, and for radioactive elements; CAUTION for IARC 2A/2B, CLP category 2, STOT RE, Acute Tox. 3.
  S2  WARNING for elements that form volatile, highly toxic species at modest temperature; CAUTION for volatile
      species of lower toxicity.
  S3  WARNING for a composition outside the oxide/hydroxide scope, and for an O-O bond in the relaxed structure
      (peroxide-like below 1.639 A, superoxide-like below 1.485 A: pymatgen's OxideType cut-offs).
  S4  WARNING when a predicted oxygen-release temperature in air is at or below T_WARN_K = room temperature
      (298 K) + the 100 K margin of thermal process safety + the largest over-prediction of the upstream
      estimator (WORST_OVERPREDICTION_K); CAUTION at or below T_CAUTION_K (one mean estimator error above).
      The 100 K margin comes from German thermal-process-safety guidance (TRAS 410, current text BAnz AT
      23.02.2021 B5; DGUV Information 213-067), where it applies to a measured decomposition onset; here it is
      borrowed by analogy, and those rules are documented as not always conservative. The two estimator
      errors are parameters: set them to the calibration of whatever estimator feeds t_release_K.
  S5  Every composition that a literature review did not find as a known compound: "not characterized:
      evaluation on paper only".
  S6  WARNING when no verified literature and hazard review exists yet (LIT_FILE, see literature_reviewed).
  S7  CAUTION for any element for which no safety evidence has been collected yet (EVIDENCE_ELEMENTS).
  S8  the table carries re-check dates; the module warns when one has passed.

Run the self-test:  python -m safety_gate.safety_gate
"""
from __future__ import annotations

import csv
import datetime
import json
import os
import warnings
from typing import Optional

import numpy as np
from pymatgen.core import Composition, Element, Structure

HERE = os.path.dirname(os.path.abspath(__file__))
RULES_FILE = os.path.join(HERE, "data", "safety_rules.csv")
# S6: a JSON file {"rows": [{"formula": "..."}, ...]} listing every composition with a verified literature and
# hazard review (a " (from ...)" suffix on the formula is ignored). Point this at your own review file.
LIT_FILE = os.path.join(HERE, "data", "literature_review.json")

SCOPE_METALLOIDS = {"B", "Si", "Ge", "As", "Sb", "Te"}
OO_SUPEROXIDE_A = 1.1 * 1.35      # pymatgen OxideType cut-offs (relative_cutoff 1.1)
OO_PEROXIDE_A = 1.1 * 1.49
ROOM_K = 298.0
SCAN_FLOOR_K = 250.0              # lowest temperature of the oxygen-release scan feeding t_release_K
WORST_OVERPREDICTION_K = 345.0    # largest over-prediction of the release-temperature estimator on known oxides
MEAN_ERROR_K = 160.0              # its mean error; both are calibration parameters of the upstream estimator
PROCESS_MARGIN_K = 100.0          # the '100 K rule' of thermal process safety (TRAS 410; DGUV Information 213-067)
T_WARN_K = ROOM_K + PROCESS_MARGIN_K + WORST_OVERPREDICTION_K     # 743 K
T_CAUTION_K = T_WARN_K + MEAN_ERROR_K                             # 903 K

# S7: the elements for which safety evidence was collected and verified (IARC and CLP checked per element)
EVIDENCE_ELEMENTS = set("""Ag Al As Au B Ba Be Bi Ca Cd Ce Co Cr Cs Cu Dy Er Eu Fe Ga Gd Ge Hg Ho In K La Li Lu Mg Mn Mo
Na Nb Nd Ni Os Pb Pd Pr Pt Rb Re Rh Ru Sb Sc Se Si Sm Sn Sr Ta Tb Te Th Ti Tl U V W Y Yb Zn Zr""".split())
EVIDENCE_DATE = "2026-09-26"      # when the element evidence was last reviewed; S8 schedules the next review
# S8: re-check dates for the element table
RECHECK = [
    ("2027-03-26", "the draft 24th adaptation of the EU CLP Regulation (Repr. 1A proposed for lithium salts) may have "
                   "entered into force: check the Official Journal"),
    ("2027-09-26", "one year since the evidence was collected: repeat the IARC and CLP check for every element"),
    ("2027-10-27", "IARC Monographs Meeting 144 (October 2027) re-evaluates metallic lead, inorganic lead compounds, "
                   "cadmium and cadmium compounds"),
]
NOT_CHARACTERIZED = "not characterized: evaluation on paper only"
# The statement for every WARNING composition; no '[S#]' tag in it, because report cards parse those.
NOT_FOR_LAB = ("excluded from any experimental follow-up unless a separate, formal safety review establishes an "
               "appropriate justification and approved procedures: evaluation on paper only")
NOT_FOR_LAB_LEGACY = "not recommended for laboratory use: evaluation on paper only"
KNOWN_LAB_USE = "known compound: any laboratory use follows the laboratory's own procedures and safety data sheet"
# For a reported compound whose identity the review has not confirmed: the known-compound sentence would read
# as a clearance.
REPORTED_UNVERIFIED_LAB_USE = ("reported in the literature, but its identity or chemistry is not yet confirmed: "
                               "evaluation on paper only")
# On every card and every review, whatever the hazard level.
COMPUTATIONAL_ONLY = ("Computational identification does not imply that synthesis or experimental testing is "
                      "recommended.")


def current_lab_use(text: str) -> str:
    """The stored lab-use statement in the current wording: tables written with the older WARNING sentence
    get the current one; everything else is returned unchanged."""
    return NOT_FOR_LAB if str(text).strip() == NOT_FOR_LAB_LEGACY else text

_RULES = None


def recheck_due(today: Optional[datetime.date] = None):
    """S8: the re-check notices whose date has passed."""
    today = today or datetime.date.today()
    return [(d, why) for d, why in RECHECK if today >= datetime.date.fromisoformat(d)]


def rules():
    """Element-level rules (S1, S2) from data/safety_rules.csv: element, oxidation_state ('' = any), applies_to,
    rule, level (EXCLUDE = warning level, CAUTION), reason, source_title, source_url, verified."""
    global _RULES
    if _RULES is None:
        _RULES = []
        if os.path.exists(RULES_FILE):
            with open(RULES_FILE, encoding="utf-8-sig") as f:
                _RULES = [r for r in csv.DictReader(f) if r.get("element")]
        for d, why in recheck_due():
            warnings.warn(f"safety table re-check due since {d}: {why}")
    return _RULES


def _level(table_level: str) -> str:
    """The table's EXCLUDE level is a WARNING label here: nothing is removed."""
    return "WARNING" if table_level == "EXCLUDE" else table_level


def scope_check(formula: str):
    """S3a: only metals, the scope metalloids, O and H."""
    bad = [el.symbol for el in Composition(formula).elements
           if not (el.is_metal or el.symbol in SCOPE_METALLOIDS or el.symbol in ("O", "H"))]
    if bad:
        return "WARNING", f"[S3] outside the oxide/hydroxide scope (contains {', '.join(bad)})"
    return "NONE", ""


def min_oo_distance(s: Structure) -> float:
    o = [i for i, site in enumerate(s) if site.specie.symbol == "O"]
    if len(o) < 2:
        return float("inf")
    d = s.distance_matrix[np.ix_(o, o)]
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def oo_bond_check(s: Optional[Structure]):
    """S3b: an O-O bond means a peroxide- or superoxide-like unit."""
    if s is None:
        return "NONE", "", float("nan")
    d = min_oo_distance(s)
    if d < OO_SUPEROXIDE_A:
        return "WARNING", f"[S3] O-O distance {d:.2f} A: a superoxide-like unit (below {OO_SUPEROXIDE_A:.3f} A)", d
    if d < OO_PEROXIDE_A:
        return "WARNING", f"[S3] O-O distance {d:.2f} A: a peroxide-like unit (below {OO_PEROXIDE_A:.3f} A)", d
    return "NONE", "", d


def _applies(cond: str, formula: str) -> bool:
    """applies_to conditions joined by '&': any | hydroxide | anhydrous | formula:X | with:A,B (any of them present)."""
    c = Composition(formula).reduced_composition
    els = {el.symbol for el in c.elements}
    for part in (cond or "any").split("&"):
        part = part.strip()
        if part in ("", "any"):
            continue
        if part == "hydroxide" and "H" not in els:
            return False
        if part == "anhydrous" and "H" in els:
            return False
        if part.startswith("formula:") and c.reduced_formula != Composition(part[8:]).reduced_formula:
            return False
        if part.startswith("with:") and not (els & set(part[5:].split(","))):
            return False
    return True


def element_rules(formula: str, states: Optional[dict] = None):
    """S1, S2 and S7. `states` maps element -> list of the oxidation states the formula forces on it (both states of
    a mixed valence). A state-specific rule applies only when that state is among them; with no state information it
    applies (conservative)."""
    out = []
    els = {el.symbol for el in Composition(formula).elements}
    for el in sorted(els):
        if Element(el).is_radioactive:
            out.append(("WARNING", f"[S1] {el}: radioactive (no stable isotope)"))
        if el not in ("O", "H") and el not in EVIDENCE_ELEMENTS:
            out.append(("CAUTION", f"[S7] {el}: no safety evidence collected yet (the evidence review of {EVIDENCE_DATE} did not cover it)"))
    for r in rules():
        if r["element"] not in els or not _applies(r.get("applies_to", "any"), formula):
            continue
        st = (r.get("oxidation_state") or "").strip()
        if st:
            qs = (states or {}).get(r["element"])
            if qs is not None:
                qs = qs if isinstance(qs, (list, tuple, set)) else [qs]
                if not any(abs(float(q) - float(st)) < 1e-6 for q in qs):
                    continue
        out.append((_level(r["level"]), f"[{r['rule']}] {r['element']}{'(' + st + ')' if st else ''}: {r['reason']} "
                                        f"(source: {r['source_title']})"))
    return out


def thermal_check(t_release_K: float):
    """S4."""
    if t_release_K != t_release_K:
        return "NONE", ""
    worst = t_release_K - WORST_OVERPREDICTION_K
    if t_release_K <= T_WARN_K:
        if t_release_K <= SCAN_FLOOR_K:   # the scan's floor: release already at its lowest temperature, not an onset
            return "WARNING", (f"[S4] the check finds oxygen release in air already at {SCAN_FLOOR_K:.0f} K, the lowest "
                               "temperature it scans, so at or below room temperature; this marks a strongly oxidizing "
                               "compound, not an onset temperature")
        if worst <= ROOM_K:      # the calibration error is larger than the distance to room temperature: no Kelvin value to quote
            return "WARNING", (f"[S4] predicted oxygen release in air about {t_release_K:.0f} K; within the worst calibration error "
                               f"({WORST_OVERPREDICTION_K:.0f} K) the release could start at or below room temperature")
        return "WARNING", (f"[S4] predicted oxygen release in air about {t_release_K:.0f} K; with the worst calibration error "
                           f"it could be {worst:.0f} K, within the 100 K safety margin of room temperature")
    if t_release_K <= T_CAUTION_K:
        return "CAUTION", (f"[S4] predicted oxygen release about {t_release_K:.0f} K is within one mean calibration error "
                           f"({MEAN_ERROR_K:.0f} K) of the warning threshold ({T_WARN_K:.0f} K)")
    return "NONE", ""


def literature_reviewed(formula: str) -> bool:
    """S6: a verified literature and hazard review exists for this composition (listed in LIT_FILE)."""
    if not os.path.exists(LIT_FILE):
        return False
    with open(LIT_FILE, encoding="utf-8") as fh:
        rows = json.load(fh)["rows"]
    keys = {r["formula"].split(" (from")[0].strip() for r in rows}
    return Composition(formula).reduced_formula in {Composition(k).reduced_formula for k in keys}


# Harmonised classifications borne by the ANION, which an element rule cannot see: a note for the report of an
# oxyanion salt, never a level.
ANION_CLP_NOTES = {
    "Br": "bromate: EU CLP Annex VI potassium bromate Carc. 1B (035-003-00-6)",
    "Cl": "chlorate: EU CLP Annex VI potassium and sodium chlorate Acute Tox. 3 (017-004-00-3, 017-005-00-9); perchlorates Ox. Sol. 1 only",
    "N": "nitrite: EU CLP Annex VI sodium and potassium nitrite Acute Tox. 3 (007-010-00-4, 007-011-00-X); nitrates Ox. Sol. only",
}


def anion_clp_note(formula: str) -> str:
    """The harmonised classification of the oxyanion of a salt (bromate, chlorate, nitrite), as a report note.
    Returns '' for a composition without such an anion. Does not set a level: S3 already labels these compositions
    as outside the oxide/hydroxide scope."""
    els = {el.symbol for el in Composition(formula).elements}
    if "O" not in els:
        return ""
    return " || ".join(ANION_CLP_NOTES[e] for e in ("Br", "Cl", "N") if e in els)


def gate(formula: str, structure: Optional[Structure] = None, states: Optional[dict] = None,
         t_release_K: float = float("nan"), known_compound: bool = False) -> dict:
    """All rules for one composition. hazard_level: WARNING > CAUTION > NONE. Nothing is removed: the result is a
    label, the reasons with their sources, and a statement on laboratory use."""
    warn, caution = [], []
    for lv, why in [scope_check(formula), oo_bond_check(structure)[:2], thermal_check(t_release_K)] + element_rules(formula, states):
        if why:
            (warn if lv == "WARNING" else caution).append(why)
    if not literature_reviewed(formula):
        warn.append("[S6] no verified literature and hazard review yet")
    level = "WARNING" if warn else ("CAUTION" if caution else "NONE")
    if level == "WARNING":
        lab = NOT_FOR_LAB
    elif not known_compound:
        lab = "[S5] " + NOT_CHARACTERIZED
    else:
        lab = KNOWN_LAB_USE
    return dict(hazard_level=level, hazard_warning=" || ".join(warn), hazard_caution=" || ".join(caution), lab_use=lab,
                min_OO_A=oo_bond_check(structure)[2] if structure is not None else float("nan"))


if __name__ == "__main__":
    from pymatgen.core import Lattice
    print(f"S4 thresholds: WARNING at or below {T_WARN_K:.0f} K, CAUTION at or below {T_CAUTION_K:.0f} K")
    print(f"element rules loaded: {len(rules())}; evidence covers {len(EVIDENCE_ELEMENTS)} elements (reviewed {EVIDENCE_DATE})")
    print("S8 re-checks due today:", recheck_due() or "none")
    later = datetime.date(2027, 11, 1)
    print(f"S8 re-checks due on {later}:", [d for d, _ in recheck_due(later)])
    for f, st in [("Mn(OH)3", {"Mn": [3]}), ("Tl(OH)3", {"Tl": [3]}), ("Cu(OH)2", {"Cu": [2]}), ("HfO2", {"Hf": [4]}),
                  ("IrO2", {"Ir": [4]}), ("KClO4", {})]:
        g = gate(f, None, st, float("nan"), known_compound=f in ("Tl(OH)3", "Cu(OH)2", "HfO2", "IrO2"))
        print(f"  {f:8s} {g['hazard_level']:8s} lab: {g['lab_use'][:60]:60s} | {(g['hazard_warning'] or g['hazard_caution'])[:90]}")
    per = Structure(Lattice.cubic(5.0), ["Ba", "O", "O"], [[0, 0, 0], [0.5, 0.5, 0.35], [0.5, 0.5, 0.35 + 1.49 / 5.0]])
    print("  O-O check, peroxide unit:", oo_bond_check(per)[:2])
    for t in (250, 655, 750, 1245):
        print(f"  thermal {t:5d} K -> {thermal_check(t)[0]}")


# ---------------------------------------------------------------- S10 to S12: powder-handling notes
# Statements built from regulatory texts on oxidizing solids and combustible dusts; evidence items were confirmed or
# corrected by an independent verifier before the wording was fixed.
S10 = ("[S10] potentially an oxidizing solid (GHS 2.14): the official screening rule can only clear an inorganic substance "
       "that contains no oxygen or halogen, so it cannot clear any oxide or hydroxide; the category needs test O.1/O.3 or an "
       "existing classification (OSHA 29 CFR 1910.1200 App. B.14.3; ECHA CLP guidance 2.14)")
S11 = ("[S11] incompatible with combustible and organic materials, reducing agents and powdered metals; keep away from "
       "combustible materials (OSHA 29 CFR 1910.1200 App. C, oxidizing solids; NOAA CAMEO 'Oxidizing Agents, Strong'; Prudent "
       "Practices in the Laboratory, National Research Council 2011)")
S12_FUEL = ("a reactive metal powder (aluminium, magnesium) handled next to it is a combustible metal dust; dust ignitability (Kst, "
            "minimum ignition energy, minimum explosible concentration) is measured per sample (ASTM E1226, E2019, E1515) and "
            "cannot be computed; fine powders are an inhalation hazard and respirable-dust limits apply (OSHA Table Z-1; NIOSH CIB 63)")


def dust_note(formula: str, states: Optional[dict] = None) -> str:
    """S12: is the solid completely oxidized (its metal in the highest oxidation state it commonly takes)? GHS Annex 11
    (UN ST/SG/AC.10/C.4/2017/3) treats a completely oxidized solid as not able to burn further; a solid that can
    still be oxidized cannot be assumed non-combustible (OSHA Technical Manual IV-6)."""
    from safety_gate.oxidation_states import accessible_states
    parts = []
    for el in sorted({e.symbol for e in Composition(formula).elements} - {"O", "H"}):
        top = max(accessible_states(el)[0] or [0])
        qs = (states or {}).get(el)
        if not qs or not top:
            parts.append(f"{el}: oxidation state not resolved")
            continue
        q = max(float(x) for x in (qs if isinstance(qs, (list, tuple, set)) else [qs]))
        low = min(float(x) for x in (qs if isinstance(qs, (list, tuple, set)) else [qs]))
        if low >= top - 1e-6:
            parts.append(f"{el} is at +{q:g}, at or above its highest common oxidation state (+{top}): completely oxidized, not expected to burn further")
        else:
            parts.append(f"{el} is below its highest common oxidation state (+{low:g} vs +{top}): it could oxidize further, so its "
                         f"dust cannot be assumed non-combustible")
    return "[S12] " + "; ".join(parts) + " (GHS Annex 11; OSHA Technical Manual IV-6). Also: " + S12_FUEL
