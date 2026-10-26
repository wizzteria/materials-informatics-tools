"""Validation of the safety gate's element rules (S1, S2) against the EU harmonised classifications, Annex VI to CLP.

The reference is the ECHA Excel export of Annex VI ("Table 3" of the harmonised classification and labelling list),
downloaded from the ECHA page "Annex VI to CLP" (https://echa.europa.eu/information-on-chemicals/annex-vi-to-clp);
it is not shipped with this package, pass its path with --xlsx. The workbook is read with the standard library
(no new package). A reference substance enters only if its name parses to one inorganic composition within the
gate's scope (oxides, hydroxides, single-oxyanion salts; group entries "X compounds" at element level).

Criterion, declared before any hazard column was read: the gate is called sensitive when at least SENS_WARN of
the WARNING-level substances are labelled WARNING and at least SENS_CAUTION of the CAUTION-or-WARNING substances
get at least CAUTION. Over-flagging is reported, not judged: labelling by element is a cost accepted by design.

Run:  python -m safety_gate.clp_validation --xlsx annex_vi_clp_table_atp23_en.xlsx --out out/clp_validation
      python -m safety_gate.clp_validation --xlsx ... --names-only     (parser coverage, hazard columns unread)
      --elements Fe,Mn,Ni    (optional: extra elements to report in the per-element table)
Outputs: <out>.md, <out>.csv, <out>_elements.csv, <out>_unparsed.csv, <out>_summary.json.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd
from pymatgen.core import Composition, Element

from safety_gate import safety_gate as sg

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
SENS_WARN, SENS_CAUTION = 0.95, 0.90

# ---- vocabulary ------------------------------------------------------------------------------------------------------
ELEMENT_NAMES = {
    "lithium": "Li", "sodium": "Na", "potassium": "K", "rubidium": "Rb", "caesium": "Cs", "cesium": "Cs", "beryllium": "Be",
    "magnesium": "Mg", "calcium": "Ca", "strontium": "Sr", "barium": "Ba", "aluminium": "Al", "aluminum": "Al", "gallium": "Ga",
    "indium": "In", "thallium": "Tl", "tin": "Sn", "lead": "Pb", "bismuth": "Bi", "zinc": "Zn", "cadmium": "Cd", "mercury": "Hg",
    "copper": "Cu", "silver": "Ag", "gold": "Au", "nickel": "Ni", "cobalt": "Co", "iron": "Fe", "manganese": "Mn", "chromium": "Cr",
    "vanadium": "V", "titanium": "Ti", "zirconium": "Zr", "hafnium": "Hf", "niobium": "Nb", "tantalum": "Ta", "molybdenum": "Mo",
    "tungsten": "W", "rhenium": "Re", "ruthenium": "Ru", "osmium": "Os", "rhodium": "Rh", "iridium": "Ir", "palladium": "Pd",
    "platinum": "Pt", "scandium": "Sc", "yttrium": "Y", "lanthanum": "La", "cerium": "Ce", "thorium": "Th", "uranium": "U",
    "boron": "B", "silicon": "Si", "germanium": "Ge", "arsenic": "As", "antimony": "Sb", "tellurium": "Te", "selenium": "Se",
    "ammonium": "NH4", "neodymium": "Nd", "praseodymium": "Pr", "samarium": "Sm", "europium": "Eu", "gadolinium": "Gd",
    "terbium": "Tb", "dysprosium": "Dy", "holmium": "Ho", "erbium": "Er", "thulium": "Tm", "ytterbium": "Yb", "lutetium": "Lu",
}
ELEMENT_ADJ = {"ferric": ("Fe", 3), "ferrous": ("Fe", 2), "cupric": ("Cu", 2), "cuprous": ("Cu", 1), "mercuric": ("Hg", 2),
               "mercurous": ("Hg", 1), "stannic": ("Sn", 4), "stannous": ("Sn", 2), "plumbic": ("Pb", 4), "plumbous": ("Pb", 2),
               "manganous": ("Mn", 2), "cobaltous": ("Co", 2), "chromic": ("Cr", 3), "nickelous": ("Ni", 2), "arsenious": ("As", 3)}
GREEK = {"mono": 1, "di": 2, "tri": 3, "tetra": 4, "penta": 5, "hexa": 6, "hepta": 7, "octa": 8, "nona": 9, "deca": 10, "sesqui": 1.5}
ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7}
ANIONS = {                           # name: (composition, charge)
    "oxide": ("O", -2), "hydroxide": ("OH", -1), "peroxide": ("O2", -2), "superoxide": ("O2", -1),
    "nitrate": ("NO3", -1), "nitrite": ("NO2", -1), "chlorate": ("ClO3", -1), "perchlorate": ("ClO4", -1),
    "bromate": ("BrO3", -1), "iodate": ("IO3", -1), "periodate": ("IO4", -1), "sulfate": ("SO4", -2), "sulphate": ("SO4", -2),
    "sulfite": ("SO3", -2), "sulphite": ("SO3", -2), "carbonate": ("CO3", -2), "selenate": ("SeO4", -2), "selenite": ("SeO3", -2),
    "phosphate": ("PO4", -3), "orthophosphate": ("PO4", -3), "metaphosphate": ("PO3", -1), "silicate": ("SiO3", -2),
    "metasilicate": ("SiO3", -2), "orthosilicate": ("SiO4", -4), "borate": ("BO3", -3), "metaborate": ("BO2", -1),
    "tetraborate": ("B4O7", -2), "chromate": ("CrO4", -2), "dichromate": ("Cr2O7", -2), "molybdate": ("MoO4", -2),
    "tungstate": ("WO4", -2), "vanadate": ("VO3", -1), "metavanadate": ("VO3", -1), "orthovanadate": ("VO4", -3),
    "manganate": ("MnO4", -2), "permanganate": ("MnO4", -1), "arsenate": ("AsO4", -3), "arsenite": ("AsO2", -1),
    "antimonate": ("SbO3", -1), "stannate": ("SnO3", -2), "titanate": ("TiO3", -2), "zirconate": ("ZrO3", -2),
    "aluminate": ("AlO2", -1), "ferrate": ("FeO4", -2), "cobaltate": ("CoO2", -1), "germanate": ("GeO3", -2),
    "tellurate": ("TeO4", -2), "tellurite": ("TeO3", -2), "plumbate": ("PbO3", -2), "ferrite": ("FeO2", -1),
    "octaborate": ("B8O13", -2),
}
COMMON_STATE = {"Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1, "Be": 2, "Mg": 2, "Ca": 2, "Sr": 2, "Ba": 2, "Al": 3, "Ga": 3, "In": 3,
                "Tl": 1, "Sn": 2, "Pb": 2, "Bi": 3, "Zn": 2, "Cd": 2, "Hg": 2, "Cu": 2, "Ag": 1, "Au": 3, "Ni": 2, "Co": 2, "Fe": 3,
                "Mn": 2, "Cr": 3, "V": 5, "Ti": 4, "Zr": 4, "Hf": 4, "Nb": 5, "Ta": 5, "Mo": 6, "W": 6, "Re": 7, "Ru": 4, "Os": 4,
                "Rh": 3, "Ir": 4, "Pd": 2, "Pt": 2, "Sc": 3, "Y": 3, "La": 3, "Ce": 3, "Th": 4, "U": 6, "B": 3, "Si": 4, "Ge": 4,
                "As": 3, "Sb": 3, "Te": 4, "Se": 4, "NH4": 1, "Nd": 3, "Pr": 3, "Sm": 3, "Eu": 3, "Gd": 3, "Tb": 3, "Dy": 3,
                "Ho": 3, "Er": 3, "Tm": 3, "Yb": 3, "Lu": 3}
FILLER = {".", "%", "…", "anhydrous", "hydrate", "monohydrate", "dihydrate", "trihydrate", "tetrahydrate", "pentahydrate", "hexahydrate",
          "heptahydrate", "octahydrate", "nonahydrate", "decahydrate", "dodecahydrate", "hemihydrate", "hydrated", "hydrous", "non-pyrophoric", "pyrophoric", "basic", "and",
          "of", "the", "salt", "salts", "technical", "grade"}
MULT_WORDS = {"bis": 2, "tris": 3, "tetrakis": 4}
# inorganic anions left out of scope (counted as such, not as parser misses)
OUT_ANIONS = {"hypochlorite", "chlorite", "peroxodisulphate", "peroxodisulfate", "persulphate", "persulfate", "dithionite",
              "metabisulphite", "metabisulfite", "thiosulphate", "thiosulfate", "hydrosulphite", "cyanate", "cyanide", "thiocyanate",
              "fluorosilicate", "hexafluorosilicate", "hexafluoroaluminate", "fluoroborate", "tetrafluoroborate", "azide", "amide",
              "hydride", "nitride", "carbide", "boride", "silicide", "phosphide", "arsenide", "sulfide", "sulphide", "selenide",
              "telluride", "fluoride", "chloride", "bromide", "iodide", "oxychloride", "chromates", "sulfamate", "sulfamidate"}
ORGANIC_ANIONS = {"acetate", "formate", "stearate", "benzoate", "lactate", "oxalate", "citrate", "tartrate", "gluconate", "octanoate",
                  "naphthenate", "propionate", "oleate", "salicylate", "diformate", "dibenzoate", "dilactate", "methanesulfonate",
                  "octadecanoate", "butyrate", "hexanoate", "neodecanoate", "trifluoroacetate", "isodecanoate", "isooctanoate",
                  "neononanoate", "2-ethylhexanoate", "sulfosuccinate"}
SCOPE_METALS = set(ELEMENT_NAMES.values())
FORMULA_TOKEN = re.compile(r"^\(?([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)+)\)?$")
OUT_STEMS = re.compile(r"fluoride|chloride|bromide|iodide|sulphide|sulfide|selenide|telluride|phosphide|hydride|nitride|carbide|"
                       r"boride|silicide|polysulph|powder|turnings|ammonia|hydrazine|phosphine|ozone|\bcarbon\b|\bnitrogen\b|"
                       r"\boxygen\b|\bfluorine\b|\bchlorine\b|\bbromine\b|\biodine\b|\bsulphur\b|\bsulfur\b|"
                       r"\bphosphorus\b|silane|cyan|azide|amide|\(iso\)|yl\b|ane\b|ene\b|ol\b|oate\b|oic\b|amine|"
                       r"benz|phen|meth|eth|prop|but|alloy|steel|slag|dust|fume|fibre|fiber|glass|wool|coal|oil|tar|pitch|"
                       r"distillate|residue|extract|reaction|mixture|mass of|solution|\d-\w|\w-\d|bis\(|tris\(|bifluoride|"
                       r"acetate|formate|stearate|benzoate|lactate|oxalate|citrate|tartrate|gluconate|carboxylate|acid|alkyl")


def name_variants(name):
    """The bracketed variants ('nickel monoxide [1] nickel oxide [2]') in order, each cut before ';' (synonyms) and
    before 'with the exception'. Every variant is tried and the first that parses to a composition is kept (taking
    only the first variant would drop 'cadmium (non-pyrophoric) [1] cadmium oxide (non-pyrophoric) [2]')."""
    out = []
    for s in re.split(r"\[\d\]", name.replace("\n", " ")):
        s = s.split(";")[0]
        s = re.split(r"\bwith the exception\b|\bexcept\b|\bincluding\b", s, flags=re.I)[0].strip()
        if s:
            out.append(s)
    return out or [""]


def name_first_variant(name):
    return name_variants(name)[0]


GROUP_PATTERNS = [re.compile(r"^(?:inorganic )?(?P<el>[a-z]+)(?: ?\( ?(?P<st>[ivx]+) ?\))? (?:compounds|salts)\s*$"),
                  re.compile(r"^(?:inorganic )?compounds of (?P<el>[a-z]+)\s*$"),
                  re.compile(r"^(?P<el>[a-z]+) acid and its salts\s*$")]
PEROXO = re.compile(r"perboric|perborate|peroxometaborate|peroxoborate")
ANION_CENTRES = {"N", "Cl", "Br", "I", "S", "C", "P", "Si", "B", "F"}
BENIGN_CATIONS = {"Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sr", "Ba", "NH4"}


def _oxide_word(w):
    """'monoxide', 'pentoxide', 'tetraoxide', 'sesquioxide', 'dihydroxide' -> (multiplicity, anion) or None."""
    m = re.match(r"^(mon|di|tri|tetr|pent|hex|hept|oct|non|dec|sesqui|hemi)a?(oxide|hydroxide|peroxide|superoxide)$", w)
    if not m:
        return None
    pref = {"mon": 1, "di": 2, "tri": 3, "tetr": 4, "pent": 5, "hex": 6, "hept": 7, "oct": 8, "non": 9, "dec": 10, "sesqui": 1.5,
            "hemi": 0.5}[m.group(1)]
    return pref, m.group(2)


def parse_name(name):
    """-> (kind, formula_or_element, note, states). kind: 'compound', 'group', 'peroxide', 'out of scope', 'unparsed'."""
    results = [_parse_variant(v) for v in name_variants(name)]
    for r in results:
        if r[0] in ("compound", "group", "peroxide"):
            return r
    return results[0]


def _parse_variant(raw):
    if PEROXO.search(raw.lower()):
        return "peroxide", "", "peroxo-borate, set apart (S3b territory)", {}
    for tok in raw.replace("(", " ( ").replace(")", " ) ").split():
        m = FORMULA_TOKEN.match(tok)
        if m and any(ch.isdigit() for ch in tok) and tok.strip("()") not in ("CO2", "NO2"):
            try:
                c = Composition(m.group(1))
                els = {e.symbol for e in c.elements}
                if "O" in els and (els & SCOPE_METALS) and (els - {"O", "H"}) <= SCOPE_METALS | {"N", "Cl", "Br", "I", "S", "C", "Se", "P", "Cr", "Mo", "W", "V", "Mn"}:
                    return "compound", c.reduced_formula, "from the formula in the name", {}
            except Exception:  # noqa: BLE001
                pass
    s = raw.lower().replace(",", " ").replace("(", " ( ").replace(")", " ) ")
    s = re.sub(r"\s+", " ", s)
    words = s.split()
    if "compounds" in words or "salts" in words:
        for pat in GROUP_PATTERNS:
            gm = pat.match(s)
            if gm and gm.group("el") in ELEMENT_NAMES and gm.group("el") != "ammonium":
                el = ELEMENT_NAMES[gm.group("el")]
                st = ROMAN.get(gm.groupdict().get("st") or "")
                return "group", el, f"state {st}" if st else "", ({el: [st]} if st else {})
        if "organic" in words:
            return "out of scope", "", "group of organic compounds", {}
        if "compounds" in words:
            return "unparsed", "", "group entry without a single element", {}
    cations, anion, state, pending_mult = [], None, None, 1
    for i, w in enumerate(words):
        if w in ("(", ")") or w in FILLER:
            continue
        if w in ROMAN and i > 0:
            state = ROMAN[w]
            continue
        if w in MULT_WORDS:
            pending_mult = MULT_WORDS[w]
            continue
        if w in ELEMENT_ADJ:
            el, st = ELEMENT_ADJ[w]
            cations.append((el, 1, st))
            continue
        if w in ORGANIC_ANIONS or w.endswith("ate") and w not in ANIONS and w.rstrip("s") not in ANIONS and w in ORGANIC_ANIONS:
            return "out of scope", "", f"organic anion '{w}'", {}
        ow = _oxide_word(w)
        core, mult = w, pending_mult
        pending_mult = 1
        if ow:
            mult, core = ow[0], ow[1]
        elif w.startswith("hydrogen") and w[8:] in ANIONS:          # hydrogensulphate
            cations.append(("H", 1, 1))
            core = w[8:]
        elif w.endswith("s") and w[:-1] in ANIONS:                   # plural
            core = w[:-1]
        elif core not in ELEMENT_NAMES and core not in ANIONS and core != "hydrogen":
            for g in sorted(GREEK, key=len, reverse=True):
                if w.startswith(g) and len(w) > len(g) + 2:
                    cand = w[len(g):]
                    if cand in ELEMENT_NAMES or cand in ANIONS or cand == "hydrogen":
                        core, mult = cand, GREEK[g]
                        break
        if core == "hydrogen":
            cations.append(("H", mult, 1))
        elif core in ELEMENT_NAMES:
            cations.append((ELEMENT_NAMES[core], mult, None))
        elif core in ANIONS:
            if anion is not None:
                return "unparsed", "", "two anions", {}
            anion = (core, mult)
        elif core in OUT_ANIONS or core.rstrip("s") in OUT_ANIONS:
            return "out of scope", "", f"anion '{w}' outside the design's families", {}
        elif core.startswith("peroxo"):
            return "peroxide", "", f"peroxo anion '{w}', set apart (S3b territory)", {}
        elif core in ORGANIC_ANIONS:
            return "out of scope", "", f"organic anion '{w}'", {}
        else:
            return ("out of scope" if OUT_STEMS.search(raw.lower()) else "unparsed"), "", f"word '{w}'", {}
    if anion is None:
        return ("out of scope", "", "no anion (element, alloy or other)", {}) if cations else ("unparsed", "", "nothing recognized", {})
    if not cations:
        return "unparsed", "", "no cation", {}
    aname, amult = anion
    acomp, acharge = ANIONS[aname]
    for el, _, _ in cations:
        if el != "H" and el not in SCOPE_METALS:
            return "out of scope", "", f"cation {el}", {}
    explicit = any(m != 1 for _, m, _ in cations) or amult != 1
    states = {}
    if explicit and len(cations) == 1 and cations[0][1] == 1 and amult != 1 and cations[0][0] not in ("H", "NH4"):
        # 'antimony trioxide', 'arsenic pentoxide': the literal SbO3 would need Sb(VI); take the smallest cation count
        # whose oxidation state exists for the element (Sb2O3). 'manganese dioxide' stays MnO2 (Mn(IV) exists).
        el = cations[0][0]
        q_lit = int(round(-acharge * amult))
        known = set(Element(el).oxidation_states)
        if q_lit not in known:
            for n in (2, 3, 4):
                if q_lit % n == 0 and q_lit // n in known:
                    cations = [(el, n, None)]
                    break
    if explicit:
        # literal composition from the stated multiplicities (dichromium trioxide, nickel dihydroxide, tetraboron disodium heptaoxide)
        parts = []
        for el, m, st in cations:
            parts.append(f"{el}{m:g}" if m != 1 else el)
        anion_part = f"({acomp}){amult:g}" if amult != 1 and len(Composition(acomp).elements) > 1 else (f"{acomp}{amult:g}" if amult != 1 else acomp)
        f = "".join(parts) + anion_part
        if len(cations) == 1 and cations[0][0] != "H":
            el, m, _ = cations[0]
            q = -acharge * amult / m
            if abs(q - round(q)) < 1e-6:
                states = {el: [int(round(q))]}
    else:
        el, m, st = cations[0]
        q = st if st is not None else (state if state is not None else COMMON_STATE.get(el, 2))
        others = cations[1:]
        q_others = sum(mm * (1 if e == "H" else COMMON_STATE.get(e, 1)) for e, mm, _ in others)
        a = abs(acharge)
        n, k = 0, 0
        for cand_n in range(1, a + 1):                          # the smallest cation count that balances the charge
            tot = cand_n * q + q_others
            if tot > 0 and tot % a == 0:
                n, k = cand_n, tot // a
                break
        if k <= 0:
            return "unparsed", "", "charge balance failed", {}
        parts = [f"{el}{n if n > 1 else ''}"] + [f"{e}{mm if mm > 1 else ''}" for e, mm, _ in others]
        f = "".join(parts) + (f"({acomp}){k}" if k > 1 and len(Composition(acomp).elements) > 1 else (f"{acomp}{k}" if k > 1 else acomp))
        states = {el: [int(q)]}
    try:
        comp = Composition(f.replace("NH4", "(NH4)"))
    except Exception:  # noqa: BLE001
        return "unparsed", "", f"formula {f}", {}
    kind = "peroxide" if aname in ("peroxide", "superoxide") else "compound"
    if any(el in {e.symbol for e in Composition(acomp).elements} for el in states):   # mixed valence (chromium(III) chromate)
        states = {}
    return kind, comp.reduced_formula, (f"state {state}" if state else ""), states


# ---- hazard codes ------------------------------------------------------------------------------------------------------
WARN_CODES = re.compile(r"Acute Tox\. [12]\b|Carc\. 1[AB]|Muta\. 1[AB]|Repr\. 1[AB]")
CAUTION_CODES = re.compile(r"Acute Tox\. 3\b|Carc\. 2\b|Muta\. 2\b|Repr\. 2\b|STOT RE [12]\b")


def level_from_codes(codes):
    c = (codes or "").replace("\n", " ")
    if WARN_CODES.search(c):
        return "WARNING"
    if CAUTION_CODES.search(c):
        return "CAUTION"
    return "NONE"


def gate_level(formula_or_element, kind, states=None):
    if not formula_or_element:
        return "NONE", ""
    f = formula_or_element if kind != "group" else f"{formula_or_element}O"
    rules = sg.element_rules(f, states)
    s12 = [r for r in rules if "[S1]" in r[1] or "[S2]" in r[1]]
    lv = "WARNING" if any(l == "WARNING" for l, _ in s12) else ("CAUTION" if s12 else "NONE")
    return lv, "; ".join(w for _, w in s12)


# ---- workbook ------------------------------------------------------------------------------------------------------------
def read_sheet(path, names_only=False):
    z = zipfile.ZipFile(path)
    sst = ET.fromstring(z.read("xl/sharedStrings.xml"))
    ss = ["".join(t.text or "" for t in si.iter("{%s}t" % NS["m"])) for si in sst.findall("m:si", NS)]
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    sheets = [(s.get("name"), s.get("sheetId")) for s in wb.find("m:sheets", NS)]
    sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))

    def col_of(ref):
        n = 0
        for ch in "".join(c for c in ref if c.isalpha()):
            n = n * 26 + ord(ch) - 64
        return n - 1

    def val(c):
        v = c.find("m:v", NS)
        if v is None:
            t = c.find("m:is", NS)
            return "".join(x.text or "" for x in t.iter("{%s}t" % NS["m"])) if t is not None else ""
        return ss[int(v.text)] if c.get("t") == "s" else (v.text or "")
    hdr, rows = None, []
    for r in sheet.find("m:sheetData", NS).findall("m:row", NS):
        cells = {col_of(c.get("r")): val(c) for c in r.findall("m:c", NS)}
        if hdr is None:
            if cells.get(0) == "Index No":
                hdr = cells
            continue
        d = dict(index=cells.get(0, ""), atp=cells.get(1, ""), name=cells.get(3, ""), ec=cells.get(4, ""), cas=cells.get(5, ""))
        if not names_only:
            d["codes"] = cells.get(6, "")
            d["notes"] = cells.get(12, "")
        rows.append(d)
    return sheets, rows


def sha16(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


# ---- the run --------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True, help="ECHA Excel export of Annex VI to CLP")
    ap.add_argument("--out", default="clp_validation", help="output prefix (.md, .csv, _elements.csv, _unparsed.csv, _summary.json)")
    ap.add_argument("--elements", default="", help="comma-separated extra elements to report in the per-element table")
    ap.add_argument("--names-only", action="store_true")
    args = ap.parse_args()
    sheets, rows = read_sheet(args.xlsx, names_only=args.names_only)
    parsed = []
    for r in rows:
        kind, f, note, states = parse_name(r["name"])
        parsed.append(dict(r, kind=kind, formula=f, note=note, states=json.dumps(states)))
    P = pd.DataFrame(parsed)
    counts = P.kind.value_counts().to_dict()
    print("sheets", sheets, "entries", len(P), counts)
    if args.names_only:
        P[P.kind.isin(["compound", "group", "peroxide"])][["index", "name", "kind", "formula", "note"]].to_csv(args.out + "_parsed.csv", index=False)
        P[P.kind == "unparsed"][["index", "name", "note"]].to_csv(args.out + "_unparsed.csv", index=False)
        return
    seals = {os.path.basename(p): sha16(p) for p in (args.xlsx, sg.RULES_FILE)}
    ref = P[P.kind.isin(["compound", "group", "peroxide"])].copy()
    ref["clp_level"] = ref.codes.map(level_from_codes)
    gl = [gate_level(f, k) for f, k in zip(ref.formula, ref.kind)]
    ref["gate_level"] = [g[0] for g in gl]
    ref["gate_rules"] = [g[1] for g in gl]
    gs = [gate_level(f, k, json.loads(s) or None) for f, k, s in zip(ref.formula, ref.kind, ref.states)]   # reported: with the forced state
    ref["gate_level_with_state"] = [g[0] for g in gs]
    order = {"NONE": 0, "CAUTION": 1, "WARNING": 2}
    ref["gate_at_least"] = [order[g] >= order[c] for g, c in zip(ref.gate_level, ref.clp_level)]
    # the table's best level per element: over every S1/S2 rule (state-specific ones included) and over the state-free ones only
    tbl_level, tbl_free = {}, {}
    for r in sg.rules():
        if r["rule"] in ("S1", "S2"):
            lv = sg._level(r["level"])
            tbl_level[r["element"]] = max(tbl_level.get(r["element"], "NONE"), lv, key=lambda x: order[x])
            if not (r.get("oxidation_state") or "").strip():
                tbl_free[r["element"]] = max(tbl_free.get(r["element"], "NONE"), lv, key=lambda x: order[x])
    tbl_state = {r["element"] for r in sg.rules() if (r.get("oxidation_state") or "").strip()}

    def elements_of(formula, kind):
        return {formula} if kind == "group" else {e.symbol for e in Composition(formula).elements} - {"O", "H"}
    cmp_ = ref[ref.kind != "peroxide"]
    warn = cmp_[cmp_.clp_level == "WARNING"]
    cw = cmp_[cmp_.clp_level.isin(["WARNING", "CAUTION"])]
    none = cmp_[cmp_.clp_level == "NONE"]
    sens_w = float((warn.gate_level == "WARNING").mean()) if len(warn) else float("nan")
    sens_c = float((cw.gate_level != "NONE").mean()) if len(cw) else float("nan")
    over = float((none.gate_level == "WARNING").mean()) if len(none) else float("nan")
    verdict = "GATE SENSITIVE" if (sens_w >= SENS_WARN and sens_c >= SENS_CAUTION) else "GATE MISSES"
    misses = warn[warn.gate_level != "WARNING"].copy()
    misses_c = cw[cw.gate_level == "NONE"].copy()

    def cause(row):
        """The three causes of a miss. Anion centres and benign cations are not listed as causes, otherwise S, C, Na,
        K, Cl, Br would appear as causes of the bromate, chlorate and nitrite misses."""
        els = elements_of(row.formula, row.kind)
        metals = sorted(els - ANION_CENTRES - BENIGN_CATIONS)
        lower = [f"{e} ({tbl_level[e]})" for e in metals if e in tbl_level and order[tbl_level[e]] < order[row.clp_level]]
        if lower:
            return "element in the table at a lower level: " + ", ".join(lower)
        absent = [e for e in metals if e not in tbl_level]
        if absent:
            return "element absent from the table: " + ", ".join(absent)
        anions = sorted(els & ANION_CENTRES)
        if anions:
            return "hazard of the anion (" + ", ".join(anions) + "): not visible to an element rule"
        return "class or route the table did not consider"
    misses["cause"] = [cause(r) for r in misses.itertuples()]
    misses_c["cause"] = [cause(r) for r in misses_c.itertuples()]
    # elements: the table level is the best over every S1/S2 rule of the element, state-specific ones included
    # (using only the state-free rules would turn Cr(VI) WARNING into a 'gap'); an entry is attributed to an
    # element only if no other metal of the entry already explains its level in the table (otherwise nickel
    # stannate's Carc. 1A would be credited to Sn).
    cand_els = sorted(e.strip() for e in args.elements.split(",") if e.strip())
    el_rows = []
    all_els = sorted(set(cand_els) | set(tbl_level))
    cmp_els = [elements_of(f, k) for f, k in zip(cmp_.formula, cmp_.kind)]
    for el in all_els:
        has = [el in e for e in cmp_els]
        explained = [el in e and any(o != el and o not in ANION_CENTRES and o not in BENIGN_CATIONS and o in tbl_level
                                     and order[tbl_level[o]] >= order[c] for o in e)
                     for e, c in zip(cmp_els, cmp_.clp_level)]
        sub = cmp_[[h and not x for h, x in zip(has, explained)]]
        best = max(sub.clp_level, key=lambda x: order[x]) if len(sub) else "no entry"
        t_lv = tbl_level.get(el, "NONE")
        gap = best in order and order[best] > order[t_lv]
        other = sorted({c.strip() for s in sub.codes for c in s.replace("\n", ";").split(";") if c.strip() and not WARN_CODES.search(c) and not CAUTION_CODES.search(c)})[:8]
        el_rows.append(dict(element=el, requested_element=el in cand_els, annex_entries=int(len(sub)), entries_explained_by_another_element=int(sum(explained)),
                            annex_best_level=best, table_level=t_lv, table_level_state_free=tbl_free.get(el, "state-specific only" if el in tbl_state else "NONE"),
                            gap=bool(gap), example=sub.name.iloc[0][:50] if len(sub) else "", classes_outside_gate=", ".join(other)))
    E = pd.DataFrame(el_rows)
    sens_w_state = float((warn.gate_level_with_state == "WARNING").mean()) if len(warn) else float("nan")
    sens_c_state = float((cw.gate_level_with_state != "NONE").mean()) if len(cw) else float("nan")
    over_state = float((none.gate_level_with_state == "WARNING").mean()) if len(none) else float("nan")
    S = dict(seals=seals, sheets=sheets, entries=int(len(P)), kinds=counts, reference=int(len(cmp_)), peroxides=int((ref.kind == "peroxide").sum()),
             clp_levels=cmp_.clp_level.value_counts().to_dict(), gate_levels=cmp_.gate_level.value_counts().to_dict(),
             sensitivity_warning=sens_w, n_warning=int(len(warn)), sensitivity_caution_or_warning=sens_c, n_caution_or_warning=int(len(cw)),
             overflag_share=over, n_none=int(len(none)), verdict=verdict,
             with_state=dict(sensitivity_warning=sens_w_state, sensitivity_caution_or_warning=sens_c_state, overflag_share=over_state),
             misses_warning=misses[["index", "name", "formula", "codes", "gate_level", "cause"]].to_dict("records"),
             misses_caution=misses_c[["index", "name", "formula", "codes", "cause"]].to_dict("records"),
             overflag_by_element=none[none.gate_level == "WARNING"].formula.map(lambda f: ",".join(sorted({e.symbol for e in Composition(f).elements} - {"O", "H"}))).value_counts().to_dict(),
             element_gaps=E[E.gap].element.tolist(), requested_elements_without_entry=E[E.requested_element & (E.annex_entries == 0)].element.tolist())
    write(S, ref, E, P[P.kind == "unparsed"], args.out)
    print(verdict, f"sens W {sens_w:.3f} ({len(warn)}), sens C+W {sens_c:.3f} ({len(cw)}), overflag {over:.3f} ({len(none)}); gaps {S['element_gaps']}")


def write(S, ref, E, U, OUT):
    with open(OUT + "_summary.json", "w", encoding="utf-8") as fh:
        json.dump(S, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    ref.to_csv(OUT + ".csv", index=False)
    E.to_csv(OUT + "_elements.csv", index=False)
    U[["index", "name", "note"]].to_csv(OUT + "_unparsed.csv", index=False)
    md = ["# Validation of the safety gate's element rules against Annex VI to CLP (ATP 23)", "",
          "Written by `safety_gate/clp_validation.py`. Reference set: the "
          "Annex VI entries whose name parses to one inorganic composition within the gate's scope (oxides, hydroxides, single-oxyanion "
          "salts; 'X compounds' group entries at element level); peroxides parsed and set apart. Annex VI level from the hazard codes with the "
          "gate's S1 definitions (WARNING: Acute Tox. 1-2, Carc./Muta./Repr. 1A-1B; CAUTION: Acute Tox. 3, CMR 2, STOT RE). Gate level: the "
          "S1 and S2 rules on the composition, no oxidation-state information (conservative).", "",
          f"Entries {S['entries']}; parsed into the reference set {S['reference']} ({S['kinds'].get('compound', 0)} compounds, "
          f"{S['kinds'].get('group', 0)} group entries; {S['peroxides']} peroxides apart); out of scope {S['kinds'].get('out of scope', 0)}; "
          f"unparsed {S['kinds'].get('unparsed', 0)} (listed in `_unparsed.csv`). Annex VI levels in the reference set: {S['clp_levels']}; "
          f"gate levels: {S['gate_levels']}.", "",
          f"**{S['verdict']}.** WARNING-level substances labelled WARNING by the gate: {100 * S['sensitivity_warning']:.1f} % of {S['n_warning']} "
          f"(threshold {100 * SENS_WARN:.0f} %); CAUTION-or-WARNING substances labelled at least CAUTION: {100 * S['sensitivity_caution_or_warning']:.1f} % "
          f"of {S['n_caution_or_warning']} (threshold {100 * SENS_CAUTION:.0f} %). Over-flagging (reported): {100 * S['overflag_share']:.1f} % of the "
          f"{S['n_none']} substances without a gate-level classification are labelled WARNING, by element: {S['overflag_by_element']}. "
          f"Second pass with the oxidation state the name forces (reported, not the criterion): {100 * S['with_state']['sensitivity_warning']:.1f} % / "
          f"{100 * S['with_state']['sensitivity_caution_or_warning']:.1f} % / over-flagging {100 * S['with_state']['overflag_share']:.1f} %.", "",
          "## 1. Misses: WARNING in Annex VI, not WARNING in the gate", "", "| index | name | formula | Annex VI codes | gate | cause |", "|---|---|---|---|---|---|"]
    md += [f"| {m['index']} | {m['name'][:60]} | {m['formula']} | {m['codes'].replace(chr(10), ' ')[:60]} | {m['gate_level']} | {m['cause']} |" for m in S["misses_warning"]] or ["| - | none | | | | |"]
    md += ["", "## 2. Misses: CAUTION or WARNING in Annex VI, no label in the gate", "", "| index | name | formula | Annex VI codes | cause |", "|---|---|---|---|---|"]
    md += [f"| {m['index']} | {m['name'][:60]} | {m['formula']} | {m['codes'].replace(chr(10), ' ')[:60]} | {m['cause']} |" for m in S["misses_caution"]] or ["| - | none | | | |"]
    md += ["", "## 3. The requested elements and the elements of the rules table against Annex VI", "",
           "Table level: the best over every S1/S2 rule of the element, state-specific rules included (state-free level also shown). "
           "An entry is attributed to an element only when no other metal of the entry already explains its level in the table "
           "(the count of such entries is shown).", "",
           "| element | requested | Annex VI entries attributed | explained by another element | best Annex VI level | table level | state-free level | gap | example | classes outside the gate's scope |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {r.element} | {'yes' if r.requested_element else ''} | {r.annex_entries} | {r.entries_explained_by_another_element} | {r.annex_best_level} | {r.table_level} | "
           f"{r.table_level_state_free} | {'GAP' if r.gap else ''} | {r.example} | {r.classes_outside_gate} |" for r in E.itertuples()]
    md += ["", f"Requested elements with no in-scope Annex VI entry (no harmonised classification is not evidence of safety; S7 covers this): "
           f"{', '.join(S['requested_elements_without_entry']) or 'none'}.", "",
           "## What this does not show",
           "- Annex VI lists only substances with a harmonised classification; absence is not safety.",
           "- A parser miss removes an entry from the reference set; the unparsed list is for manual review.",
           "- The gate labels by element on purpose; over-flagging is a cost accepted by design.",
           "- Nothing here changes a rule; gaps are proposals for the rules table, for its maintainer to approve.",
           "", "Seals (SHA-256[:16]): " + "; ".join(f"{k} {x}" for k, x in S["seals"].items()) + "."]
    with open(OUT + ".md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
