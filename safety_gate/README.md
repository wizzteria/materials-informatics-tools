# safety_gate

A hazard-label gate for computational screening of inorganic compounds. It
looks at a composition (and, if given, its relaxed structure, oxidation states
and a predicted oxygen-release temperature) and attaches a hazard label with
its cause and its source. It never removes a row: a flagged composition keeps
the scientific information it carries, and the reader sees why it is flagged.

## What it does

`safety_gate.gate(formula, structure=None, states=None, t_release_K=nan, known_compound=False)`
returns `hazard_level` (WARNING > CAUTION > NONE), the reasons tagged by rule,
and a statement on laboratory use. The rules, declared before the evidence was
collected:

- S1, S2: element rules from `data/safety_rules.csv` (31 rows: IARC Group 1/2,
  EU CLP harmonised Acute Tox., Carc./Muta./Repr. classifications, radioactivity,
  volatile toxic species), each with a public source and URL, each confirmed by
  an independent verifier. A rule may be restricted to an oxidation state or a
  composition pattern (`applies_to`).
- S3: composition outside the oxide/hydroxide scope; O-O bonds (peroxide- or
  superoxide-like units) in the structure.
- S4: predicted oxygen-release temperature near room temperature, with a 100 K
  thermal-process-safety margin and two calibration parameters of the upstream
  estimator (set them to your own estimator's errors).
- S5, S6: compositions without a verified literature review (`LIT_FILE`, a JSON
  `{"rows": [{"formula": ...}]}`) are labelled as not characterized.
- S7, S8: elements with no collected evidence get CAUTION; the module warns when
  a scheduled re-check date has passed.
- S10-S12: standing notes on oxidizing solids and combustible dusts (`dust_note`).

`clp_validation.py` checks the element rules against the EU CLP Annex VI
harmonised classifications: it parses the substance names of the ECHA Excel
export into compositions, maps the hazard codes to the gate's levels and
reports sensitivity and over-flagging. The workbook is not shipped; download it
from https://echa.europa.eu/information-on-chemicals/annex-vi-to-clp.

## How to use it

```python
from safety_gate import safety_gate as sg
sg.gate("CdO", states={"Cd": [2]}, known_compound=True)
sg.element_rules("CrO3", {"Cr": [6]})
sg.anion_clp_note("KBrO3")
```

```
python -m safety_gate.safety_gate                                        # self-test
python -m unittest discover -s tests -p "test_safety_gate*.py"           # tests
python -m safety_gate.clp_validation --xlsx annex_vi.xlsx --out out/clp  # validation
```

## Where the rules come from

IARC Monographs (Vols. 86, 87, 93, 100C, 100D, 118, 131, 136); EU CLP
Regulation Annex VI (ECHA export, ATP 23) and the GB mandatory classification
list; NIOSH Pocket Guide; PubChem/HSDB; NIST Chemistry WebBook; the pymatgen
`OxideType` cut-offs for O-O bonds; TRAS 410 and DGUV Information 213-067 for
the 100 K margin; OSHA 29 CFR 1910.1200, ECHA CLP guidance 2.14, GHS Annex 11
and the OSHA Technical Manual for the powder notes. Every row of the rules
table cites its source and URL.
