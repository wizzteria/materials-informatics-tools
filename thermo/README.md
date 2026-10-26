# thermo

Readers for four public thermochemical sources, and one layer over them that returns a `Species`
whose every number carries the source it came from. The layer refuses to extrapolate a species
outside the temperature range its polynomial was fitted on, refuses to mix two sources inside one
temperature chain, and stops at a phase boundary that does not join physically instead of
skipping it.

## Modules

| Module | What it reads | Needs |
|---|---|---|
| `burcat.py` | Burcat Extended Third Millennium database (NASA 7-coefficient polynomials, fixed-column text). Parses every 4-line block, finds the join temperature between the two coefficient sets by continuity of H and Cp, indexes blocks by exact composition and charge, and applies an optional register of known defects in the file. | `BURCAT.THR` (external) |
| `cea_reader.py` | NASA CEA `thermo.inp` (NASA 9-coefficient polynomials) after conversion to Cantera YAML. Same block interface as Burcat (`cp`, `H`, `S`, `G`, `chain`). Matches Cantera to 1e-9 relative away from range boundaries. | converted YAML (external) |
| `cea_convert.py` | Converts `thermo.inp` to the YAML plus a phase file that `cea_reader.py` needs. | `thermo.inp` |
| `shomate_reader.py` | NIST Shomate equations from a small CSV (`data/nist_shomate.csv`). | shipped |
| `nbs_tables.py` | Formation enthalpies, entropies and heat capacities at 298.15 K from the NBS Tables of Chemical Thermodynamic Properties, with the crystalline form chosen per compound and the reason recorded. | `data/nbs_wagman_1982.csv` (shipped) |
| `thermo_sources.py` | `get_species(formula, phase)`: dHf(298) anchored on NBS when available, H(T), S(T), Cp(T) from the first source that has a chain (Burcat, then CEA, then Shomate), with a `Property` (value, units, source, ThermoChimie source type, note) for every 298 K quantity. | the above |

Every block class raises a `BurcatRangeError` subclass when asked for a temperature outside its
fitted range; pass `strict=False` to extrapolate on purpose. A Burcat block whose two sets do not
join cleanly refuses to compute until the register records a choice or the caller passes
`use_set='high'` or `'low'`.

## Data sources and licences

| Source | Where | Licence | Shipped here |
|---|---|---|---|
| Burcat and Ruscic, *Third Millennium Ideal Gas and Condensed Phase Thermochemical Database for Combustion* | http://garfield.chem.elte.hu/Burcat/burcat.html (file `BURCAT.THR`); https://burcat.technion.ac.il/ | free for research use, see the site | no: download it and set `BURCAT_THR` |
| NASA CEA `thermo.inp` (McBride, Zehe and Gordon, NASA/TP-2002-211556) | https://www1.grc.nasa.gov/research-and-engineering/ceaweb/ ; also inside the `rocketcea` package | US government work | no: convert it with `cea_convert.py` and set `CEA_NASA9_YAML` |
| NBS Tables of Chemical Thermodynamic Properties (Wagman et al. 1982, with the 1989 erratum) | https://data.nist.gov/od/id/mds2-2124 | public domain (NIST) | `data/nbs_wagman_1982.csv`, unchanged |
| NIST WebBook Shomate coefficients (Chase 1998, JANAF 4th ed.) for CoO and Co3O4 | https://webbook.nist.gov/chemistry/ | NIST Standard Reference Data; three rows of published fit coefficients transcribed with their URLs | `data/nist_shomate.csv` |
| Register of defects found in `BURCAT.THR` (phase letters, unjoined sets, units, internal offsets) | this repository | MIT | `data/burcat_known_defects.csv` |

File locations are read from environment variables (`BURCAT_THR`, `CEA_NASA9_YAML`,
`NIST_SHOMATE_CSV`, `BURCAT_DEFECTS_CSV`); the defaults point at `data/` and `data/external/`.

## Usage

```python
import os
os.environ["BURCAT_THR"] = r"C:\path\to\BURCAT.THR"          # or set it in the shell

from thermo import get_species, BurcatDatabase

al2o3 = get_species("Al2O3", "solid")
print(al2o3.dHf_298)                  # Property(value=-1675700.0, units='J/mol', source_ref='NBS', ...)
print(al2o3.H(1500.0) / 1000)         # kJ/mol, NBS anchor + Burcat increment
print(al2o3.H_with_sources(1500.0))   # (value, ['NBS', 'Burcat'])

db = BurcatDatabase(os.environ["BURCAT_THR"])
block = db.pick("Al2O3", 2000.0, "condensed")   # the stable block at 2000 K: AL2O3(b)
print(block.name, block.cp(2000.0))
```

A reaction enthalpy at any temperature is a sum over `get_species(...).H(T)` of the products and
reactants; elements in their reference state have H(298.15) = 0 on this scale.

## Tests

From the repository root:

```
python -m unittest discover -s tests -p "test_thermo_*.py"
```

`test_thermo_shomate_reader.py` and the formula-parser tests run on the shipped files alone.
`test_thermo_burcat.py`, `test_thermo_cea_reader.py` and `test_thermo_sources.py` skip with a
message when `BURCAT.THR` or the converted CEA YAML is not installed; the CEA comparison against
Cantera also skips when `cantera` is not importable. With both external files present the suite is
56 tests, all passing.

Requirements: numpy, scipy, pandas, PyYAML.
