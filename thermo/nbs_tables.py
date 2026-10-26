"""Experimental formation enthalpies from a public-domain source, with every choice recorded.

THE SOURCE

The NBS Tables of Chemical Thermodynamic Properties (Wagman et al., J. Phys. Chem. Ref. Data 11,
Supplement 2, 1982), with the 1989 erratum values folded in, released by NIST as a machine-readable
CSV in the public domain:

    https://data.nist.gov/od/id/mds2-2124
    https://data.nist.gov/od/ds/mds2-2124/NBS_Tables_with_ErratumValues.csv

Local copy: data/nbs_wagman_1982.csv (15,552 species rows), or the path in the environment
variable NBS_TABLES_CSV.

Section 5 of the CRC Handbook is a reprint of these same NBS tables, so the paywalled reference and
the free one carry identical numbers. It is also the lineage HSC Chemistry draws on.

WHY POLYMORPH SELECTION IS NOT A DETAIL

Several common compounds have multiple crystalline forms with meaningfully different formation
enthalpies. Picking silently would bury a few kJ/mol in every downstream number:

    TiO2     four crystalline entries spanning 25 kJ/mol
    PbO      yellow (massicot) and red (litharge) differ by 1.7 kJ/mol
    Zn(OH)2  beta and epsilon differ by 1.3 kJ/mol
    Co(OH)2  three precipitate descriptions, only one carrying a value

So every selection below names the state code AND the reason. A reader can disagree with a choice,
which is the point; they cannot fail to see that a choice was made.

Run:  python -m thermo.nbs_tables [output.csv]   writes the curated table with its reasons.
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.environ.get("NBS_TABLES_CSV") or os.path.join(ROOT, "data", "nbs_wagman_1982.csv")
OUT = os.path.join(ROOT, "data", "reference_enthalpies.csv")

SOURCE = "NBS Tables (Wagman 1982, + 1989 erratum), NIST public domain, data.nist.gov mds2-2124"

# target formula -> (formula as spelled in NBS, state code, scale factor, reason for the choice)
# scale factor converts the NBS entry to one mole of the TARGET formula
SELECT = {
    # --- oxides of common metals and metalloids
    "Al2O3":   ("Al2O3",  "cr",  1.0,   "corundum, the only crystalline entry"),
    "MgO":     ("MgO",    "cr",  1.0,   "periclase"),
    "SiO2":    ("SiO2",   "cr",  1.0,   "quartz"),
    "B2O3":    ("B2O3",   "cr",  1.0,   ""),
    "ZrO2":    ("ZrO2",   "cr",  1.0,   "baddeleyite"),
    "TiO2":    ("TiO2",   "cr3", 1.0,   "rutile, the stable form; cr is anatase at -939.7"),
    "H2O":     ("H2O",    "l",   1.0,   "liquid, the standard state"),
    "H2O(g)":  ("H2O",    "g",   1.0,   "gas, the state relevant at high temperature"),
    # --- transition-metal and main-group oxides and salts
    "CuO":     ("CuO",    "cr",  1.0,   "tenorite"),
    "Cu2O":    ("Cu2O",   "cr",  1.0,   "cuprite"),
    "Fe2O3":   ("Fe2O3",  "cr",  1.0,   "hematite"),
    "Fe3O4":   ("Fe3O4",  "cr",  1.0,   "magnetite"),
    "MoO3":    ("MoO3",   "cr",  1.0,   ""),
    "NiO":     ("NiO",    "cr",  1.0,   "bunsenite"),
    "WO3":     ("WO3",    "cr",  1.0,   ""),
    "MnO2":    ("MnO2",   "cr",  1.0,   "pyrolusite"),
    "Co3O4":   ("Co3O4",  "cr",  1.0,   ""),
    "V2O5":    ("V2O5",   "cr",  1.0,   ""),
    "Cr2O3":   ("Cr2O3",  "cr",  1.0,   "eskolaite"),
    "SnO2":    ("SnO2",   "cr",  1.0,   "cassiterite"),
    "ZnO":     ("ZnO",    "cr",  1.0,   "zincite"),
    "Ag2O":    ("Ag2O",   "cr",  1.0,   ""),
    "PbO":     ("PbO",    "cr2", 1.0,   "red litharge, the stable form; yellow massicot is -217.32"),
    "Sb2O3":   ("Sb4O6",  "cr",  0.5,   "NBS files this as the dimer Sb4O6 (senarmontite); halved"),
    "I2O5":    ("I2O5",   "cr",  1.0,   ""),
    "TeO2":    ("TeO2",   "cr",  1.0,   ""),
    "SeO2":    ("SeO2",   "cr",  1.0,   ""),
    "CdO":     ("CdO",    "cr",  1.0,   ""),
    "Nb2O5":   ("Nb2O5",  "cr",  1.0,   ""),
    "Ta2O5":   ("Ta2O5",  "cr",  1.0,   ""),
    "CeO2":    ("CeO2",   "cr",  1.0,   ""),
    "Bi2O3":   ("Bi2O3",  "cr",  1.0,   ""),
    "KNO3":    ("KNO3",   "cr",  1.0,   ""),
    "NaNO3":   ("NaNO3",  "cr",  1.0,   ""),
    "KClO4":   ("KClO4",  "cr",  1.0,   ""),
    "KIO3":    ("KIO3",   "cr",  1.0,   ""),
    "AgIO3":   ("AgIO3",  "cr",  1.0,   ""),
    "CaO":     ("CaO",    "cr",  1.0,   ""),
    # --- hydroxides
    "Mg(OH)2": ("Mg(OH)2", "cr",  1.0,  "brucite"),
    "Ca(OH)2": ("Ca(OH)2", "cr",  1.0,  "portlandite"),
    "Cu(OH)2": ("Cu(OH)2", "cr",  1.0,  ""),
    "Zn(OH)2": ("Zn(OH)2", "cr3", 1.0,  "epsilon, the stable form; beta is -641.91"),
    "Ni(OH)2": ("Ni(OH)2", "cr",  1.0,  ""),
    "Co(OH)2": ("Co(OH)2", "cr2", 1.0,  "pink precipitate, the only form carrying a value"),
    "Mn(OH)2": ("Mn(OH)2", "am",  1.0,  "NBS lists only an amorphous precipitate"),
    "Al(OH)3": ("Al(OH)3", "am",  1.0,  "NBS LISTS ONLY AMORPHOUS. Gibbsite is about -1293 elsewhere"),
    "La(OH)3": ("La(OH)3", "cr",  1.0,  ""),
    "Bi(OH)3": ("Bi(OH)3", "cr",  1.0,  ""),
    "Fe(OH)3": ("Fe(OH)3", "cr",  1.0,  "precipitated"),
    "Cr(OH)3": ("Cr(OH)3", "cr",  1.0,  ""),
    "Sn(OH)4": ("Sn(OH)4", "cr",  1.0,  ""),
}

# Genuinely not in this source, under any spelling that was searched for.
NOT_IN_NBS = {
    "Ce(OH)4": "no Ce-O-H entry of any kind; the Ce rows are oxides, salts and aqueous complexes",
    "Y(OH)3": "the crystalline row EXISTS but its formation enthalpy cell is empty - listed, not measured",
    "Ti(OH)4": "no Ti-O-H entry; the Ti hydrogen rows are hydride and adducts",
}

_DF = None
DFH = "ΔfH°"
CP = "Cp J mol-1 K-1"
S_PREFIX = "S°"                  # the S° header carries a run of blanks: matched by prefix


def _table():
    global _DF
    if _DF is None:
        d = pd.read_csv(CSV, header=1, dtype=str, encoding="utf-8-sig", low_memory=False)
        d.columns = [str(c).replace("\n", " ").strip() for c in d.columns]
        d["Formula"] = d["Formula"].astype(str).str.strip()
        d["State"] = d["State"].astype(str).str.strip()
        _DF = d
    return _DF


def _value(nbs_formula, state):
    d = _table()
    hit = d[(d["Formula"] == nbs_formula) & (d["State"] == state)]
    if not len(hit):
        return None, None
    raw = str(hit.iloc[0][DFH]).strip()
    desc = str(hit.iloc[0]["State Description"]).strip()
    txt = raw.rstrip("*").strip()
    if txt in ("", "-", "nan"):
        return None, desc
    try:
        return float(txt), desc
    except ValueError:
        return None, desc


def _num(raw):
    txt = str(raw).strip().rstrip("*").strip()
    if txt in ("", "-", "nan"):
        return None
    try:
        return float(txt)
    except ValueError:
        return None


def lookup(formula, state=None):
    """dHf° (kJ/mol), S° and Cp° (J/(mol·K)) at 298.15 K and 1 bar, per mole of the requested formula.

    If the formula is in SELECT its recorded choice is used (NBS formula, state and scale, which
    also applies to S° and Cp°). If it is not, the lookup needs an explicit state
    (state='cr', 'l', 'am'...), without guessing polymorphs, and the row comes back curated=False.
    Returns a dict, or None if there is no row. A value NBS does not give ('-') comes back as None."""
    if formula in SELECT and state is None:
        nbs_f, st, scale, why = SELECT[formula]
        curated = True
    elif state is not None:
        nbs_f, st, scale, why, curated = formula, state, 1.0, "", False
    else:
        return None
    d = _table()
    hit = d[(d["Formula"] == nbs_f) & (d["State"] == st)]
    if not len(hit):
        return None
    row = hit.iloc[0]
    s_col = next(c for c in d.columns if c.startswith(S_PREFIX))
    scaled = lambda v: None if v is None else v * scale          # noqa: E731
    desc = str(row["State Description"]).strip()
    return dict(formula=formula, nbs_formula=nbs_f, state=st, scale=scale,
                dHf_kJ=scaled(_num(row[DFH])), S_J_per_molK=scaled(_num(row[s_col])),
                Cp_J_per_molK=scaled(_num(row[CP])),
                state_description=None if desc in ("nan", "") else desc,
                reason=why, curated=curated)


def build():
    rows = []
    for target, (nbs_f, state, scale, why) in SELECT.items():
        v, desc = _value(nbs_f, state)
        rows.append(dict(
            formula=target, dHf_kJ_per_mol=None if v is None else round(v * scale, 3),
            nbs_formula=nbs_f, nbs_state=state, scale=scale,
            nbs_state_description=None if desc in ("nan", "") else desc,
            selection_reason=why, source=SOURCE,
            status="ok" if v is not None else "NO VALUE IN SOURCE"))
    for target, why in NOT_IN_NBS.items():
        rows.append(dict(formula=target, dHf_kJ_per_mol=None, nbs_formula=None, nbs_state=None,
                         scale=None, nbs_state_description=None, selection_reason=why,
                         source=SOURCE, status="ABSENT FROM SOURCE"))
    return pd.DataFrame(rows)


def main(out=OUT):
    if not os.path.exists(CSV):
        raise SystemExit("missing {}\n  download from {}".format(CSV, SOURCE))
    df = build()
    df.to_csv(out, index=False, encoding="utf8")

    ok = df[df.status == "ok"]
    bad = df[df.status != "ok"]
    print("EXPERIMENTAL FORMATION ENTHALPIES, sourced\n")
    print("  source : {}".format(SOURCE))
    print("  local  : {}".format(CSV))
    print("  wrote  : {}\n".format(out))
    print("  {} of {} target compounds resolved to a value.\n".format(len(ok), len(df)))

    print("  SELECTIONS THAT REQUIRED A JUDGEMENT CALL\n")
    for _, r in ok.iterrows():
        if r.selection_reason and (r.nbs_state not in ("cr", "l", "g") or "ONLY" in r.selection_reason):
            print("    {:<10} {:>11.2f}  [{}] {}".format(
                r.formula, r.dHf_kJ_per_mol, r.nbs_state, r.selection_reason))

    if len(bad):
        print("\n  UNRESOLVED\n")
        for _, r in bad.iterrows():
            print("    {:<10} {}".format(r.formula, r.selection_reason))

    print("""
  NOTE

    Al(OH)3   The NBS tables carry only an AMORPHOUS entry. The gibbsite value (about -1293 kJ/mol)
              found in other compilations is the crystalline mineral. Those are different substances
              with genuinely different enthalpies, not two transcriptions of one number. Which one
              belongs in a calculation depends on which material is actually used.""")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else OUT)
