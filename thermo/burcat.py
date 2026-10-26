"""Parser of the Burcat Extended Third Millennium thermochemical database (NASA-7 polynomials).

The database file (BURCAT.THR) is not shipped with this package. Download it from the official
site of A. Burcat and B. Ruscic, "Third Millennium Ideal Gas and Condensed Phase Thermochemical
Database for Combustion with Updates from Active Thermochemical Tables":

    http://garfield.chem.elte.hu/Burcat/burcat.html      (mirror, file BURCAT.THR)
    https://burcat.technion.ac.il/                        (Technion home page)

Give the path to `BurcatDatabase(path)` or set the environment variable BURCAT_THR.

File format: NASA 7-coefficient records in fixed columns. Each block is 4 lines:
  Line 1: name (0:18), ref code (18:24), 4 elements x 5 chars (24:44),
          phase G/S/L/C (44), T_low (45:55), T_high (55:65), column 79 = '1'
  Lines 2-4: 5 coefficients of 15 chars each in (0:75), column 79 = '2','3','4'

The 15 coefficients of a block:
  1-7:   high-temperature set a1..a7
  8-14:  low-temperature set a1..a7
  15:    dHf(298)/R of the reference phase at 298 K

Single range: if one of the two sets is all zero, the other applies to the whole range. This
happens in both directions: 148 condensed blocks (among them AL(cr), solid aluminium, and Bi(S))
keep their only polynomial in the LOW set; others, such as AL2O3(b), in the high set.

Equations (per mole):
  Cp/R = a1 + a2*T + a3*T^2 + a4*T^3 + a5*T^4
  H/RT = a1 + a2*T/2 + a3*T^2/3 + a4*T^3/4 + a5*T^4/5 + a6/T
  S/R  = a1*ln(T) + a2*T + a3*T^2/2 + a4*T^3/3 + a5*T^4/4 + a7

H is already the absolute enthalpy on the formation scale (a6 carries it): in a block that
contains 298.15 K, H(298.15) = dHf(298). It is not re-referenced: doing so in a block that does
not contain 298 K (liquids, high-T polymorphs) would erase the transition enthalpies.

The join point T_mid is not stored in the file: it is found by continuity of H and Cp (1000 K
for the gases, 700 K for AL2O3(a)). If a two-set block does not join cleanly, a set is NEVER
chosen blindly, not even with strict=False: either the defect register records a choice (with
its evidence), or the caller asks for use_set='high'/'low' on purpose.

Defect register (optional CSV, see `load_defects_csv`): a defect in Burcat is a fact about the
file, not a preference of one project, so everyone reading the file should see the same
exceptions. The register is read when a path (or a list of rows) is passed as `defects`, or when
the environment variable BURCAT_DEFECTS_CSV points to one. A row that does not match the file is
an error. The repository ships a register of the defects found so far in data/burcat_known_defects.csv.

strict=True (default): evaluating outside [T_low, T_high] raises BurcatRangeError.
strict=False only allows extrapolation; it does not touch the join rule.
"""
from __future__ import annotations

import csv
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import brentq

R_GAS = 8.31446261815324  # J/(mol·K)
T_REF = 298.15            # K
JOIN_TOL = 1.0            # |dH| in kJ/mol + |dCp| in J/(mol·K) at the join point

VALID_PHASES = {'G', 'S', 'L', 'C'}
CONDENSED = {'S', 'L', 'C'}                     # Burcat uses C for condensed (solid or liquid)
PHASE_GROUPS = {'gas': {'G'}, 'condensed': CONDENSED}

_ELEMENTS = frozenset([
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
    'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te',
    'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm',
    'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og',
    'D', 'T',  # hydrogen isotopes that Burcat writes explicitly; they are not collapsed to H
])

_REF_RE = re.compile(r'REF\s*=\s*(.+?)(?=\s+HF298|\s+HF0|\s+Max|\s+MAX|\s+REF\s*=|\s*$)')
_BRACES_RE = re.compile(r'\{[^{}]*\}')


def _comment_refs(comments: list[str]) -> tuple[str, str]:
    """References from the record comment, outside braces. Burcat puts alternatives (other HF298
    values with their REF) and sometimes molecular constants between braces; those braces can span
    several lines and are removed before searching. (Taking the REF from inside the braces when the
    first one is followed by a brace would give Al 'JANAF 83}' although its value is CODATA 1989.)
    Returns (ref, ref_hf298):
      ref       the first REF outside braces; '-' if there is none.
      ref_hf298 the REF that follows the first HF298 outside braces, i.e. the declared source of
                the adopted dHf; '' if the comment does not declare it."""
    text, prev = ' '.join(comments), None
    while prev != text:                                       # nested braces
        prev, text = text, _BRACES_RE.sub(' ', text)
    text = re.sub(r'\{.*$', ' ', text).replace('}', ' ')     # unclosed or leftover brace
    m = _REF_RE.search(text)
    ref = m.group(1).strip() if m else '-'
    ref_hf, i = '', text.find('HF298')
    if i >= 0:
        m = _REF_RE.search(text, i)
        if m and 'HF298' not in text[i + 5:m.start()]:
            ref_hf = m.group(1).strip()
    return ref, ref_hf
_COEFF_LINE_RE = re.compile(r'^\s*[-+]?\d\.\d{8}E[-+]\d{2}')

_env_defects = os.environ.get('BURCAT_DEFECTS_CSV')
DEFECTS_CSV: Optional[Path] = Path(_env_defects) if _env_defects else None

# Register rules. The parser rules apply block by block and need burcat_line.
PARSER_RULES = {
    'use_set_high', 'use_set_low',                              # set chosen in a block without a join
    'set_phase_S', 'set_phase_L', 'set_phase_C', 'set_phase_G',  # wrong phase letter in the file
    'exclude_from_formula_index',                               # composition miswritten (tritium as H or D)
}
# Applied per species by thermo_sources.py; the parser only checks that the line matches.
SPECIES_RULES = {'use_nbs_dHf', 'flag_only', 'no_calculate_above_T'}


def load_defects_csv(path) -> list[dict]:
    """Read the defect register (utf-8-sig, in case it was saved from Excel). Each row carries
    '_row', its line number in the CSV, for error messages.

    Columns: species, phase, source ('Burcat' rows are validated against the file), defect_type,
    description, correct_source, burcat_line (line of the block's first line), evidence,
    override_rule (one of PARSER_RULES or SPECIES_RULES), T_K (for no_calculate_above_T)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Defect register not found: {path}. '
                                f'defects=None skips it and reads the raw file.')
    with open(path, encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    for n, r in enumerate(rows, start=2):
        r['_row'] = n
    return rows


class BurcatJoinError(ValueError):
    """Two-set block without a clean join and without an explicit choice of set."""


class BurcatRangeError(ValueError):
    """A property was requested outside [T_low, T_high] with strict=True."""


# Older names, kept for compatibility.
NoJoinError = BurcatJoinError
OutOfRangeError = BurcatRangeError


def _poly_cp(c, T):
    return R_GAS * (c[0] + c[1]*T + c[2]*T*T + c[3]*T**3 + c[4]*T**4)


def _poly_H(c, T):
    return R_GAS * T * (c[0] + c[1]*T/2 + c[2]*T*T/3 + c[3]*T**3/4 + c[4]*T**4/5 + c[5]/T)


def _poly_S(c, T):
    return R_GAS * (c[0]*np.log(T) + c[1]*T + c[2]*T*T/2 + c[3]*T**3/3 + c[4]*T**4/4 + c[6])


@dataclass(eq=False)          # identity equality: ndarray fields break == and `in`
class BurcatBlock:
    """One 4-line NASA block."""
    name: str
    phase: str                          # 'G','S','L','C'
    T_low: float
    T_high: float
    elements: dict                      # exact symbols: {'Al': 2.0, 'O': 3.0}; D and T are kept
    coeffs_high: np.ndarray             # high set (in a single-range block, the only set)
    coeffs_low: np.ndarray              # low set (in a single-range block, the same only set)
    dHf_298: float                      # J/mol, coefficient 15 x R
    ref: str
    source_line: int
    charge: float = 0.0
    single_range: bool = False
    single_set: Optional[str] = None    # which set held the only polynomial: 'high', 'low' or 'both' (equal)
    forced_set: Optional[str] = None    # choice recorded in the defect register
    forced_reason: str = ''
    T_mid: Optional[float] = None       # recomputed in __post_init__
    join_residual: float = 0.0
    phase_in_file: Optional[str] = None  # original letter if the register corrected it
    ref_code: str = ''                  # source and date code of the header, columns 19-24 (tpis96, B  /89...)
    ref_hf298: str = ''                 # declared source of the adopted dHf (the REF after the first HF298), if any

    def __post_init__(self):
        if not self.single_range:
            self.T_mid, self.join_residual = self._find_join()

    # ---------------------------------------------------------------- set selection
    @property
    def join_ok(self) -> bool:
        return self.is_join_ok()

    def is_join_ok(self, tol: float = JOIN_TOL) -> bool:
        """True if the block is single-range or its two sets join cleanly."""
        return self.single_range or self.join_residual < tol

    def contains(self, T: float) -> bool:
        return self.T_low - 1e-6 <= T <= self.T_high + 1e-6

    def _coeffs_at(self, T: float, strict: bool = True, use_set: Optional[str] = None) -> np.ndarray:
        if strict and not self.contains(T):
            raise BurcatRangeError(f'{self.name} ({self.phase}): T = {T:.2f} K outside [{self.T_low}, {self.T_high}]')
        if use_set is not None:                                  # the caller's deliberate choice
            if use_set not in ('high', 'low'):
                raise ValueError("use_set must be 'high' or 'low'")
            return self.coeffs_high if use_set == 'high' else self.coeffs_low
        if self.single_range:
            return self.coeffs_high
        if self.forced_set is not None:                          # recorded choice, with evidence
            return self.coeffs_high if self.forced_set == 'high' else self.coeffs_low
        if not self.is_join_ok():                                # never blindly, not even with strict=False
            raise BurcatJoinError(
                f'{self.name} ({self.phase}, {self.T_low}-{self.T_high} K): the two sets do not join '
                f'(residual {self.join_residual:.2f}); record use_set_high or use_set_low in the '
                f'defect register (line {self.source_line}) or pass use_set')
        return self.coeffs_high if T >= self.T_mid else self.coeffs_low

    # ---------------------------------------------------------------- properties
    def cp(self, T: float, strict: bool = True, use_set: Optional[str] = None) -> float:
        """Cp in J/(mol·K)."""
        return _poly_cp(self._coeffs_at(T, strict, use_set), T)

    def H(self, T: float, strict: bool = True, use_set: Optional[str] = None) -> float:
        """H°(T) in J/mol, formation scale (H(298.15) = dHf(298) in the reference phase).
        It is the polynomial as is: not re-referenced to 298 K."""
        return _poly_H(self._coeffs_at(T, strict, use_set), T)

    def S(self, T: float, strict: bool = True, use_set: Optional[str] = None) -> float:
        """S°(T) in J/(mol·K)."""
        return _poly_S(self._coeffs_at(T, strict, use_set), T)

    def G(self, T: float, strict: bool = True, use_set: Optional[str] = None) -> float:
        """G°(T) = H°(T) - T·S°(T), in J/mol."""
        return self.H(T, strict, use_set) - T * self.S(T, strict, use_set)

    def H_low(self, T: float) -> float:
        """H°(T) with the low set, no checks (diagnostic)."""
        return _poly_H(self.coeffs_low, T)

    def H_high(self, T: float) -> float:
        """H°(T) with the high set, no checks (diagnostic)."""
        return _poly_H(self.coeffs_high, T)

    def H_raw(self, T: float, strict: bool = True) -> float:
        """Same as H(T); kept for compatibility. (It is not H(T) - H(0).)"""
        return self.H(T, strict)

    def dHf_check(self) -> Optional[float]:
        """H(298.15) of the polynomial minus dHf(298) of coefficient 15, in kJ/mol.
        Only if the block contains 298.15 K and can be evaluated; otherwise None."""
        if not self.contains(T_REF):
            return None
        try:
            return (self.H(T_REF) - self.dHf_298) / 1000.0
        except BurcatJoinError:
            return None

    def _find_join(self) -> tuple[float, float]:
        """T_mid by continuity of H and Cp: fine grid plus brentq where dH changes sign.
        Returns (T_mid, residual in kJ/mol + J/(mol·K))."""
        hi, lo = self.coeffs_high, self.coeffs_low

        def resid(T):
            return abs(_poly_H(hi, T) - _poly_H(lo, T)) / 1000.0 + abs(_poly_cp(hi, T) - _poly_cp(lo, T))

        Ts = np.linspace(self.T_low, self.T_high, 2001)
        dH = _poly_H(hi, Ts) - _poly_H(lo, Ts)
        candidates = [float(Ts[int(np.argmin(resid(Ts)))])]
        for k in np.nonzero(np.sign(dH[:-1]) * np.sign(dH[1:]) <= 0)[0]:
            try:
                candidates.append(brentq(lambda T: _poly_H(hi, T) - _poly_H(lo, T), Ts[k], Ts[k + 1]))
            except ValueError:
                pass
        best = min(candidates, key=resid)
        return float(best), float(resid(best))


# -------------------------------------------------------------------- formulas
def parse_formula(formula: str) -> Optional[dict]:
    """Formula -> {symbol: count}, with nested parentheses and decimal counts.
    Recognises D and T as symbols of their own. Returns None if a symbol is not an element.
    Examples: 'Al2O3', 'Ca(HO)2', 'Ge(OH)2', 'D2O', 'Fe0.947O'."""
    s = formula.replace(' ', '')
    stack = [{}]
    i = 0

    def number(j):
        m = re.match(r'\d+(?:\.\d+)?|\.\d+', s[j:])
        return (float(m.group(0)), j + len(m.group(0))) if m else (1.0, j)

    while i < len(s):
        m = re.match(r'[A-Z][a-z]?', s[i:])
        if m:
            sym = m.group(0)                                     # 'Co' is cobalt; 'CO' is C and O
            if sym not in _ELEMENTS:
                return None
            n, i = number(i + len(sym))
            stack[-1][sym] = stack[-1].get(sym, 0.0) + n
        elif s[i] == '(':
            stack.append({})
            i += 1
        elif s[i] == ')':
            if len(stack) == 1:
                return None
            n, i = number(i + 1)
            group = stack.pop()
            for k, v in group.items():
                stack[-1][k] = stack[-1].get(k, 0.0) + v * n
        else:
            return None
    if len(stack) != 1 or not stack[0]:
        return None
    return stack[0]


def _elements_key(elements: dict) -> tuple:
    return tuple(sorted((k, round(float(v), 6)) for k, v in elements.items()))


# -------------------------------------------------------------------- database
class BurcatDatabase:
    """Loads and queries the whole database."""

    def __init__(self, path, defects=DEFECTS_CSV):
        """defects: path of the register (default: the BURCAT_DEFECTS_CSV environment variable, or
        None), a list of rows already read, or None to read the raw file."""
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f'Not found: {self.path}')
        self.blocks: list[BurcatBlock] = []
        self.rejected: list[tuple[int, str]] = []    # (line, reason) of the rejected blocks
        self.defects_applied: list[tuple[int, str]] = []   # (line, rule) applied by the parser
        self._excluded: set[int] = set()
        self._parse()
        if defects is not None:
            rows = defects if isinstance(defects, list) else load_defects_csv(defects)
            self.apply_defects(rows, reindex=False)
        self._index()

    @staticmethod
    def _ends(line: str, digit: str) -> bool:
        s = line.rstrip()
        return len(s) >= 75 and s.endswith(digit)

    def _is_block_start(self, lines, i) -> bool:
        return (i + 3 < len(lines) and self._ends(lines[i], '1') and self._ends(lines[i + 1], '2')
                and self._ends(lines[i + 2], '3') and self._ends(lines[i + 3], '4'))

    def _parse(self):
        lines = self.path.read_text(encoding='utf-8-sig', errors='replace').splitlines()
        comments: list[str] = []
        i = 0
        while i < len(lines):
            if self._is_block_start(lines, i):
                blk = self._parse_block(lines, i, comments)
                if blk is not None:
                    self.blocks.append(blk)
                # The comment still applies to the following phases of the same species
                # (AL2O3(a), (b) and (L) share one); it is reset at the next blank line.
                i += 4
                continue
            line = lines[i]
            if line.strip() == '':
                comments = []
            elif not _COEFF_LINE_RE.match(line):
                comments.append(line.strip())
            i += 1

    def _reject(self, i, why) -> None:
        self.rejected.append((i + 1, why))
        return None

    def _parse_block(self, lines, i, comments) -> Optional[BurcatBlock]:
        s = lines[i].rstrip('\n').ljust(80)
        name = s[0:18].strip()
        elements: dict = {}
        charge = 0.0
        for k in range(4):
            field = s[24 + 5*k: 29 + 5*k]
            sym_raw, cnt_str = field[:2].strip(), field[2:].strip()
            if not sym_raw and not cnt_str:
                continue
            try:
                cnt = float(cnt_str) if cnt_str else None
            except ValueError:
                # Text such as 'WARNING' written into the element columns.
                return self._reject(i, f'unreadable element field {field!r}')
            if cnt is None:
                return self._reject(i, f'symbol without count {field!r}')
            if cnt == 0:
                continue                                          # filler: '   0.' or '0  0.'
            if sym_raw.upper() == 'E':
                charge = -cnt                                     # electrons: the species is an ion
                continue
            sym = sym_raw[0].upper() + sym_raw[1:].lower() if len(sym_raw) == 2 else sym_raw.upper()
            if sym not in _ELEMENTS:
                return self._reject(i, f'symbol that is not an element {sym_raw!r}')
            elements[sym] = elements.get(sym, 0.0) + cnt
        if not elements:
            return self._reject(i, 'no elements')
        phase = s[44].upper()
        if phase not in VALID_PHASES:
            return self._reject(i, f'unknown phase {phase!r}')
        try:
            T_low, T_high = float(s[45:55]), float(s[55:65])
        except ValueError:
            return self._reject(i, 'unreadable T range')
        if T_high <= T_low:
            return self._reject(i, 'T_high <= T_low')

        coeffs = []
        for j in range(1, 4):
            line = lines[i + j].rstrip('\n').ljust(80)
            for k in range(5):
                chunk = line[k*15:(k + 1)*15].strip()
                if not chunk:
                    coeffs.append(0.0)
                    continue
                try:
                    coeffs.append(float(chunk.replace('D', 'E')))
                except ValueError:
                    # Fortran reads '0.0       E 00' or '0.00000000E 00' as zero: blanks inside the
                    # number do not count. This is how the unused set of NaCL(cr), MgTiO3(cr) and
                    # Th(b) is written. Accepted only if it is zero; a non-zero number with blanks
                    # is still unreadable.
                    try:
                        v = float(chunk.replace(' ', '').replace('D', 'E'))
                    except ValueError:
                        v = None
                    if v != 0.0:
                        return self._reject(i, f'unreadable coefficient {chunk!r}')
                    coeffs.append(0.0)
        coeffs = np.array(coeffs)
        high, low = coeffs[0:7], coeffs[7:14]
        high_zero, low_zero = bool(np.allclose(high, 0.0)), bool(np.allclose(low, 0.0))
        if high_zero and low_zero:
            return self._reject(i, 'both sets are zero')
        if high_zero:                                             # only set in the low slot (AL(cr), Bi(S))
            coeffs_high, coeffs_low, single, single_set = low.copy(), low.copy(), True, 'low'
        elif low_zero:                                            # only set in the high slot (AL2O3(b))
            coeffs_high, coeffs_low, single, single_set = high.copy(), high.copy(), True, 'high'
        elif np.array_equal(high, low):                           # the same polynomial twice (He, H-)
            coeffs_high, coeffs_low, single, single_set = high.copy(), high.copy(), True, 'both'
        else:
            coeffs_high, coeffs_low, single, single_set = high, low, False, None

        ref, ref_hf298 = _comment_refs(comments)
        return BurcatBlock(
            name=name, phase=phase, T_low=T_low, T_high=T_high, elements=elements,
            coeffs_high=coeffs_high, coeffs_low=coeffs_low, dHf_298=float(coeffs[14]) * R_GAS,
            ref=ref, source_line=i + 1, charge=charge, single_range=single, single_set=single_set,
            ref_code=s[18:24].strip(), ref_hf298=ref_hf298,
        )

    def validate_defects(self, rows) -> list[tuple[BurcatBlock, str, dict]]:
        """Check ALL Burcat rows of the register without applying anything: known rule, the line
        starts an accepted block, and the block name starts with the row's species.
        If something does not match, the register is out of step with the file: ValueError."""
        by_line = {b.source_line: b for b in self.blocks}
        todo = []
        for r in rows:
            if (r.get('source') or '').strip() != 'Burcat':
                continue
            rule = (r.get('override_rule') or '').strip()
            sp = (r.get('species') or '').strip()
            where = f'defect register, row {r.get("_row", "?")} ({sp}, {rule})'
            if rule not in PARSER_RULES and rule not in SPECIES_RULES:
                raise ValueError(f'{where}: unknown rule {rule!r}')
            line = (r.get('burcat_line') or '').strip()
            if not line:
                if rule in PARSER_RULES:
                    raise ValueError(f'{where}: a parser rule needs burcat_line')
                continue
            b = by_line.get(int(line))
            if b is None:
                raise ValueError(f'{where}: line {line} does not start an accepted block of '
                                 f'{self.path.name}; did the file change?')
            nm, s = b.name.lower(), sp.lower()
            if not (nm.startswith(s) and (len(nm) == len(s) or not nm[len(s)].isalnum())):
                raise ValueError(f'{where}: line {line} is {b.name!r}, not {sp!r}; did the file change?')
            todo.append((b, rule, r))
        return todo

    def apply_defects(self, rows, reindex: bool = True) -> None:
        """Apply the parser rules after validating all rows."""
        for b, rule, r in self.validate_defects(rows):
            if rule in ('use_set_high', 'use_set_low'):
                b.forced_set = rule.split('_')[-1]
                b.forced_reason = (r.get('description') or '').strip()
            elif rule.startswith('set_phase_'):
                b.phase_in_file = b.phase_in_file or b.phase
                b.phase = rule[-1]
            elif rule == 'exclude_from_formula_index':
                self._excluded.add(b.source_line)
            else:
                continue                                          # rule of thermo_sources.py
            self.defects_applied.append((b.source_line, rule))
        if reindex:
            self._index()

    def _index(self):
        self._by_name = defaultdict(list)
        self._by_key = defaultdict(list)      # (exact elements, charge) -> blocks of ALL phases
        for b in self.blocks:
            self._by_name[b.name].append(b)
            if b.source_line not in self._excluded:           # still reachable by name
                self._by_key[(_elements_key(b.elements), b.charge)].append(b)

    @staticmethod
    def _phases(phase: Optional[str]) -> Optional[set]:
        if phase is None:
            return None
        p = phase.strip()
        return PHASE_GROUPS.get(p.lower(), {p.upper()})

    def get(self, name: str) -> list[BurcatBlock]:
        """All blocks with that exact name (includes ions and isotopologues)."""
        return list(self._by_name.get(name, []))

    def get_by_formula(self, formula: str, phase: Optional[str] = None,
                       charge: Optional[float] = 0.0) -> list[BurcatBlock]:
        """Blocks by exact composition. phase: 'G','S','L','C' or the groups 'gas'/'condensed';
        the filter applies block by block. charge=None returns all charges."""
        elems = parse_formula(formula)
        if elems is None:
            return []
        key = _elements_key(elems)
        wanted = self._phases(phase)
        out = []
        for (k, c), blks in self._by_key.items():
            if k != key or (charge is not None and c != charge):
                continue
            out.extend(b for b in blks if wanted is None or b.phase in wanted)
        return sorted(out, key=lambda b: (b.T_low, b.source_line))

    def pick(self, formula: str, T: float, phase: Optional[str] = None,
             charge: float = 0.0) -> Optional[BurcatBlock]:
        """The block that applies at T. If several cover T (isomers, the boundary between two
        segments, or gas and condensed when phase=None), returns the one with the lowest G(T): the
        stable form at 1 bar. If one of the blocks covering T refuses to compute (no join and no
        choice), pick refuses too with BurcatJoinError: the minimum without it would be a blind choice."""
        best, best_G = None, None
        for b in self.get_by_formula(formula, phase, charge):
            if not b.contains(T):
                continue
            g = b.G(T)
            if best is None or g < best_G:
                best, best_G = b, g
        return best

    def chain(self, formula: str, phase: Optional[str] = None, charge: float = 0.0) -> list[BurcatBlock]:
        """Blocks ordered by T_low (chains of polymorphs and phase changes)."""
        return self.get_by_formula(formula, phase, charge)
