"""Reader of NIST Shomate equations (data/nist_shomate.csv).

Fourth source of thermo_sources.py, after Burcat and NASA CEA, for species that neither of the
two has in the condensed phase. The shipped CSV carries Co3O4 and CoO (JANAF 4th ed., via the
NIST WebBook, https://webbook.nist.gov/chemistry/). Add rows in the same layout to extend it.

Shomate equations (t = T/1000):
  Cp       = A + B t + C t^2 + D t^3 + E / t^2                       J/(mol K)
  H - H298 = A t + B t^2/2 + C t^3/3 + D t^4/4 - E/t + F - H         kJ/mol
  S        = A ln t + B t + C t^2/2 + D t^3/3 - E/(2 t^2) + G        J/(mol K)
H is the source's dHf°(298), so the enthalpy on the formation scale is
  H(T) = A t + B t^2/2 + C t^3/3 + D t^4/4 - E/t + F                 kJ/mol

Note the JANAF values at 298 K for Co3O4 (dHf = -910.02, S = 114.44): USGS Bulletin 1452 (Robie,
Hemingway and Fisher 1978) and NBS give -891.2 and 102.5. thermo_sources.py anchors dHf and S on
NBS and takes from here only the T dependence, which agrees with the USGS table within 3 % up
to 900 K.

Each CSV row is a T segment; the segments of one species and phase form a block.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .burcat import BurcatRangeError, parse_formula

T_REF = 298.15


class ShomateRangeError(BurcatRangeError):
    """T outside the block's range with strict=True (subclass of the Burcat error)."""


def _key(elements: dict) -> tuple:
    return tuple(sorted((k, round(float(v), 6)) for k, v in elements.items()))


@dataclass(eq=False)
class ShomateBlock:
    name: str
    phase: str                     # 'C' (condensed) or 'G'
    elements: dict
    segments: list                 # [(T_low, T_high, (A, B, C, D, E, F, G, H)), ...] contiguous
    source_line: int               # CSV row (the first of the block)
    ref: str
    charge: float = 0.0

    @property
    def T_low(self) -> float:
        return self.segments[0][0]

    @property
    def T_high(self) -> float:
        return self.segments[-1][1]

    def contains(self, T: float) -> bool:
        # NIST ranges start at 298 K; 298.15 is accepted
        return self.T_low - 0.2 <= T <= self.T_high + 1e-6

    def _c(self, T: float, strict: bool = True):
        if strict and not self.contains(T):
            raise ShomateRangeError(f"{self.name}: T = {T:.2f} K outside [{self.T_low}, {self.T_high}]")
        for lo, hi, c in self.segments:
            if T <= hi + 1e-6:
                return c
        return self.segments[-1][2]

    def cp(self, T: float, strict: bool = True) -> float:
        A, B, C, D, E, F, G, H = self._c(T, strict)
        t = T / 1000.0
        return A + B * t + C * t**2 + D * t**3 + E / t**2

    def H(self, T: float, strict: bool = True) -> float:
        """H°(T) in J/mol, formation scale of the source (H(298) = the source's dHf)."""
        A, B, C, D, E, F, G, H = self._c(T, strict)
        t = T / 1000.0
        return 1000.0 * (A * t + B * t**2 / 2 + C * t**3 / 3 + D * t**4 / 4 - E / t + F)

    def S(self, T: float, strict: bool = True) -> float:
        A, B, C, D, E, F, G, H = self._c(T, strict)
        t = T / 1000.0
        return A * np.log(t) + B * t + C * t**2 / 2 + D * t**3 / 3 - E / (2 * t**2) + G

    def G(self, T: float, strict: bool = True) -> float:
        return self.H(T, strict) - T * self.S(T, strict)

    @property
    def dHf_298(self) -> Optional[float]:
        """Tabulated dHf°(298.15) (NIST coefficient H, in J/mol), not the polynomial at 298.15: the
        same rule as field 15 of Burcat. The fit differs by 0.05 kJ/mol for CoO and Co3O4."""
        return 1000.0 * self.segments[0][2][7] if self.contains(T_REF) else None


class ShomateDatabase:
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        groups = defaultdict(list)
        with open(self.path, encoding="utf-8-sig", newline="") as f:
            for n, r in enumerate(csv.DictReader(f), start=2):
                if r["phase"] not in ("solid", "liquid", "gas"):
                    raise ValueError(f"{self.path.name} row {n}: phase {r['phase']!r}")
                coeffs = tuple(float(r[k]) for k in "ABCDEFGH")
                groups[(r["species"], r["phase"])].append(
                    (float(r["T_low_K"]), float(r["T_high_K"]), coeffs, n,
                     f"{r['source']}, last reviewed {r['last_reviewed']}"))
        self.blocks: list[ShomateBlock] = []
        for (sp, ph), segs in groups.items():
            segs.sort(key=lambda s: s[0])
            for a, b in zip(segs[:-1], segs[1:]):
                if abs(a[1] - b[0]) > 1e-6:
                    raise ValueError(f"{sp}: segments {a[:2]} and {b[:2]} do not touch")
            el = parse_formula(sp)
            if el is None:
                raise ValueError(f"{self.path.name}: unreadable formula {sp!r}")
            self.blocks.append(ShomateBlock(
                name=f"{sp}({ph})", phase="G" if ph == "gas" else "C", elements=el,
                segments=[(s[0], s[1], s[2]) for s in segs], source_line=segs[0][3], ref=segs[0][4]))
        self._by_key = defaultdict(list)
        for b in self.blocks:
            self._by_key[(_key(b.elements), b.charge)].append(b)

    def get_by_formula(self, formula, phase: Optional[str] = None, charge: float = 0.0) -> list[ShomateBlock]:
        el = parse_formula(formula) if isinstance(formula, str) else formula
        if el is None:
            return []
        out = self._by_key.get((_key(el), charge), [])
        if phase is not None:
            want = {"gas": "G", "condensed": "C"}[phase]
            out = [b for b in out if b.phase == want]
        return sorted(out, key=lambda b: b.T_low)

    def chain(self, formula, phase: str, charge: float = 0.0) -> list[ShomateBlock]:
        """The block stable at 298.15 K (each species currently has a single one, with its segments)."""
        cands = [b for b in self.get_by_formula(formula, phase, charge) if b.contains(T_REF)]
        return [min(cands, key=lambda b: b.G(T_REF))] if cands else []
