"""Reader of the NASA CEA thermodynamic database (NASA-9 polynomials) in Cantera YAML form.

The database is not shipped with this package. The source file is `thermo.inp` from the NASA
Chemical Equilibrium with Applications program (McBride, Zehe and Gordon, NASA/TP-2002-211556):

    https://www1.grc.nasa.gov/research-and-engineering/ceaweb/      (CEA home page)
    https://cearun.grc.nasa.gov/                                     (CEA online)

The same file ships with the `rocketcea` Python package. Convert it once with
`python -m thermo.cea_convert thermo.inp cea_nasa9.yaml`, which writes the YAML and the phase
file `cea_nasa9.yaml.phase` next to it. Give the YAML path to `CEADatabase(path)` or set the
environment variable CEA_NASA9_YAML.

Format: Cantera YAML converted from thermo.inp. Each species has T ranges with 9 coefficients
per range (NASA-9, not NASA-7).

Equations (per mole):
  Cp/R = a1/T² + a2/T + a3 + a4·T + a5·T² + a6·T³ + a7·T⁴
  H/RT = -a1/T² + a2·ln(T)/T + a3 + a4·T/2 + a5·T²/3 + a6·T³/4 + a7·T⁴/5 + b1/T
  S/R  = -a1/(2T²) - a2/T + a3·ln T + a4·T + a5·T²/2 + a6·T³/3 + a7·T⁴/4 + b2

H is absolute on the formation scale, as in Burcat. At a boundary between two ranges of the same
species the lower range is used (Cantera uses the upper one; the data jump slightly there in
37 species). Away from the boundaries the reader matches Cantera to 1e-9 relative.

The phase comes from a separate file (cea_nasa9.yaml.phase): each line is name TAB 'gas' or
'condensed'. If that file is missing, or a species is absent from it, it is an error: assuming
'gas' would misclassify the 780 condensed species.

YAML 1.1 (PyYAML) reads NO, ON, OFF, YES... as booleans: the species NO (nitric oxide) would
arrive as False and break the lookup. The file is read without resolving booleans (Cantera uses
YAML 1.2 and does not have this problem).

Condensed phases are separate species by name (TiO2(cr), TiO2(L)); split ranges carry a suffix
(Fe(a)_2). chain() builds the chain by composition from the block stable at 298.15 K and only
follows blocks that touch, like the Burcat chain.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import yaml

from .burcat import BurcatRangeError, parse_formula

R_GAS = 8.31446261815324
T_REF = 298.15
REF = "NASA CEA thermo.inp (McBride, Zehe and Gordon 2002)"
CHAIN_TOL = 0.5                              # K, as in thermo_sources.py


class CEARangeError(BurcatRangeError):
    """T outside the block's ranges with strict=True (subclass of the Burcat error, so that code
    which already catches BurcatRangeError catches this one too)."""


class _NoBoolLoader(getattr(yaml, "CSafeLoader", yaml.SafeLoader)):
    pass


_NoBoolLoader.yaml_implicit_resolvers = {
    k: [r for r in v if r[0] != "tag:yaml.org,2002:bool"]
    for k, v in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


def _norm(sym: str) -> str:
    return sym if len(sym) == 1 else sym[0].upper() + sym[1:].lower()


def _key(elements: dict) -> tuple:
    return tuple(sorted((k, round(float(v), 6)) for k, v in elements.items()))


@dataclass(eq=False)          # identity equality: ndarray fields break == and `in`
class CEABlock:
    name: str
    phase: str                     # 'G' or 'C'
    elements: dict                 # without electrons
    charge: float
    T_ranges: np.ndarray           # n_ranges + 1 increasing temperatures
    coeffs: np.ndarray             # (n_ranges, 9)
    source_line: int               # line of '- name:' in the YAML
    ref: str = REF

    @property
    def T_low(self) -> float:
        return float(self.T_ranges[0])

    @property
    def T_high(self) -> float:
        return float(self.T_ranges[-1])

    def contains(self, T: float) -> bool:
        return self.T_low - 1e-6 <= T <= self.T_high + 1e-6

    def _pick_range(self, T: float, strict: bool = True) -> np.ndarray:
        if strict and not self.contains(T):
            raise CEARangeError(f"{self.name}: T = {T:.2f} K outside [{self.T_low}, {self.T_high}]")
        i = int(np.searchsorted(self.T_ranges, T, side="left")) - 1   # at a boundary, the lower range
        return self.coeffs[min(max(i, 0), len(self.coeffs) - 1)]

    def cp(self, T: float, strict: bool = True) -> float:
        """Cp in J/(mol·K)."""
        a1, a2, a3, a4, a5, a6, a7 = self._pick_range(T, strict)[:7]
        return R_GAS * (a1/T**2 + a2/T + a3 + a4*T + a5*T**2 + a6*T**3 + a7*T**4)

    def H(self, T: float, strict: bool = True) -> float:
        """H°(T) in J/mol, formation scale."""
        a1, a2, a3, a4, a5, a6, a7, b1, b2 = self._pick_range(T, strict)
        return R_GAS * T * (-a1/T**2 + a2*np.log(T)/T + a3 + a4*T/2
                            + a5*T**2/3 + a6*T**3/4 + a7*T**4/5 + b1/T)

    def S(self, T: float, strict: bool = True) -> float:
        """S°(T) in J/(mol·K)."""
        a1, a2, a3, a4, a5, a6, a7, b1, b2 = self._pick_range(T, strict)
        return R_GAS * (-a1/(2*T**2) - a2/T + a3*np.log(T)
                        + a4*T + a5*T**2/2 + a6*T**3/3 + a7*T**4/4 + b2)

    def G(self, T: float, strict: bool = True) -> float:
        return self.H(T, strict) - T * self.S(T, strict)

    @property
    def dHf_298(self) -> Optional[float]:
        """dHf(298.15) in J/mol if the block contains 298.15 K (the YAML does not store it separately)."""
        return self.H(T_REF) if self.contains(T_REF) else None


class CEADatabase:
    def __init__(self, yaml_path: Path, phase_path: Optional[Path] = None):
        self.path = Path(yaml_path)
        self.phase_path = Path(phase_path) if phase_path else self.path.with_name(self.path.name + ".phase")
        if not self.phase_path.exists():
            raise FileNotFoundError(f"Missing {self.phase_path}: without it gas and condensed cannot be told apart")
        self._phases: dict[str, str] = {}
        for line in self.phase_path.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) == 2:
                self._phases[parts[0].strip()] = parts[1].strip()
        self.blocks: list[CEABlock] = []
        self._parse()
        self._by_key = defaultdict(list)
        for b in self.blocks:
            self._by_key[(_key(b.elements), b.charge)].append(b)

    def _parse(self):
        text = self.path.read_text(encoding="utf-8")
        where = {m.group(1).strip(): text.count("\n", 0, m.start()) + 1
                 for m in re.finditer(r"^- name: (.+)$", text, flags=re.M)}
        data = yaml.load(text, Loader=_NoBoolLoader)
        species = data if isinstance(data, list) else data.get("species", [])
        for sp in species:
            name, th = str(sp["name"]), sp.get("thermo", {})
            if th.get("model") != "NASA9":
                raise ValueError(f"{name}: model {th.get('model')!r}, expected NASA9")
            phase = self._phases.get(name)
            if phase not in ("gas", "condensed"):
                raise ValueError(f"{name}: absent (or not gas/condensed) in {self.phase_path.name}")
            T = np.array(th["temperature-ranges"], dtype=float)
            C = np.array(th["data"], dtype=float)
            if C.ndim != 2 or C.shape != (len(T) - 1, 9) or np.any(np.diff(T) <= 0):
                raise ValueError(f"{name}: {len(T)} temperatures and data of shape {C.shape}")
            comp = {_norm(k): float(v) for k, v in (sp.get("composition") or {}).items()}
            charge = -comp.pop("E", 0.0)
            self.blocks.append(CEABlock(name=name, phase="G" if phase == "gas" else "C",
                                        elements=comp, charge=charge, T_ranges=T, coeffs=C,
                                        source_line=where.get(name, 0)))

    def get(self, name: str) -> list[CEABlock]:
        return [b for b in self.blocks if b.name == name]

    def segments(self, name: str) -> list[CEABlock]:
        """The segments of one phase by name (Fe(a), Fe(a)_2), ordered by T_low."""
        base = re.sub(r"_\d+$", "", name)
        return sorted([b for b in self.blocks if re.sub(r"_\d+$", "", b.name) == base],
                      key=lambda b: b.T_low)

    @staticmethod
    def _elements(formula: Union[str, dict]) -> Optional[dict]:
        return parse_formula(formula) if isinstance(formula, str) else formula

    def get_by_formula(self, formula: Union[str, dict], phase: Optional[str] = None,
                       charge: float = 0.0) -> list[CEABlock]:
        """Blocks by exact composition. phase: 'gas', 'condensed' or None."""
        el = self._elements(formula)
        if el is None:
            return []
        out = self._by_key.get((_key(el), charge), [])
        if phase is not None:
            want = {"gas": "G", "condensed": "C"}[phase]
            out = [b for b in out if b.phase == want]
        return sorted(out, key=lambda b: (b.T_low, b.source_line))

    def pick(self, formula: Union[str, dict], T: float, phase: Optional[str] = None,
             charge: float = 0.0) -> Optional[CEABlock]:
        """The block with the lowest G(T) among those containing T: the stable form at 1 bar."""
        cands = [b for b in self.get_by_formula(formula, phase, charge) if b.contains(T)]
        return min(cands, key=lambda b: b.G(T)) if cands else None

    def chain(self, formula: Union[str, dict], phase: str, charge: float = 0.0,
              tol: float = CHAIN_TOL) -> list[CEABlock]:
        """Chain from the block stable at 298.15 K, following only blocks that touch
        (if several meet at a boundary, the one with the lowest G there)."""
        blocks = self.get_by_formula(formula, phase, charge)
        start = self.pick(formula, T_REF, phase, charge)
        if start is None:
            return []
        chain = [start]
        while True:
            T = chain[-1].T_high
            nxt = [b for b in blocks if all(b is not c for c in chain) and abs(b.T_low - T) < tol]
            if not nxt:
                break
            chain.append(min(nxt, key=lambda b: b.G(b.T_low)))
        while True:
            T = chain[0].T_low
            prv = [b for b in blocks if all(b is not c for c in chain) and abs(b.T_high - T) < tol]
            if not prv:
                break
            chain.insert(0, min(prv, key=lambda b: b.G(b.T_high)))
        return chain
