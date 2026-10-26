"""Accessible oxidation states of an element, from the ICSD bond-valence statistics shipped with pymatgen.

An oxidation state counts as accessible when it occurs in at least MIN_ICSD crystal structures of those
statistics; an element with no statistics at all falls back to pymatgen's textbook common states.
"""
from __future__ import annotations

from pymatgen.core import Element, Species

MIN_ICSD = 20                      # an oxidation state is accessible if it occurs in at least this many ICSD structures


def icsd_count(el: str, ox: int) -> int:
    from pymatgen.analysis.bond_valence import ICSD_BV_DATA
    try:
        d = ICSD_BV_DATA.get(Species(el, int(ox)))
        return int(d["n_data_pts"]) if d else 0
    except Exception:
        return 0


def has_icsd_statistics(el: str) -> bool:
    return any(icsd_count(el, ox) > 0 for ox in range(-4, 9) if ox)


def accessible_states(el: str):
    """(positive oxidation states an element can take, source). ICSD-frequent states; textbook fallback."""
    if has_icsd_statistics(el):
        return [ox for ox in range(1, 9) if icsd_count(el, ox) >= MIN_ICSD], "ICSD (>= %d structures)" % MIN_ICSD
    return [int(ox) for ox in Element(el).common_oxidation_states if ox > 0], "textbook common states (no ICSD statistics)"
