"""SevenNet-MF-ompa (task mpa) worker: relaxations, single points and phonons through ASE, by file bridge.

The worker runs in whatever environment has sevenn installed (sevenn 0.13.0 with torch was used; a plain
CPU torch build works, a GPU one is faster). It reads an extended-XYZ file in which every frame carries an
`id` in its info line, and writes a JSON file of results beside it. Nothing else is read or written.
Resumable: rows whose id is already in the output file are skipped, so an interrupted run is continued
by calling the same command again. Every mode writes its JSON through save(): whole file to a sibling,
then rename, so an interrupted dump never truncates the file it resumes from. The device comes from the
environment variable MLIP_DEVICE ("cpu" by default, "cuda" on a GPU machine).

Modes
  relax    for every structure: relax cell and positions with ASE FIRE under a FrechetCellFilter, fmax
           0.1 eV/A, at most 300 steps; record the raw SevenNet energy per atom of the final structure,
           the energy per atom and the largest force at the starting structure (so the gain of the
           relaxation is logged without a second run), the steps, the largest final force, the seconds and
           the final cell, positions and numbers.
  single   for every structure: one energy and force evaluation at the structure as given, no relaxation;
           records the raw energy per atom, the largest atomic force and the seconds.
  phonons  for every structure: a tight relaxation (fmax 0.005 eV/A, at most 1500 steps), then
           ase.phonons.Phonons with a supercell of ceil(8 A / a_i) repeats per axis, displacement 0.01 A,
           acoustic sum rule, frequencies on the eight q-points of GRID; record the lowest frequency over
           the grid and at Gamma, in THz (a negative value flags a dynamical instability).

Run   python sevennet_worker.py relax   <input.extxyz> <output.json>
      python sevennet_worker.py single  <input.extxyz> <output.json>
      python sevennet_worker.py phonons <input.extxyz> <output.json> <work_dir>
"""
import json
import os
import sys
import time

import numpy as np
from ase.filters import FrechetCellFilter
from ase.io import read
from ase.optimize import FIRE

THZ_PER_EV = 1 / 4.135667e-3
GRID = np.array([(0, 0, 0), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5),
                 (0.5, 0.5, 0.5)])


DEVICE = os.environ.get("MLIP_DEVICE", "cpu")  # "cuda" on a GPU machine or notebook


def calculator():
    from sevenn.calculator import SevenNetCalculator
    return SevenNetCalculator("7net-mf-ompa", modal="mpa", device=DEVICE)


def relax(atoms, calc, fmax, steps):
    atoms.calc = calc
    opt = FIRE(FrechetCellFilter(atoms), logfile=None)
    opt.run(fmax=fmax, steps=steps)
    f = np.linalg.norm(atoms.get_forces(), axis=1).max()
    return float(f), int(opt.get_number_of_steps())


def load(path):
    """The resumable output: existing rows kept; the header (model, versions) filled on both paths, so a file pre-seeded
    by another script still records the environment of the first worker run that touched it."""
    import sevenn
    import torch
    out = json.load(open(path)) if os.path.exists(path) else dict(rows=[])
    for k, v in dict(model="SevenNet-MF-ompa", modal="mpa", sevenn_version=sevenn.__version__, torch_version=torch.__version__,
                     device=DEVICE).items():
        out.setdefault(k, v)
    out.setdefault("rows", [])
    # a file started on one device and continued on another records both, so the mixed origin of the rows is never hidden
    out.setdefault("devices_used", [])
    if DEVICE not in out["devices_used"]:
        out["devices_used"].append(DEVICE)
    return out


def save(out, path):
    """Write the whole file to a sibling and rename: an interrupted dump never truncates the resume file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f)
    os.replace(tmp, path)


def run_relax(inp, out_path):
    out = load(out_path)
    out.update(fmax=0.1, steps=300, optimizer="FIRE + FrechetCellFilter")
    done = {r["id"] for r in out["rows"]}
    todo = [a for a in read(inp, index=":") if a.info["id"] not in done]
    print(f"relax: {len(done)} done, {len(todo)} to do", flush=True)
    calc = calculator()
    t0 = time.time()
    for n, a in enumerate(todo, start=1):
        t1 = time.time()
        try:
            a.calc = calc
            e_start = float(a.get_potential_energy()) / len(a)  # the energy at the starting structure, before any step
            fmax_start = float(np.linalg.norm(a.get_forces(), axis=1).max())  # the largest force there, same evaluation
            fm, st = relax(a, calc, 0.1, 300)
            rec = dict(id=a.info["id"], energy_per_atom=float(a.get_potential_energy()) / len(a), energy_start_per_atom=e_start,
                       fmax_start=fmax_start, steps=st, fmax_final=fm, converged=bool(fm <= 0.1), seconds=time.time() - t1,
                       cell=a.cell.array.tolist(), positions=a.positions.tolist(), numbers=a.numbers.tolist())
        except Exception as e:  # noqa: BLE001
            rec = dict(id=a.info["id"], error=str(e)[:200], seconds=time.time() - t1)
        out["rows"].append(rec)
        if n % 5 == 0 or n == len(todo):
            save(out, out_path)
            el = time.time() - t0
            print(f"  {n}/{len(todo)} ({el:.0f} s, {el / n:.1f} s each, about {el / n * (len(todo) - n) / 60:.0f} min left)", flush=True)
    save(out, out_path)


def run_single(inp, out_path):
    """One energy and force evaluation per structure, as given; no relaxation."""
    out = load(out_path)
    out.update(mode="single point at the structure as given, no relaxation")
    done = {r["id"] for r in out["rows"]}
    todo = [a for a in read(inp, index=":") if a.info["id"] not in done]
    print(f"single: {len(done)} done, {len(todo)} to do", flush=True)
    calc = calculator()
    t0 = time.time()
    for n, a in enumerate(todo, start=1):
        t1 = time.time()
        a.calc = calc
        try:
            rec = dict(id=a.info["id"], n_atoms=len(a), energy_per_atom=float(a.get_potential_energy()) / len(a),
                       fmax_start=float(np.linalg.norm(a.get_forces(), axis=1).max()), seconds=time.time() - t1)
        except Exception as e:  # noqa: BLE001
            rec = dict(id=a.info["id"], error=str(e)[:200], seconds=time.time() - t1)
        out["rows"].append(rec)
        if n % 10 == 0 or n == len(todo):
            save(out, out_path)
            el = time.time() - t0
            print(f"  {n}/{len(todo)} ({el:.0f} s, {el / n:.1f} s each, about {el / n * (len(todo) - n) / 60:.0f} min left)", flush=True)
    save(out, out_path)


def run_phonons(inp, out_path, work):
    from ase.phonons import Phonons
    out = load(out_path)
    out.update(fmax=0.005, steps=1500, delta=0.01, supercell_rule="ceil(8 A / a_i)")
    done = {r["id"] for r in out["rows"]}
    todo = [a for a in read(inp, index=":") if a.info["id"] not in done]
    print(f"phonons: {len(done)} done, {len(todo)} to do", flush=True)
    calc = calculator()
    for a in todo:
        t1 = time.time()
        tag = a.info["id"].replace("->", "_to_").replace("(", "").replace(")", "")
        try:
            fres, st = relax(a, calc, 0.005, 1500)
            n = [max(1, int(np.ceil(8.0 / L))) for L in a.cell.lengths()]
            wd = os.path.join(work, tag)
            os.makedirs(wd, exist_ok=True)
            ph = Phonons(a, calc, supercell=tuple(n), delta=0.01, name=os.path.join(wd, "ph"))
            ph.run()
            ph.read(acoustic=True)
            w = np.real(ph.band_structure(GRID, verbose=False)) * THZ_PER_EV
            rec = dict(id=a.info["id"], min_THz=float(w.min()), gamma_min_THz=float(w[0].min()), supercell="x".join(map(str, n)),
                       tight_residual_force=fres, tight_steps=st, seconds=time.time() - t1)
        except Exception as e:  # noqa: BLE001
            rec = dict(id=a.info["id"], error=str(e)[:200], seconds=time.time() - t1)
        out["rows"].append(rec)
        save(out, out_path)
        print(f"  {a.info['id']}: {rec.get('min_THz', rec.get('error'))} ({rec['seconds']:.0f} s)", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "relax":
        run_relax(sys.argv[2], sys.argv[3])
    elif mode == "single":
        run_single(sys.argv[2], sys.argv[3])
    elif mode == "phonons":
        run_phonons(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        raise SystemExit("mode must be relax, single or phonons")
