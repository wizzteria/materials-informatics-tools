"""Write an example input for sevennet_worker.py: three small public structures built in code (no
downloads), one frame each, with the `id` the worker keys its results on.

Run:  python mlip_worker/make_example_input.py [input.extxyz]
"""
import sys

from ase import Atoms
from ase.io import write


def example_atoms():
    nacl = Atoms("NaCl", scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]], pbc=True,
                 cell=[[0, 2.82, 2.82], [2.82, 0, 2.82], [2.82, 2.82, 0]])
    si = Atoms("Si2", scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]], pbc=True,
               cell=[[0, 2.715, 2.715], [2.715, 0, 2.715], [2.715, 2.715, 0]])
    mgo = Atoms("MgO", scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]], pbc=True,
                cell=[[0, 2.105, 2.105], [2.105, 0, 2.105], [2.105, 2.105, 0]])
    for a, name in zip((nacl, si, mgo), ("NaCl_rocksalt", "Si_diamond", "MgO_rocksalt")):
        a.info["id"] = name
    return [nacl, si, mgo]


def main(path="input.extxyz"):
    frames = example_atoms()
    write(path, frames, format="extxyz")
    print(f"wrote {len(frames)} structures to {path}: {[a.info['id'] for a in frames]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "input.extxyz")
