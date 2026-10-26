# mlip_worker

A small file-bridge worker around the SevenNet-MF-ompa machine-learned
interatomic potential (task `mpa`), driven through ASE. You hand it an
extended-XYZ file, it writes a JSON file of results beside it. The point is to
run the potential in its own environment (or on a rented GPU) without the rest
of a project having to import it.

## What it does

`sevennet_worker.py` has three modes, all resumable (rows whose `id` is already
in the JSON are skipped, so an interrupted run continues with the same command):

- `relax`: FIRE optimizer under a FrechetCellFilter (cell and positions),
  fmax 0.1 eV/A, at most 300 steps. Records the energy per atom before and after,
  the largest force before and after, the steps, and the final cell/positions.
- `single`: one energy and force evaluation at the structure as given.
- `phonons`: tight relaxation (fmax 0.005 eV/A), then `ase.phonons.Phonons` on a
  supercell of ceil(8 A / a_i) repeats per axis; records the lowest frequency on
  eight q-points and at Gamma, in THz.

The device comes from `MLIP_DEVICE` (`cpu` default, `cuda` on a GPU). The JSON
header records the sevenn and torch versions and every device the rows were
produced on. Writes go to a sibling file and are renamed, so a crash never
truncates the file being resumed.

## How to use it

```
pip install sevenn==0.13.0            # brings torch; use a CUDA build for a GPU
python mlip_worker/make_example_input.py input.extxyz
MLIP_DEVICE=cpu python mlip_worker/sevennet_worker.py relax input.extxyz output.json
python mlip_worker/sevennet_worker.py single  input.extxyz output.json
python mlip_worker/sevennet_worker.py phonons input.extxyz output.json phonon_work/
```

Every frame of the input needs an `id` key in its info line (`make_example_input.py`
shows how). `kaggle_gpu_notebook.ipynb` runs the same worker on a Kaggle GPU:
add the worker and `input.extxyz` as a dataset, enable the GPU and internet, run
the four cells, download `output.json`.

## Where the method comes from

SevenNet: Y. Park et al., "Scalable Parallel Algorithm for Graph Neural Network
Interatomic Potentials in Molecular Dynamics Simulations", J. Chem. Theory
Comput. 20, 4857 (2024); the multi-fidelity `7net-mf-ompa` checkpoint is described
in the SevenNet repository (github.com/MDIL-SNU/SevenNet). Relaxation and
phonon machinery: ASE (Larsen et al., J. Phys.: Condens. Matter 29, 273002, 2017).
