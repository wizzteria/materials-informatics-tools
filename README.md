# materials-informatics-tools

Small, tested tools I wrote for machine-learned screening of inorganic crystals. Each folder is a self-contained
package with its own README, a short example and unit tests. Everything here runs on public data (Materials Project,
Matbench, WBM, NBS and NIST thermochemical tables); the screening results of the research project these tools came
from are not part of this repository.

| Folder | What it does |
|---|---|
| `crystal_graph/` | Crystal structure to graph (numpy), and a CGCNN-style graph network written in plain PyTorch, no graph library. |
| `edit_distance/` | A canonical prototype key for a crystal and the edit distance from a crystal to the nearest one a model was trained on, so that model error can be measured as a function of how far a new crystal is from the training set. |
| `mlip_worker/` | A file-bridge worker that relaxes structures, computes single points or phonons with a universal interatomic potential (SevenNet) through ASE, with resumable JSON output, plus a Kaggle notebook that runs it on a free GPU. |
| `safety_gate/` | A hazard-label gate for computational screening: every element rule carries its cause and a public regulatory source, labels are attached and never used to delete rows, and the element rules are validated against the EU CLP Annex VI harmonised classifications. |
| `thermo/` | Readers for four public thermochemical sources (Burcat, NASA CEA, NIST Shomate, NBS Tables) behind one layer that records the provenance of every number and refuses to extrapolate a species outside its fitted temperature range. |
| `reading_copy_check/` | A check that a "reading copy" of a script (comments and docstrings rewritten for readers) has exactly the program logic of the original, by comparing the two syntax trees with prose blanked. |

## Working rules these tools enforce

- A design is written and frozen before its result is seen; thresholds and bins live in the code as constants and are
  not edited after a run.
- No thermochemical value is typed by hand: every number is read from its source with the source recorded beside it.
- One species, one temperature range: a reader refuses a species outside the range its polynomial was fitted on.
- Flags, not deletions: a hazard or a model-disagreement label travels with the row; the person reading the list decides.

## Setup

Python 3.11. The thermochemistry and graph tools need only numpy, pandas, scipy and pymatgen; the graph network needs
PyTorch; the worker needs ASE and `sevenn` (installed in the notebook, not here).

```
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m unittest discover -s tests
```

Tests that need an external database (Burcat, NASA CEA, the ECHA Annex VI workbook) skip with a message when the file
is not present; each package README says where to download it.

## Licence

MIT, see `LICENSE`. The NBS table in `data/` is a United States government work in the public domain, included
unchanged. The small NIST Shomate file holds a few published coefficient sets transcribed from the NIST WebBook, with
the source of each row named in the file and in `thermo/README.md`. The Burcat, NASA CEA and ECHA files are not
redistributed: each package README gives the download page and the environment variable that points to the file.
