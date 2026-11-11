# temp: NOT FINAL CODE

One-off scripts and notes from the KAE editing sessions (Nov 2026), copied as they were from a temporary working folder.
They are experiments, not the product. Final code will be rewritten into `scripts/` or `koopmann/`.

- Paths are hard-coded for the HPC (`/home/nsa325/work/koopmann`, `/scratch/nsa325/datasets`).
- Scripts expect `temp/venv` (a link to the repo's `.venv`) and run from inside `temp/`.
- Only code and notes (`.py`, `.sh`, `.md`) are tracked. Data, figures, logs and checkpoints live next to them on the
  HPC but are git-ignored (see `temp/.gitignore`).
  Home has little space, so on the HPC they sit in `/scratch/nsa325/koopmann_data/temp` and are linked into `temp/`.
- The main training scripts: `control_train.py` (patch-token control), `cls_train.py` (CLS-only state, the current plan).
  Test scripts: `angle_tests.py`, `where_control.py`. The review report builder: `review/`.
