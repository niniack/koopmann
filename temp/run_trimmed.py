"""Executes the notebook's setup cells and the new cells only, to test the new cells before a full run."""
import sys

import nbformat
from nbclient import NotebookClient

source, out = sys.argv[1], sys.argv[2]
nb = nbformat.read(source, 4)
last_setup = max(i for i, c in enumerate(nb.cells) if "group = np.select" in c.source)
last_old = max(i for i, c in enumerate(nb.cells) if "by_angle.round(3)" in c.source)
nb.cells = nb.cells[: last_setup + 1] + nb.cells[last_old + 1 :]
print(len(nb.cells), "cells", flush=True)
try:
    NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": "/home/nsa325/work/koopmann/second_analysis"}}).execute()
finally:
    nbformat.write(nb, out)
