"""Dumps an executed notebook's outputs: text to stdout, images to <out dir>/cell<i>_<j>.png. Usage: dump_notebook.py <notebook> <out dir>"""
import base64
import sys
from pathlib import Path

import nbformat

nb = nbformat.read(sys.argv[1], as_version=4)
out = Path(sys.argv[2])
out.mkdir(exist_ok=True)
for i, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    print(f"\n===== cell {i}: {cell.source.splitlines()[0][:80]}")
    for j, output in enumerate(cell.outputs):
        if output.output_type == "stream":
            print(output.text.rstrip())
        elif output.output_type == "error":
            print("ERROR", output.ename, output.evalue)
        else:
            data = output.get("data", {})
            if "image/png" in data:
                path = out / f"cell{i}_{j}.png"
                path.write_bytes(base64.b64decode(data["image/png"]))
                print(f"[image {path}]")
            elif "text/plain" in data:
                print(data["text/plain"])
