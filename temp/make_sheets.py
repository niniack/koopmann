"""Lays the painted pictures out in two sheets (mode edits, colour edits) and prints the numbers behind them."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import open_clip
import pandas as pd
import torch
from PIL import Image

out = Path(sys.argv[1] if len(sys.argv) > 1 else "inv")
INK, MUTED = "#52514e", "#898781"
names = open_clip.IMAGENET_CLASSNAMES
results = torch.load(out / "results.pt", weights_only=False)
notes = torch.load("inv/notes.pt", weights_only=False)
MODE_COLUMNS = {"photo": "photo", "real": "DINOv3's block 12", "recon": "KAE, block 12\nin and out", "pred": "KAE prediction\nfrom block 6", "same": "prediction,\ncontinued", "removed": "mode removed", "x2": "mode doubled", "+90": "mode turned +90°", "-90": "mode turned -90°", "control": "another mode\nremoved"}
# Classes the user must never be shown (see the galleries); a painted picture whose reading or whose target state
# has one of them among its three best answers is left out of the sheet
HIDDEN_CLASSES = {*range(69, 80), 110, 111, *range(113, 117), *range(118, 127), *range(300, 329), 410, 599, 815}
is_hidden = lambda result: bool(HIDDEN_CLASSES & {*result["answers"][0], *result["target answers"][0]})
COLOUR_COLUMNS = {"photo": "photo", "real": "DINOv3's block 12", "real gray": "DINOv3 on the\ngray copy", "real sepia": "DINOv3 on the\nsepia copy", "pred": "KAE prediction\nfrom block 6", "pred of gray copy": "KAE prediction\nfor the gray copy", "knob gray": "KAE, gray knob", "knob sepia": "KAE, sepia knob"}


def show(columns, indices, path, colour=False):
    fig, axes = plt.subplots(len(indices), len(columns), figsize=(2.05 * len(columns), 2.5 * len(indices)), squeeze=False)
    for row, index in zip(axes, indices):
        for ax, (kind, title) in zip(row, columns.items()):
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_visible(False)
            if ax in axes[0]:
                ax.set_title(title, fontsize=8.5, color=INK)
            if kind == "photo":
                ax.imshow(Image.open(f"inv/photo_{index}.png"))
                note = notes[index]
                ax.set_xlabel(f"mode {note['mode']}: {note['meaning']}", fontsize=7.5, color=INK)
                continue
            result = results.get(f"{index}_{kind}")
            if result is None:
                ax.axis("off")
                continue
            if is_hidden(result):
                ax.set_facecolor("#eeeeec")
                ax.set_xlabel("left out\n(reads as a class on the no-show list)", fontsize=7, color=MUTED)
                continue
            ax.imshow(result["picture"].permute(1, 2, 0).numpy())
            read = f"{names[result['answers'][0][0]][:20]} {result['answers'][1][0]:.2f}"
            asked = f"{names[result['target answers'][0][0]][:20]}"
            label = f"reads as: {read}\nstate says: {asked}"
            if colour:
                label += f"\ncolour {result['colourfulness']:.3f}"
            ax.set_xlabel(label, fontsize=7, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)


indices = list(notes)
show(MODE_COLUMNS, indices[:5], out / "sheet_modes_1.png")
show(MODE_COLUMNS, indices[5:], out / "sheet_modes_2.png")
coloured = [i for i in indices if notes[i]["coloured"]]
show(COLOUR_COLUMNS, coloured, out / "sheet_colour.png", colour=True)

rows = []
for name, result in results.items():
    rows.append({
        "image": result["index"], "kind": result["kind"], "steps": result["steps"], "state error left": result["error"],
        "reads as": names[result["answers"][0][0]], "p": result["answers"][1][0], "state says": names[result["target answers"][0][0]],
        "agree": result["answers"][0][0] == result["target answers"][0][0], "colour": result["colourfulness"],
    })
table = pd.DataFrame(rows)
pd.set_option("display.width", 250, "display.max_rows", 500)
print(table.round(3).to_string(index=False))
print(table.groupby("kind")[["state error left", "agree", "colour"]].mean().round(3))
table.to_csv(out / "numbers.csv", index=False)
