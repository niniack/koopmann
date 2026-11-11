"""One figure for the report: the strongest photos of a few modes real CLIP holds constant, cut from the two galleries."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).parent
INK = "#52514e"
# (gallery, row in the gallery, what the photos share)
ROWS = [
    ("on_none", 0, "KAE without control: big cats"),
    ("on_none", 3, "KAE without control: gymnastics"),
    ("on_none", 4, "KAE without control: patterned rugs and covers"),
    ("on_none", 7, "KAE without control: birds on bare branches"),
    ("on_none", 12, "KAE without control: printed text"),
    ("on_none", 15, "KAE without control: sailing ships"),
    ("on_add", 1, "KAE with additive control: snow or a plain pale background"),
    ("on_add", 2, "KAE with additive control: motion blur"),
    ("on_add", 9, "KAE with additive control: close-up with a blurred background"),
]

fig, axes = plt.subplots(len(ROWS), 1, figsize=(10, 1.45 * len(ROWS)))
for ax, (run, row, title) in zip(axes, ROWS):
    sheet = Image.open(HERE / f"constant_gallery_{run}.png")
    height = sheet.height / 16
    ax.imshow(sheet.crop((0, int(row * height), int(sheet.width * 0.47), int((row + 1) * height))))
    ax.set_title(title, loc="left", fontsize=9, color=INK, pad=2)
    ax.axis("off")
fig.tight_layout()
fig.savefig(HERE.parent / "report" / "constant.png", dpi=130)
