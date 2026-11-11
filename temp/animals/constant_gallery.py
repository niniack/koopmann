"""
Photos for the modes real CLIP holds constant (from constant_val.py): per mode, the 8 validation photos where it is
strongest and 8 where it is off or weakest. Photos of hidden classes are never shown (demo_utils.get_hidden).

Usage: constant_gallery.py <KAE run name>
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, "/home/nsa325/work/koopmann/second_analysis")
import demo_utils  # noqa: E402

HERE = Path(__file__).parent
name = sys.argv[1]
saved = torch.load(HERE / f"constant_val_{name}.pt")
states, labels = demo_utils.load_val()
shown = ~demo_utils.get_hidden(demo_utils.Kae("im_iso_2x4k"), states, labels)
images = demo_utils.load_images()
strength = saved["z6"].abs()
strength[~shown] = float("nan")

fig, axes = plt.subplots(len(saved["modes"]), 17, figsize=(17 * 1.3, len(saved["modes"]) * 1.45))
for row, mode in enumerate(saved["modes"].tolist()):
    order = strength[:, row].nan_to_num(-1).argsort(descending=True)
    weakest = strength[:, row].nan_to_num(float("inf")).argsort()[:8]
    for col, idx in enumerate([*order[:8].tolist(), None, *weakest.tolist()]):
        ax = axes[row, col]
        ax.axis("off")
        if idx is None:
            ax.text(0.5, 0.5, f"mode {mode + 1}\nstrongest | weakest", ha="center", va="center", fontsize=7, color=demo_utils.INK)
            continue
        picture = images[idx]["image"].convert("RGB")
        picture.thumbnail((160, 160))
        ax.imshow(np.asarray(picture))
        ax.set_title(demo_utils.CLASS_NAMES[labels[idx]][:18], fontsize=6, color=demo_utils.INK, pad=1)
fig.tight_layout()
fig.savefig(HERE / f"constant_gallery_{name}.png", dpi=110)
print("saved")
