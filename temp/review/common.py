"""
Shared by every script of the review: folders, the chart style and colours, and the rule for pictures that are never shown.

Import it first in a figure script:  from common import *
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

HERE = Path(__file__).parent  # .../scratchpad/review
SCRATCH = HERE.parent
DATA, FIGS, DIGEST, SECTIONS = HERE / "data", HERE / "figs", HERE / "digest", HERE / "sections"
sys.path.insert(0, str(SCRATCH))
sys.path.insert(0, "/home/nsa325/work/koopmann/second_analysis")
from demo_utils import BLUE, HIDDEN_CLASSES, HIDDEN_IMAGES, INK, MUTED, ORANGE  # noqa: E402  (also sets the plot style)

# More series colours, in this fixed order after BLUE and ORANGE. GOOD and BAD are only for right / wrong.
AQUA, YELLOW, MAGENTA, VIOLET = "#1baf7a", "#eda100", "#e87ba4", "#6250d6"
GOOD, BAD = "#0ca30c", "#d03b3b"
GRID, LIGHT_BLUE = "#e8e7e3", "#9ec5f4"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.facecolor": "white", "savefig.facecolor": "white"})

# The models of the review: file name of the checkpoint, and the plain name every chart and table uses for it
MODELS = {
    "im_iso_2x4k": "KAE, no control (older state)",
    "im_mlpk_2x4k": "MLP in place of eigenvalues (older state)",
    "on_none": "KAE, no control",
    "on_add": "KAE, additive control",
    "on_bilinear": "KAE, bilinear control",
    "cls_add": "CLS-only KAE, additive control",
    "cls_add_real": "CLS-only KAE, real start",
    "cls_mlp": "CLS-only, MLP in place of eigenvalues",
    "share_real_w01": "CLS-only KAE, control penalty 0.1",
    "share_real_w03": "CLS-only KAE, control penalty 0.3",
    "share_real_w1": "CLS-only KAE, control penalty 1.0",
}


def save(fig, name):
    """Saves a figure to figs/<name>.png and closes it."""

    fig.savefig(FIGS / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def hidden_images(labels, top5):
    """Validation images no picture may show: a hidden class as the label or among real CLIP's 5 best answers."""

    hidden = torch.isin(labels, HIDDEN_CLASSES) | torch.isin(top5, HIDDEN_CLASSES).any(dim=1)
    hidden[HIDDEN_IMAGES] = True
    return hidden
