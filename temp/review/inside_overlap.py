"""
Question f: do the kept modes identify the class? The test of arXiv 2608.29057, with no fitted probe.

For 5,000 random validation images and each KAE:
  overlap of two images = modes both keep / modes either keeps
  1. the mean overlap of two images of the same class, against two images of different classes
  2. the share of images whose nearest image by overlap has the same class (ties share the credit)
For reference, the nearest image by cosine of real CLIP's state after block 6 and after block 12 (CLS token, and whole state).

Writes figs/inside_overlap.png and data/inside_overlap.json.
"""

import json

from common import *
from demo_utils import load_val
from inside_load import KAES, load_z6

torch.manual_seed(0)
states, labels = load_val()
pick = torch.randperm(len(labels))[:5_000]
same = labels[pick][:, None] == labels[pick][None]
other = ~torch.eye(len(pick), dtype=torch.bool)


def nearest_same(alike):
    """Share of images whose most alike other image has the same class. Images tied for most alike share the credit."""

    alike = alike.masked_fill(~other, -1)
    tied = alike == alike.max(dim=1, keepdim=True).values
    return ((tied & same).sum(dim=1) / tied.sum(dim=1)).mean().item()


table = {"chance": ((same & other).sum(dim=1) / other.sum(dim=1)).float().mean().item()}
for block, width, key in [(0, 768, "real CLIP, CLS token after block 6"), (1, 768, "real CLIP, CLS token after block 12"), (0, 2304, "real CLIP, whole state after block 6"), (1, 2304, "real CLIP, whole state after block 12")]:  # fmt: skip
    tokens = states[pick, block, :width] - states[:, block, :width].mean(dim=0)
    tokens = tokens / tokens.norm(dim=1, keepdim=True)
    table[key] = nearest_same(tokens @ tokens.T)
for name in KAES:
    kept = (load_z6(name)[pick] != 0).float()
    shared = kept @ kept.T
    overlap = shared / (kept.sum(dim=1)[:, None] + kept.sum(dim=1)[None] - shared)
    table[name] = {"label": MODELS[name], "same class": overlap[same & other].mean().item(), "different class": overlap[~same].mean().item(), "nearest image has the same class": nearest_same(overlap)}
    print(name, table[name], flush=True)
print({key: value for key, value in table.items() if not isinstance(value, dict)})
(DATA / "inside_overlap.json").write_text(json.dumps(table, indent=1, ensure_ascii=False))

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
rows = torch.arange(len(KAES))
for offset, key, color in [(-0.19, "same class", BLUE), (0.19, "different class", ORANGE)]:
    axes[0].barh(rows + offset, [table[name][key] for name in KAES], height=0.36, color=color, label=f"two images of the {key}" if key == "same class" else "two images of different classes")
axes[0].set_xlabel("overlap of the kept modes (shared / used by either)")
axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9, handlelength=1.2, labelcolor=INK)
axes[1].barh(rows, [100 * table[name]["nearest image has the same class"] for name in KAES], height=0.5, color=BLUE)
for key, color, label in [("real CLIP, CLS token after block 6", MUTED, "real CLIP\nafter block 6"), ("real CLIP, CLS token after block 12", INK, "real CLIP\nafter block 12")]:
    axes[1].axvline(100 * table[key], color=color, lw=1.2)
    axes[1].text(100 * table[key], -0.75, label, color=INK, fontsize=8, ha="center", va="bottom")
axes[1].set_xlabel("images whose nearest image has the same class (%)")
axes[0].set_yticks(rows, [MODELS[name] for name in KAES])
axes[0].invert_yaxis()
axes[0].tick_params(axis="y", colors=INK, length=0)
for ax in axes:
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
fig.suptitle("Which modes an image keeps says little about its class, and almost nothing once the model has a control", x=0.01, ha="left", color=INK, fontsize=12)
fig.tight_layout()
save(fig, "inside_overlap")
