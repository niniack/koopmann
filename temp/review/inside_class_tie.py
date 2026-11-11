"""
Question e: does the size of λ follow the class?

The older finding (second_analysis/demo.ipynb, cell 15): over the modes used by 200+ validation images, the rank
correlation between the size of λ and the mode's class tie is +0.79. The class tie of a mode is the share of the
image-to-image variation of |z6| that the class label explains (demo_utils.class_tie). The same is computed here for
every KAE of the review, and for the angle of λ.

Writes figs/inside_class_tie.png and data/inside_class_tie.json.
"""

import json

from common import *
from demo_utils import class_tie, load_val, rank_correlation
from inside_load import KAES, get_modes, load_results, load_z6

labels = load_val()[1]
table = {}
for name in KAES:
    modes, z6 = get_modes(load_results(name)), load_z6(name)
    live = modes["usage"] >= 200
    size, angle, usage, tie = modes["size"][live], modes["angle"][live], modes["usage"][live].float(), class_tie(z6.abs(), labels)[live]
    table[name] = {
        "label": MODELS[name],
        "modes used by 200+ images": int(live.sum()),
        "class tie, median": tie.median().item(),
        "size of λ": rank_correlation(size, tie),
        "size of λ, usage held fixed": rank_correlation(size, tie, (usage,)),
        "angle of λ": rank_correlation(angle, tie),
        "angle of λ, size and usage held fixed": rank_correlation(angle, tie, (size, usage)),
        "usage": rank_correlation(usage, tie),
    }
    print(name, {key: round(value, 3) if isinstance(value, float) else value for key, value in table[name].items()}, flush=True)
(DATA / "inside_class_tie.json").write_text(json.dumps(table, indent=1, ensure_ascii=False))

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True, width_ratios=[1.6, 1])
rows = torch.arange(len(KAES))
for offset, key, color in [(-0.19, "size of λ", BLUE), (0.19, "angle of λ", ORANGE)]:
    axes[0].barh(rows + offset, [table[name][key] for name in KAES], height=0.36, color=color, label=key)
axes[0].axvline(0, color=INK, lw=0.8)
axes[0].set_xlim(-0.4, 1)
axes[0].set_xlabel("rank correlation with the mode's class tie\n(+1: the larger it is, the better the mode follows the class)")
axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9, handlelength=1.2, labelcolor=INK)
axes[1].barh(rows, [table[name]["class tie, median"] for name in KAES], height=0.5, color=AQUA)
axes[1].set_xlabel("class tie of the median mode\n(share of its variation the class explains)")
axes[0].set_yticks(rows, [MODELS[name] for name in KAES])
axes[0].invert_yaxis()
axes[0].tick_params(axis="y", colors=INK, length=0)
for ax in axes:
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
fig.suptitle("The size of λ follows the class in every KAE; the angle of λ does not. With a control the modes follow the class far less", x=0.01, ha="left", color=INK, fontsize=12)
fig.tight_layout()
save(fig, "inside_class_tie")
