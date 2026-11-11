"""
Question g: do modes reproduce across seeds? Draws the saved results of seed_match.py (nothing is recomputed):
two KAEs with the same settings and different seeds, their modes paired by how alike they fire on validation.
Pairs shown: each mode is the other's best partner ("mutual best").

Writes figs/inside_seeds.png and data/inside_seeds.json.
"""

import json

from common import *
from demo_utils import rank_correlation

# The two seed pairs, and the median phase relation of random pairs (from the logs of seed_match.py)
PAIRS = {
    "im_iso_2x4k_im_iso_2x4k_s1": ("KAE, no control (older state)", 0.156, "seed_match_s0_s1.log"),
    "im_realstart_2x4k_wd_im_realstart_2x4k_wd_s1": ("KAE, real start and weight decay (older state)", 0.152, "seed_match_realstart_wd.log"),
}


def scatter(ax, values, title, unit, log=False):
    ax.scatter(values[0], values[1], s=12, color=BLUE, alpha=0.6, lw=0)
    low, high = values.min().item(), values.max().item()
    ax.plot([low, high], [low, high], color=MUTED, lw=0.8)
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _: f"{value:g}"))
    ax.set_xlabel(f"{unit}, seed 0")
    ax.set_ylabel(f"{unit}, seed 1")
    ax.set_title(title, fontsize=10)


table = {}
fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.2))
for row, (key, (label, random_phase, log_name)) in zip(axes, PAIRS.items()):
    match = torch.load(SCRATCH / f"seed_match_{key}.pt")
    pick = match["mutual"]
    size, angle = match["size"][:, pick], match["angle"][:, pick].abs()
    phase = match["ties"][pick, :2].max(dim=1).values
    phase = phase[phase == phase]  # pairs with too few shared images have no phase relation
    numbers = {"pairs": int(pick.sum()), "size of λ": rank_correlation(size[0], size[1]), "angle of λ": rank_correlation(angle[0], angle[1]),
               "phase relation, median": phase.quantile(0.5).item(), "phase relation of random pairs, median": random_phase, "source": log_name}  # fmt: skip
    table[label] = numbers

    scatter(row[0], size, f"Size of λ: rank correlation {numbers['size of λ']:+.2f}", "size of λ", log=True)
    scatter(row[1], angle, f"Angle of λ: rank correlation {numbers['angle of λ']:+.2f}", "angle of λ (degrees)")
    row[2].hist(phase, bins=torch.linspace(0, 1, 21), color=BLUE, edgecolor="white", lw=1)
    row[2].axvline(random_phase, color=MUTED, lw=1.2)
    row[2].text(random_phase + 0.02, 0.97, "random pairs\n(median)", color=INK, fontsize=8, va="top", transform=row[2].get_xaxis_transform())
    row[2].set_xlabel("strength of the phase relation (0 to 1)")
    row[2].set_ylabel("matched modes")
    row[2].set_title(f"Phase of z: median {numbers['phase relation, median']:.2f}", fontsize=10)
    row[0].annotate(f"{label}\n{numbers['pairs']} matched modes", (0, 1.16), xycoords="axes fraction", color=INK, fontsize=10, fontweight="bold", va="bottom")

fig.suptitle("Across seeds the size of λ and the phase of z come back; the angle of λ only under the real-start rule", x=0.01, ha="left", color=INK, fontsize=12)
fig.tight_layout(h_pad=3)
save(fig, "inside_seeds")
(DATA / "inside_seeds.json").write_text(json.dumps(table, indent=1, ensure_ascii=False))
print(json.dumps(table, indent=1, ensure_ascii=False))
