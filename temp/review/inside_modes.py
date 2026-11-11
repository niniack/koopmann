"""
Questions b and c: the eigenvalues of each KAE, and how its modes are used.

b. One panel per KAE: each mode is a dot at (angle of λ, size of λ), larger when more images use the mode.
c. One panel per KAE: the share of images that use each mode, modes sorted from most to least used.
A mode is "in use" when 50 or more of the 50,000 validation images keep it. A mode is "constant" when the size of
its λ is within 0.05 of 1 and its angle is under 2 degrees. Mode 0, the single real mode, is left out.
The DINOv3 KAE's numbers need inside_dino.py to have run.

Writes figs/inside_eigenvalues.png, figs/inside_usage.png and data/inside_modes.json.
"""

import json

from common import *
from inside_load import EXTRA, KAES, NAMES, get_modes, load_latents, load_results

MAX_ANGLE = 40


def count(name):
    """The numbers of questions b and c for one KAE."""

    results = load_results(name)
    modes = get_modes(results)
    size, angle, usage = modes["size"], modes["angle"], modes["usage"]
    used = usage >= 50
    by_use = usage.sort(descending=True).values.float()
    return {
        "label": NAMES[name],
        "modes used by 50+ images": int(used.sum()),
        "modes never used": int((usage == 0).sum()),
        "size of λ, median": size[used].median().item(),
        "angle of λ, median": angle[used].median().item(),
        "angle of λ, 90th percentile": angle[used].quantile(0.9).item(),
        "constant modes": int((used & ((size - 1).abs() < 0.05) & (angle < 2)).sum()),
        "constant modes in real CLIP": count_real_constant(name),
        "modes with size of λ above 1": int((used & (size > 1)).sum()),
        "modes that carry half of all uses": int((by_use.cumsum(0) < by_use.sum() / 2).sum()) + 1,
        "modes used by over half of the images": int((usage > results["images"] / 2).sum()),
        "modes used by 99% of the images": int((usage >= 0.99 * results["images"]).sum()),
    }


def count_real_constant(name):
    """Modes that real CLIP itself holds constant, as in animals/constant_val.py: over the images that keep the mode
    (1,000 or more), the encoded block 12 is the encoded block 6 times a slope within 0.2 of 1, and that fit explains
    half of it or more."""

    to_complex = lambda latent: torch.complex(latent[:, 1::2], latent[:, 2::2])
    latents = load_latents(name)
    z6, end = to_complex(latents["z6"]), to_complex(latents["end"])
    slope = (z6.conj() * end).sum(dim=0) / z6.abs().pow(2).sum(dim=0).clamp_min(1e-12)
    fit = 1 - (end - slope * z6).abs().pow(2).sum(dim=0) / end.abs().pow(2).sum(dim=0).clamp_min(1e-12)
    return int((((z6 != 0).sum(dim=0) >= 1_000) & ((slope - 1).abs() < 0.2) & (fit >= 0.5)).sum())


def get_grid(title):
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 9), sharex=True, sharey=True)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
    for ax, name in zip(axes.flat, KAES):
        ax.set_title(MODELS[name], fontsize=9.5)
    return fig, axes


def draw_eigenvalues(table):
    fig, axes = get_grid("Most modes shrink from block 6 to block 12 and turn by only a few degrees. With a bilinear control every λ stays near 1")
    for ax, name in zip(axes.flat, KAES):
        modes = get_modes(load_results(name))
        used = modes["usage"] >= 50
        angle = modes["angle"].clamp(max=MAX_ANGLE)
        ax.scatter(angle[~used], modes["size"][~used], s=3, color=MUTED, alpha=0.35, lw=0, label="mode used by fewer than 50 images")
        ax.scatter(angle[used], modes["size"][used], s=4 + 70 * modes["usage"][used] / 50_000, color=BLUE, alpha=0.55, lw=0, label="mode used by 50 or more images (larger dot: more images)")
        ax.axhline(1, color=INK, lw=0.8)
        ax.text(0.97, 0.92, f"{table[name]['modes used by 50+ images']:,} of 1,024 modes in use", transform=ax.transAxes, ha="right", fontsize=8, color=INK)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylim(0.005, 10)
    axes[0, 0].set_xlim(-1, MAX_ANGLE + 1)
    axes[0, 0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _: f"{value:g}"))
    axes[0, 0].text(MAX_ANGLE, 1.15, "size 1: the mode keeps its size", ha="right", fontsize=8, color=INK)
    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="upper right", bbox_to_anchor=(0.99, 0.965), ncol=2, fontsize=9, labelcolor=INK, markerscale=1.5)
    for ax in axes[-1]:
        ax.set_xlabel(f"angle of λ (degrees; {MAX_ANGLE} = {MAX_ANGLE} or more)")
    for ax in axes[:, 0]:
        ax.set_ylabel("size of λ")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "inside_eigenvalues")


def draw_usage(table):
    fig, axes = get_grid("With an additive control about 300 of the 1,024 modes are in use. 66 to 158 modes carry half of all uses")
    for ax, name in zip(axes.flat, KAES):
        share = 100 * get_modes(load_results(name))["usage"].sort(descending=True).values / 50_000
        half = table[name]["modes that carry half of all uses"]
        ax.fill_between(torch.arange(1, len(share) + 1), share, color=LIGHT_BLUE, lw=0)
        ax.plot(torch.arange(1, len(share) + 1), share, color=BLUE, lw=2)
        ax.axvline(half, color=INK, lw=0.8)
        ax.text(0.97, 0.9, f"half of all uses: the first {half} modes\nused by over half of the images: {table[name]['modes used by over half of the images']} modes", transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK)
    axes[0, 0].set_xlim(0, 1024)
    axes[0, 0].set_ylim(0, 105)
    for ax in axes[-1]:
        ax.set_xlabel("modes, from most to least used")
    for ax in axes[:, 0]:
        ax.set_ylabel("images that use the mode (%)")
    fig.tight_layout()
    save(fig, "inside_usage")


if __name__ == "__main__":
    table = {name: count(name) for name in [*KAES, *EXTRA]}
    (DATA / "inside_modes.json").write_text(json.dumps(table, indent=1, ensure_ascii=False))
    for name, numbers in table.items():
        print(name, {key: round(value, 2) if isinstance(value, float) else value for key, value in numbers.items() if key != "label"})
    draw_eigenvalues(table)
    draw_usage(table)
