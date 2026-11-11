"""Figures of report.html: the control run's accuracy so far, species accuracy, example species photos, and speed."""

import re
import sys
import tarfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "animals"))
sys.path.insert(0, "/home/nsa325/work/koopmann/second_analysis")
from demo_utils import BLUE, INK, MUTED, ORANGE, load_clip  # noqa: E402  (also sets the plot style)
from extract_hoofed import SHARDS  # noqa: E402
from probe_kaes import get_output, load_kae  # noqa: E402

# Per run: label, colour, and how far (in points) the label sits above its last dot
RUNS = {"on_none": ("no control", MUTED, -14), "on_add": ("additive control", ORANGE, -13), "on_bilinear": ("bilinear control", BLUE, 6)}
# Species accuracy by CLIP's zero-shot head on held-out photos, from the step-7,500 checkpoints, before and after tuning eigenvalues
SPECIES = [("KAE, no control", 25.6, MUTED), ("KAE, no control, eigenvalues tuned", 28.6, MUTED), ("KAE, additive control", 31.6, ORANGE),
           ("KAE, additive control, eigenvalues tuned", 35.0, ORANGE), ("real CLIP", 43.7, INK)]  # fmt: skip
LOOKALIKES = ["n02422699", "n02423218", "n02423589", "n02427470", "n02422391", "n02420828"]


def draw_diagram():
    """The model: CLIP's blocks 1 to 6, then the KAE in place of blocks 7 to 12, with the patch tokens as a control."""

    fig, ax = plt.subplots(figsize=(12.5, 4.2))
    boxes = {
        "clip_start": (0.8, 2.0, "CLIP\nblocks 1 to 6", INK),
        "clip_end": (5.6, 3.4, "CLIP blocks 7 to 12\n(replaced by the KAE)", MUTED),
        "head": (11.3, 2.0, "CLIP's head\n(the answer)", INK),
        "summary": (3.0, 2.0, "the CLS token\nof block 6", INK),
        "encoder": (4.9, 2.0, "encoder", BLUE),
        "modes": (6.2, 2.0, "modes z₆\n(128 kept)", BLUE),
        "step": (7.9, 2.0, "z₁₂ = λ · z₆ + control", BLUE),
        "decoder": (9.7, 2.0, "decoder", BLUE),
        "patches": (3.0, 0.5, "the 49 patch tokens\nof block 6", INK),
        "control": (5.5, 0.5, "control encoder", ORANGE),
    }
    for x, y, text, color in boxes.values():
        ax.text(x, y, text, ha="center", va="center", fontsize=9, color=INK, bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=color, lw=1.5, ls="--" if color == MUTED else "-"))
    arrows = [((1.65, 2.0), (2.0, 2.0), INK), ((4.0, 2.0), (4.4, 2.0), INK), ((5.4, 2.0), (5.6, 2.0), BLUE), ((6.75, 2.0), (6.95, 2.0), BLUE),
              ((8.9, 2.0), (9.15, 2.0), BLUE), ((10.2, 2.0), (10.6, 2.0), BLUE), ((1.0, 1.6), (2.0, 0.6), INK), ((4.0, 0.5), (4.6, 0.5), INK),
              ((6.4, 0.6), (7.8, 1.65), ORANGE), ((1.0, 2.4), (4.4, 3.35), MUTED), ((6.8, 3.35), (11.1, 2.45), MUTED)]  # fmt: skip
    for start, end, color in arrows:
        ax.annotate("", end, start, arrowprops=dict(arrowstyle="->", color=color, lw=1.5, ls="--" if color == MUTED else "-"))
    ax.text(7.7, 0.8, "the control: added to the\nmodes this image uses", fontsize=9, color=ORANGE, va="center")
    ax.text(7.3, 2.6, "the KAE", fontsize=10, color=BLUE, ha="center")
    ax.set_xlim(-0.2, 12.3)
    ax.set_ylim(0, 4)
    ax.axis("off")
    fig.savefig(HERE / "diagram.png", dpi=150, bbox_inches="tight")


def draw_lane(ax, y, texts, color, xs):
    """One row of boxes joined by arrows. Returns the boxes."""

    boxes = [ax.text(x, y, text, ha="center", va="center", fontsize=8, color=INK, bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=color, lw=1.4)) for x, text in zip(xs, texts)]
    for left, right in zip(boxes, boxes[1:]):
        join(ax, left, right, color)
    return boxes


def join(ax, start, end, color):
    """An arrow from one box to another, stopping at their edges."""

    props = dict(arrowstyle="->", color=color, lw=1.4, patchA=start.get_bbox_patch(), patchB=end.get_bbox_patch(), shrinkA=2, shrinkB=2)
    ax.annotate("", end.get_position(), start.get_position(), arrowprops=props)


def draw_architecture():
    """Every layer of the CLS-only KAE with additive control, with the size of what passes between them."""

    fig, ax = plt.subplots(figsize=(18, 6.6))
    xs = [1.0, 3.7, 6.4, 9.1, 11.8, 14.5]
    state = ["CLS token\nafter block 6\n[768]", "standardize\n(fixed mean and std\nper coordinate)", "encoder MLP\n768 → 4096 → 4096 → 2049\nLeakyReLU between layers",
             "modes [2049]\n1 real number +\n1,024 pairs of numbers", "real-start rule (optional)\neach pair (x, y)\nbecomes (x, 0)", "TopK\nkeep the 128 largest\nmodes, drop the rest"]  # fmt: skip
    control = ["49 patch tokens\nafter block 6\n[49 × 768]", "LayerNorm\nLinear 768 → 512\n+ learned positions", "cross-attention, 1 layer\n16 learned queries, 8 heads\n[16 × 512]",
               "flatten [8192]\nLinear 8192 → 2048\nLeakyReLU:  this is u", "Linear 2048 → 2049\nthis is B\n(starts at zero)", "gate\nzero for the modes\nTopK dropped"]  # fmt: skip
    output = ["z₁₂ [2049]", "decoder MLP\n2049 → 4096 → 4096 → 768\nLeakyReLU between layers", "undo\nstandardize", "predicted CLS token\nafter block 12\n[768]",
              "CLIP's own head (frozen)\nLayerNorm, projection,\nclass-name embeddings", "the answer"]  # fmt: skip
    top, middle, bottom = draw_lane(ax, 5.6, state, BLUE, xs), draw_lane(ax, 3.6, control, ORANGE, xs), draw_lane(ax, 0.9, output, BLUE, xs)
    step = ax.text(17.4, 4.6, "the step\nz₁₂ = λ · z₆ + gate · B · u\nλ: 1 real + 1,024 complex\nnumbers, shared by all images", ha="center", va="center", fontsize=8, color=INK,
                   bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=INK, lw=1.8))  # fmt: skip
    join(ax, top[-1], step, BLUE)
    join(ax, middle[-1], step, ORANGE)
    ax.plot([17.4, 17.4, 1.0], [3.95, 2.2, 2.2], color=INK, lw=1.4)
    ax.annotate("", (1.0, 1.25), (1.0, 2.2), arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.text(16.1, 5.45, "z₆", fontsize=9, color=BLUE)
    for y, label, color in [(6.4, "State path (the KAE's encoder)", BLUE), (4.4, "Control path (the control encoder)", ORANGE), (1.75, "      Output path (the KAE's decoder, then CLIP's head)", BLUE)]:
        ax.text(0.0, y, label, fontsize=10, color=color)
    ax.set_xlim(-0.2, 19.3)
    ax.set_ylim(0.2, 6.8)
    ax.axis("off")
    fig.savefig(HERE / "architecture.png", dpi=150, bbox_inches="tight")


def plot_training():
    """ImageNet accuracy of the three KAEs of the control run at each checkpoint so far."""

    log = (HERE.parent / "night" / "on.log").read_text()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for run, (label, color, lift) in RUNS.items():
        points = [(int(step), 100 * float(accuracy)) for step, accuracy in re.findall(rf"step (\d+) \| {run}: accuracy ([\d.]+)", log)]
        steps, accuracy = zip(*points)
        ax.plot(steps, accuracy, color=color, lw=2, marker="o", ms=6)
        ax.annotate(f"{label}: {accuracy[-1]:.1f}", (steps[-1], accuracy[-1]), xytext=(8, lift), textcoords="offset points", color=INK, fontsize=9)
    for level, label in [(63.3, "real CLIP: 63.3"), (52.1, "the KAE before the control, fully trained: 52.1")]:
        ax.axhline(level, color=MUTED, lw=1, ls=":")
        ax.annotate(label, (300, level), xytext=(0, 4), textcoords="offset points", color=MUTED, fontsize=8)
    ax.set_xlim(0, 30_000)
    ax.set_ylim(40, 66)
    ax.set_xlabel("training step (stopped at 15,000 of 30,000 to start the CLS-only run)")
    ax.set_ylabel("ImageNet accuracy (%)")
    ax.set_title("Patch tokens as a control lift the KAE by about 7 points")
    fig.savefig(HERE / "training.png", dpi=150, bbox_inches="tight")


def plot_species():
    """Accuracy on the 49 species by CLIP's zero-shot head."""

    fig, ax = plt.subplots(figsize=(7, 2.8))
    names, values, colors = zip(*SPECIES[::-1])
    ax.barh(names, values, color=colors, height=0.6)
    for row, value in enumerate(values):
        ax.annotate(f"{value:.1f}", (value, row), xytext=(4, -3), textcoords="offset points", color=INK, fontsize=9)
    ax.set_xlim(0, 50)
    ax.set_xlabel("species accuracy with CLIP's zero-shot head (%)")
    ax.set_title("The control and eigenvalue tuning both help on look-alike species")
    ax.grid(axis="y", visible=False)
    fig.savefig(HERE / "species.png", dpi=150, bbox_inches="tight")


def open_first(names, synset):
    """The first image of a species in the shards, cut to a square."""

    shard, name = next((shard, name) for shard in sorted(names) for name in names[shard] if name.startswith(synset))
    with tarfile.open(SHARDS / shard) as tar:
        picture = Image.open(tar.extractfile(name)).convert("RGB")
    side, (width, height) = min(picture.size), picture.size
    return picture.crop(((width - side) // 2, (height - side) // 2, (width + side) // 2, (height + side) // 2))


def show_lookalikes():
    """One photo of each of six antelope species that look alike."""

    found = torch.load(HERE.parent / "animals" / "w21_hoofed.pt")
    fig, axes = plt.subplots(1, len(LOOKALIKES), figsize=(2 * len(LOOKALIKES), 2.3))
    for ax, synset in zip(axes, LOOKALIKES):
        ax.imshow(open_first(found["names"], synset))
        ax.set_xlabel(found["synsets"][synset]["name"].split(",")[0], fontsize=9, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(HERE / "lookalikes.png", dpi=150, bbox_inches="tight")


def clock(run):
    """Best of 5 timings, in ms."""

    times = []
    for _ in range(5):
        began = time.perf_counter()
        run()
        times.append(1000 * (time.perf_counter() - began))
    return min(times)


def print_speed():
    """Time for 256 images on this CPU: CLIP's blocks 7 to 12 against the KAE without and with the control."""

    blocks = load_clip().visual.transformer.resblocks[6:]
    tokens = torch.randn(256, 50, 768)  # [images, tokens, width]
    summary = torch.cat([tokens[:, 0], tokens[:, 1:].mean(dim=1), tokens[:, 1:].std(dim=1)], dim=1)[:, None].repeat(1, 2, 1)

    def run_clip():
        x = tokens
        for block in blocks:
            x = block(x)

    print(f"CLIP blocks 7-12: {clock(run_clip):.0f} ms")
    for run in ["on_none", "on_bilinear"]:
        kae, stats = load_kae(run)
        print(f"{run}: {clock(lambda: get_output(kae, stats, summary, tokens[:, 1:])):.0f} ms")


if __name__ == "__main__":
    torch.set_num_threads(16)
    draw_diagram()
    plot_training()
    plot_species()
    show_lookalikes()
    print_speed()
