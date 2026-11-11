"""
Writes second_analysis/dino_modes_<name>.ipynb: the spectrum, the snow test and the modes of a KAE trained on DINOv3 ViT-B/16.
Usage: make_dino_modes_notebook.py <name> <expected probe accuracy> <expected agreement> [findings.md]
The optional file holds the findings written after a first execution; it becomes the second cell.
"""
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

name, expected_accuracy, expected_agreement = sys.argv[1:4]
findings = Path(sys.argv[4]).read_text() if len(sys.argv) > 4 else None
cells = []
md = lambda s: cells.append(new_markdown_cell(s.strip()))
code = lambda s: cells.append(new_code_cell(s.strip()))

md(f"""
### The spectrum, snow and modes of `{name}`

The KAE stands in for DINOv3 ViT-B/16's blocks 7 to 12. It encodes the state after block 6, applies K once, and decodes a prediction of the state after block 12. DINOv3's final layer norm and an ImageNet linear probe then read the predicted CLS token as if it were the real block 12. The probe plays the part CLIP's zero-shot classifier plays in the CLIP notebooks.

A state is the CLS token, the mean of the 196 patch tokens and their spread (std), so 2,304 numbers per block. Only the CLS part goes through the probe.

`{name}`: Jordan K (1 real mode and 1,024 two-by-two blocks), TopK 256 (128 modes per image), shared modes, a 2-layer encoder and decoder 4,096 wide, weight decay 0.1, trained for 12 epochs on 3 views of ImageNet plus 1.16 million ImageNet-21k images.

Each two-by-two block is one mode with one complex eigenvalue `λ`. Its size `|λ|` is how much the mode grows from block 6 to block 12, and its angle is how far the mode's two numbers turn. We edit a mode by multiplying `λ`: by a real number to change its size, or by `e^(iθ)` to turn it by `θ`. Nothing else changes.

NOTE: DINOv3 is loaded with `transformers`, which has to be importable from the notebook's kernel.
""")

if findings:
    md(findings)

code(f"""
import math
import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open_clip
import pandas as pd
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from PIL import Image
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score
from torchvision import transforms

sys.path.insert(0, "../scripts/tokae")
import train

load_dotenv("../.env")
torch.set_grad_enabled(False)
torch.manual_seed(0)
torch.set_num_threads(32)
pd.set_option("display.width", 220)

NAME, ENCODER = "{name}", "dinov3-vitb16"
CKPT_DIR = Path(os.getenv("WEIGHTS_CACHE")) / "tokae"
# From wandb, for this checkpoint. The prediction cell must reproduce them, or loading is wrong.
EXPECTED = {{"probe accuracy": {expected_accuracy}, "agrees with DINOv3": {expected_agreement}}}
NUM_SNOW_MODES = 6
NUM_COHERENT = 10
NUM_RANDOM = 6
SNOW_FOLDERS = {{
    "armytank": 847,
    "carwheel": 479,
    "firetruck": 555,
    "motorscooter": 670,
    "racingcar": 751,
    "schoolbus": 779,
    "trafficlight": 920,
}}

BLUE, ORANGE, MUTED, INK = "#2a78d6", "#eb6834", "#898781", "#52514e"
plt.rcParams.update(
    {{
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": "#e8e7e3",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
    }}
)
""")

md("""
##### Model and data

States are standardized with the statistics the model was trained with. With shared modes, block 12 keeps the modes TopK picked at block 6, so every kept mode has a value at both ends. We write each mode's two numbers as one complex number: `z₆` at block 6 and `z₁₂` at block 12. K then says `z₁₂ ≈ λ·z₆`, mode by mode.

NOTE: No gallery in this notebook shows insects, arachnids or their relatives (crabs, lobsters and crayfish included). Images of those classes are skipped, along with worms, snails, slugs, chitons, starfish, sea urchins, honeycombs, apiaries, spider webs and any image for which one of those classes is among DINOv3's top 5 answers. Modes that mostly fire on such images are left out of the galleries. They still count in every number.
""")

code("""
root = Path(os.getenv("DATASETS_CACHE"))
stats = torch.load(CKPT_DIR / f"{NAME}_stats.pt")
mean, std = stats["mean"], stats["std"]

# shape: [images, 2, 2304], the state after blocks 6 and 12
val = torch.load(root / "imagenet" / f"clip_cls_patch_std_{ENCODER}_val_blocks_5_11.pt")
val_x, val_y = (val["states"].float() - mean) / std, val["labels"]
images = load_dataset("ILSVRC/imagenet-1k", split="validation", cache_dir=os.getenv("DATASETS_CACHE"))
assert torch.equal(val_y, torch.tensor(list(images["label"]))), "cached labels don't follow the image order"
class_names = open_clip.IMAGENET_CLASSNAMES

kae = train.JordanKoopmanAutoencoder(
    top_k=256,
    shared_modes=stats["shared_modes"],
    in_features=val_x.shape[-1],
    latent_features=2049,
    hidden_config=stats["hidden_config"],
    bias=True,
    batchnorm=False,
    nonlinearity="leaky_relu",
)
kae.load_state_dict(load_file(CKPT_DIR / f"{NAME}.safetensors"))
kae.eval()

# DINOv3's final layer norm and the ImageNet linear probe, as used in training
head = train.DinoHead(ENCODER, mean, std, torch.device("cpu"), zero_shot=True)

# Images we never show: insects, arachnids and their relatives (crabs, lobsters and crayfish too), worms, snails, slugs and chitons,
# starfish and sea urchins (leggy or spiny), honeycombs and apiaries (bees) and spider webs,
# by label or by any of DINOv3's top 5 answers, as images of other classes can contain them too
HIDDEN_CLASSES = torch.tensor([*range(69, 80), 110, 111, *range(113, 117), *range(118, 127), *range(300, 329), 410, 599, 815])
hidden = torch.isin(val_y, HIDDEN_CLASSES) | torch.isin(head.get_log_probs(val_x[:, 1]).topk(5).indices, HIDDEN_CLASSES).any(dim=1)

# shape: [modes]. Mode 0 is the real eigenvalue, mode m > 0 is the block on latent coordinates 2m - 1 and 2m
eigenvalues = torch.cat([kae.real_eigenvalue.to(torch.cfloat), torch.complex(kae.block_real, kae.block_imag)])
size, angle = eigenvalues.abs(), torch.rad2deg(torch.angle(eigenvalues))


def to_complex(phi):
    \"\"\"Each mode as one complex number (mode 0 is real).\"\"\"

    return torch.cat([phi[..., :1].to(torch.cfloat), torch.view_as_complex(phi[..., 1:].unflatten(-1, (-1, 2)).contiguous())], dim=-1)


def to_latent(z):
    return torch.cat([z[..., :1].real, torch.view_as_real(z[..., 1:]).flatten(-2)], dim=-1)


def decode(z):
    return torch.cat([kae.decode(to_latent(batch)) for batch in z.split(5_000)])


def edit(z_start, changes):
    \"\"\"
    The KAE's prediction of block 12 with some eigenvalues multiplied by a factor. `changes` maps mode to factor.
    NOTE: The encoder doesn't depend on K, so we encode once and only redo K and the decoder.
    \"\"\"

    edited = eigenvalues.clone()
    for mode, factor in changes.items():
        edited[mode] = edited[mode] * factor
    return decode(edited * z_start)


def turn(degrees):
    return complex(math.cos(math.radians(degrees)), math.sin(math.radians(degrees)))


# shape: [images, modes], complex, at blocks 6 and 12
z = to_complex(torch.cat([train.encode_pairs(kae, batch) for batch in val_x.split(2_500)]))
z6, z12 = z[:, 0], z[:, 1]
assert torch.allclose(to_complex(kae.koopman_forward(to_latent(z6[:100]), 1)), eigenvalues * z6[:100], atol=1e-4), "K is not λ·z₆ mode by mode"

kept = z6 != 0
usage = kept.double().mean(dim=0)
print(f"{NAME}: hidden {stats['hidden_config']}, shared modes {stats['shared_modes']}, standardization {stats['standardization']}, {int(kept.sum(dim=1).median())} modes kept per image")
""")

md("""
##### Prediction

Probe accuracy on ImageNet's 50,000 validation images, and how often the KAE gives the same answer as DINOv3, right or wrong. These must match the numbers wandb logged for this checkpoint.
""")

code("""
true_answers = head.classify(val_x[:, 1])
answers = head.classify(decode(eigenvalues * z6))
scores = {
    "probe accuracy": (answers == val_y).double().mean().item(),
    "agrees with DINOv3": (answers == true_answers).double().mean().item(),
}
for key, value in scores.items():
    assert abs(value - EXPECTED[key]) < 0.002, f"{key}: {value:.4f} here, {EXPECTED[key]} on wandb"

pd.DataFrame(
    {
        "DINOv3": {"probe accuracy": (true_answers == val_y).double().mean().item(), "agrees with DINOv3": 1.0},
        "KAE": scores,
    }
).T.round(3)
""")

md("""
##### The spectrum

One eigenvalue per mode. Each block also has the conjugate `a - ib`, so the sign of an angle is a convention and we plot its size only. On the left, the complex plane, folded onto its upper half. On the right, the same modes as angle against size, with size on a log scale.

We sort modes into five groups, by rules on the validation images:
- **never kept**: TopK never picks the mode.
- **growth**: on for at least 99% of images, with a value that differs from image to image.
- **offset**: on for at least 99% of images with nearly the same value for all of them (the average value is over twice the spread), so it carries the average state, not something about the image.
- **inert**: an offset mode holding under 0.1% of the latent energy at block 6.
- **feature**: everything else, on for some images and off for others.
""")

code("""
energy6, energy12 = z6.abs().pow(2).sum(dim=0), z12.abs().pow(2).sum(dim=0)
share6, share12 = energy6 / energy6.sum(), energy12 / energy12.sum()

# How steady a mode's block-6 value is over images: the size of its average, over its spread around it
mean6 = z6.mean(dim=0)
steadiness = mean6.abs() / (z6 - mean6).abs().pow(2).mean(dim=0).sqrt().clamp_min(1e-12)
never, always = usage == 0, usage >= 0.99
steady = always & (steadiness > 2)
group = np.select(
    [never.numpy(), (always & ~steady).numpy(), (steady & (share6 < 1e-3)).numpy(), steady.numpy()],
    ["never kept", "growth", "inert", "offset"],
    "feature",
)

spectrum = pd.DataFrame(
    {
        "group": group,
        "|λ|": size.numpy(),
        "|angle|": angle.abs().numpy(),
        "kept": usage.numpy(),
        "share of block-6 latent energy": share6.numpy(),
        "share of block-12 latent energy": share12.numpy(),
    }
)
summary = spectrum.groupby("group").agg(
    modes=("kept", "size"),
    median_size=("|λ|", "median"),
    median_angle=("|angle|", "median"),
    median_kept=("kept", "median"),
    block_6_energy=("share of block-6 latent energy", "sum"),
    block_12_energy=("share of block-12 latent energy", "sum"),
)
summary = summary.loc[[g for g in ["growth", "offset", "inert", "feature", "never kept"] if g in summary.index]]

fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})
styles = [("never kept", MUTED, never), ("kept on some images", BLUE, ~never & ~always), ("on for at least 99% of images", ORANGE, always)]

ax = axes[0]
for radius in [1, 5, 10, 15]:
    arc = radius * np.exp(1j * np.linspace(0, np.pi, 200))
    ax.plot(arc.real, arc.imag, color=MUTED, lw=0.7, ls="--" if radius == 1 else ":")
    ax.annotate("unit circle" if radius == 1 else f"|λ| = {radius}", (radius * math.cos(2.2), radius * math.sin(2.2)), fontsize=7, color=MUTED, ha="right")
for label, color, mask in styles:
    ax.scatter(eigenvalues.real[mask].numpy(), eigenvalues.imag[mask].abs().numpy(), s=8 + 90 * usage[mask].numpy(), color=color, alpha=0.5, lw=0, label=label)
for mode in np.flatnonzero(group == "growth"):
    ax.annotate(f"mode {mode}", (eigenvalues.real[mode].item(), abs(eigenvalues.imag[mode].item())), xytext=(0, 9), textcoords="offset points", fontsize=8, color=INK, ha="center")
ax.set_xlabel("real part")
ax.set_ylabel("imaginary part (size)")
ax.set_aspect("equal")
ax.set_ylim(0, None)
ax.legend(loc="upper left", fontsize=8)
ax.set_title("Eigenvalues of K; dot area is how often the mode is kept", loc="left", color=INK)

ax = axes[1]
for label, color, mask in styles:
    ax.scatter(angle[mask].abs().numpy(), size[mask].numpy(), s=8 + 90 * usage[mask].numpy(), color=color, alpha=0.5, lw=0)
ax.axhline(1.0, color=MUTED, lw=0.7, ls="--")
ax.set_yscale("log")
ax.set_xlim(-4, 184)
ax.set_xticks([0, 45, 90, 135, 180])
ax.set_xlabel("angle of λ (degrees, size)")
ax.set_ylabel("|λ| (log)")
offsets = {"growth": (12, 0), "offset": (0, 16), "feature": (14, 14), "inert": (14, 16), "never kept": (14, -16)}
for name, row in summary.iterrows():
    ax.annotate(
        f"{name} ({int(row['modes'])})",
        (row["median_angle"], row["median_size"]),
        xytext=offsets[name],
        textcoords="offset points",
        fontsize=8,
        color=INK,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
    )
ax.set_title("The same modes, as angle against size", loc="left", color=INK)
fig.tight_layout()
plt.show()

used = ~never
quantiles = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])
print(f"{int(used.sum())} of {len(usage)} modes are ever kept, {int(always.sum())} are on for at least 99% of images")
print("Kept modes, |λ| at the 5th, 25th, 50th, 75th and 95th percentile:", [round(q, 2) for q in size[used].quantile(quantiles).tolist()])
print("Kept modes, angle (size) at the same percentiles:", [round(q, 1) for q in angle[used].abs().quantile(quantiles).tolist()])
summary.round(3)
""")

md("""
##### Does a mode do what its eigenvalue says?

`λ` is one number shared by every image, so a mode behaves as `λ` says only if `z₁₂ ≈ λ·z₆` image by image. For each mode kept on at least 0.5% of images, over the images that keep it:
- **measured gain**: the median of `|z₁₂| / |z₆|`, against `|λ|`.
- **measured turn**: the angle of `Σ z₁₂·conj(z₆)`, which is the one turn that best fits all the images, against the angle of `λ`.
- **turn agreement**: `|Σ z₁₂·conj(z₆)| / Σ |z₁₂||z₆|`. It is 1 if every image turns by the same angle, and near 0 if the images turn every which way.
- **R²**: how much of the image-to-image variation of `z₁₂` is explained by `λ·z₆`.
""")

code("""
rows = []
for mode in (usage >= 0.005).nonzero()[:, 0].tolist():
    on = kept[:, mode]
    a, b = z6[on, mode], z12[on, mode]
    pooled = (b * a.conj()).sum()
    rows.append(
        {
            "mode": mode,
            "group": group[mode],
            "|λ|": size[mode].item(),
            "angle": angle[mode].item(),
            "kept": usage[mode].item(),
            "measured gain": (b.abs() / a.abs()).median().item(),
            "measured turn": math.degrees(torch.angle(pooled).item()),
            "turn agreement": (pooled.abs() / (b.abs() * a.abs()).sum()).item(),
            "R²": 1 - ((b - eigenvalues[mode] * a).abs().pow(2).sum() / (b - b.mean()).abs().pow(2).sum().clamp_min(1e-12)).item(),
        }
    )
modes = pd.DataFrame(rows).set_index("mode")
turn_gap = (modes["measured turn"] - modes["angle"] + 180) % 360 - 180

fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
colors = np.where(modes["group"] == "feature", BLUE, ORANGE)
axes[0].scatter(modes["|λ|"], modes["measured gain"], s=9, color=colors, alpha=0.6, lw=0)
axes[0].plot([0.4, 20], [0.4, 20], color=INK, lw=0.8, ls=":")
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlabel("|λ| (log)")
axes[0].set_title("measured gain (log)", loc="left", color=INK)
axes[1].scatter(modes["angle"], modes["measured turn"], s=9, color=colors, alpha=0.6, lw=0)
axes[1].plot([-180, 180], [-180, 180], color=INK, lw=0.8, ls=":")
axes[1].set_xticks([-180, -90, 0, 90, 180])
axes[1].set_yticks([-180, -90, 0, 90, 180])
axes[1].set_xlabel("angle of λ (degrees)")
axes[1].set_title("measured turn (degrees)", loc="left", color=INK)
axes[2].hist(modes["turn agreement"], bins=40, color=BLUE, edgecolor="white", linewidth=0.5)
axes[2].set_xlabel("turn agreement")
axes[2].set_ylabel("modes")
axes[2].set_title("do images turn together?", loc="left", color=INK)
axes[3].hist(modes["R²"].clip(lower=-1), bins=40, color=BLUE, edgecolor="white", linewidth=0.5)
axes[3].set_xlabel("R² (values under -1 shown at -1)")
axes[3].set_title("R² of λ·z₆ for z₁₂", loc="left", color=INK)
fig.tight_layout()
plt.show()

print(f"{len(modes)} modes kept on at least 0.5% of images. Orange: growth, offset and inert modes; blue: feature modes.")
print(f"Gain against |λ|: correlation {np.corrcoef(np.log(modes['measured gain']), np.log(modes['|λ|']))[0, 1]:.3f} (of logs). Turn against angle: median gap {turn_gap.abs().median():.1f}°, 90th percentile {turn_gap.abs().quantile(0.9):.1f}°")
modes.groupby("group")[["turn agreement", "R²"]].agg(["count", "median"]).round(3)
""")

md("""
##### The modes that are always on

Every mode on for at least 99% of images. `steadiness` is the size of the mode's average value over its spread across images, at block 6: high means the mode is close to a constant. Then two edits, read on all 50,000 validation images: removing the mode (`λ` to 0), and keeping its size but setting its angle to 0, so that it no longer turns.
""")

code("""
base_accuracy = (answers == val_y).double().mean().item()
rows = []
for mode in always.nonzero()[:, 0].tolist():
    removed = head.classify(edit(z6, {mode: 0.0}))
    unturned = head.classify(edit(z6, {mode: turn(-angle[mode].item())}))
    rows.append(
        {
            "mode": mode,
            "group": group[mode],
            "|λ|": size[mode].item(),
            "angle": angle[mode].item(),
            "steadiness": steadiness[mode].item(),
            "share of block-6 latent energy": share6[mode].item(),
            "share of block-12 latent energy": share12[mode].item(),
            "accuracy, removed": (removed == val_y).double().mean().item(),
            "accuracy, angle set to 0": (unturned == val_y).double().mean().item(),
        }
    )
print(f"Accuracy as trained: {base_accuracy:.3f}")
pd.DataFrame(rows).set_index("mode").sort_values(["group", "share of block-6 latent energy"], ascending=[True, False]).round(3)
""")

md("""
##### Where the gap to DINOv3 opens

The KAE has three parts that can each lose accuracy: the encoder and decoder, the sharing of modes between the blocks, and K. To see which one does, we swap parts of the KAE's prediction for what DINOv3 really does at block 12:
- **block 12 through the KAE, its own modes**: encode block 12, decode. No K, no sharing. This is the cost of the encoder and decoder.
- **block 12 on the modes picked at block 6**: the same, but keeping the modes TopK picked at block 6. This is what a perfect step would give.
- **K's size, true turn** and **K's turn, true size**: K's prediction for every mode, with its angle or its size swapped for the true one at block 12.
- **true values for one group of modes**: K for every mode except one group, which takes its true block-12 value.

NOTE: Every row but the last uses block 12 itself, so these are ceilings. A step from block 6 cannot know everything about block 12, so the gap to "K as trained" is the most a better step could win back, not what it would.
""")

code("""
def mix(z_start, z_end, part):
    \"\"\"K's prediction of every kept mode, with its size or its turn swapped for the true one at block 12.\"\"\"

    predicted = eigenvalues * z_start
    if part == "size":
        mixed = z_end.abs() * torch.exp(1j * torch.angle(predicted))
    else:
        mixed = predicted.abs() * torch.exp(1j * torch.angle(z_end))
    return torch.where(z_start != 0, mixed, torch.zeros_like(mixed))


def get_scores(z_end):
    found = head.classify(decode(z_end))
    return {"probe accuracy": (found == val_y).double().mean().item(), "agrees with DINOv3": (found == true_answers).double().mean().item()}


def with_true(mask):
    \"\"\"K's prediction, with the modes in `mask` taking their true block-12 value.\"\"\"

    mixed = eigenvalues * z6
    mixed[:, mask] = z12[:, mask]
    return mixed


is_feature = torch.tensor(group == "feature")
strong, weak = is_feature & (size > 5), is_feature & (size <= 5)
rows = {
    "DINOv3's block 12": {"probe accuracy": (true_answers == val_y).double().mean().item(), "agrees with DINOv3": 1.0},
    "block 12 through the KAE, its own modes": get_scores(to_complex(torch.cat([kae.encode(batch) for batch in val_x[:, 1].split(5_000)]))),
    "block 12 on the modes picked at block 6": get_scores(z12),
    "K's size, true turn": get_scores(mix(z6, z12, "turn")),
    "K's turn, true size": get_scores(mix(z6, z12, "size")),
    f"true values for the {int(strong.sum())} feature modes with |λ| > 5": get_scores(with_true(strong)),
    f"true values for the {int(weak.sum())} feature modes with |λ| ≤ 5": get_scores(with_true(weak)),
    f"true values for the {int(always.sum())} always-on modes": get_scores(with_true(always)),
    "K as trained": scores,
}
pd.DataFrame(rows).T.round(3)
""")

md("""
##### Snow: where the KAE fails and DINOv3 does not

129 photos of vehicles in snow ([Santurkar et al.](https://arxiv.org/abs/2112.01008)), against the 350 normal validation images of the same 7 classes. The snowy photos go through DINOv3 here, so first a check that this extraction (CLS, patch mean, patch std, skipping the register tokens) matches the cached validation states on a few images.
""")

code("""
from transformers import AutoModel

dino = AutoModel.from_pretrained(train.DINO_MODELS[ENCODER], cache_dir=f"{os.getenv('HF_HOME')}/hub").eval()
first_patch = 1 + dino.config.num_register_tokens
preprocess = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        train.to_rgb,
        transforms.ToTensor(),
        transforms.Normalize(*train.IMAGENET_MEAN_STD),
    ]
)


def get_states(pixels):
    \"\"\"Unstandardized states after blocks 6 and 12, built the way the training cache was.\"\"\"

    states = []

    def keep(_, __, output):
        output = output[0] if isinstance(output, tuple) else output
        patches = output[:, first_patch:]
        states.append(torch.cat([output[:, 0], patches.mean(dim=1), patches.std(dim=1)], dim=-1))

    hooks = [dino.model.layer[block].register_forward_hook(keep) for block in stats["blocks"]]
    for batch in pixels.split(32):
        dino(pixel_values=batch)
    for hook in hooks:
        hook.remove()

    # NOTE: Each batch adds block 6 then block 12
    return torch.cat([torch.stack(states[i : i + 2], dim=1) for i in range(0, len(states), 2)])


check = get_states(torch.stack([preprocess(images[i]["image"]) for i in range(5)]))
cached = val["states"][:5].float()
gap = ((check - cached).norm(dim=-1) / cached.norm(dim=-1)).max().item()
print(f"Largest relative gap to the cached states: {gap:.2e}")
assert gap < 1e-2, "snowy states would not be comparable to the cached ones"

paths = sorted((root / "snowy_imagenet").glob("*/*.jpg"))
snowy_x = (get_states(torch.stack([preprocess(Image.open(path)) for path in paths])) - mean) / std
snowy_y = torch.tensor([SNOW_FOLDERS[path.parent.name] for path in paths])

normal = torch.isin(val_y, torch.tensor(list(SNOW_FOLDERS.values())))
x, y = torch.cat([val_x[normal], snowy_x]), torch.cat([val_y[normal], snowy_y])
is_snowy = torch.cat([torch.zeros(int(normal.sum()), dtype=torch.bool), torch.ones(len(snowy_x), dtype=torch.bool)])

snow_pairs = to_complex(train.encode_pairs(kae, x))
snow_z, snow_z12 = snow_pairs[:, 0], snow_pairs[:, 1]
snow_prediction = decode(eigenvalues * snow_z)
snow_answers, snow_true_answers = head.classify(snow_prediction), head.classify(x[:, 1])
kae_right, true_right = snow_answers == y, snow_true_answers == y

by_class = {
    name: {
        "snowy images": int((is_snowy & (y == c)).sum()),
        "DINOv3, snowy": true_right[is_snowy & (y == c)].double().mean().item(),
        "DINOv3, normal": true_right[~is_snowy & (y == c)].double().mean().item(),
        "KAE, snowy": kae_right[is_snowy & (y == c)].double().mean().item(),
        "KAE, normal": kae_right[~is_snowy & (y == c)].double().mean().item(),
    }
    for name, c in SNOW_FOLDERS.items()
}
by_class["all 7 classes"] = {
    "snowy images": int(is_snowy.sum()),
    "DINOv3, snowy": true_right[is_snowy].double().mean().item(),
    "DINOv3, normal": true_right[~is_snowy].double().mean().item(),
    "KAE, snowy": kae_right[is_snowy].double().mean().item(),
    "KAE, normal": kae_right[~is_snowy].double().mean().item(),
}
pd.DataFrame(by_class).T.round(3)
""")

md("""
What does the KAE answer on the snowy images it gets wrong, and which stage is to blame? We read each stage with the probe: block 12 itself, the KAE's reconstruction of block 12 (encode block 12, decode, no K), and the KAE's prediction (encode block 6, apply K, decode). Errors are over the spread of all validation states around their mean, so that both groups share one denominator.
""")

code("""
def get_error(pred, target, reference):
    return ((pred - target).pow(2).sum() / len(pred)).item() / ((reference - reference.mean(dim=0)).pow(2).sum() / len(reference)).item()


wrong = is_snowy & ~kae_right
print(f"The KAE's answers on the {int(wrong.sum())} snowy images it gets wrong:", Counter(class_names[c] for c in snow_answers[wrong].tolist()).most_common(6))
print(f"DINOv3's answers on the {int((is_snowy & ~true_right).sum())} it gets wrong:", Counter(class_names[c] for c in snow_true_answers[is_snowy & ~true_right].tolist()).most_common(4))

reconstruction = kae.decode(kae.encode(x[:, 1]))
rows = {}
for label, mask in [("snowy", is_snowy), ("normal", ~is_snowy)]:
    rows[label] = {
        "probe on block 12": true_right[mask].double().mean().item(),
        "probe on its reconstruction": (head.classify(reconstruction[mask]) == y[mask]).double().mean().item(),
        "probe on the prediction": kae_right[mask].double().mean().item(),
        "reconstruction error, CLS": get_error(reconstruction[mask, :768], x[mask, 1, :768], val_x[:, 1, :768]),
        "prediction error, CLS": get_error(snow_prediction[mask, :768], x[mask, 1, :768], val_x[:, 1, :768]),
    }
pd.DataFrame(rows).T.round(3)
""")

md("""
##### The snow modes

We score each mode by how well its size at block 6 alone separates snowy from normal images (AUC), leaving out modes that are on for at least 90% of validation, as they cannot be specific to snow. Below the table, the 8 validation images where each of the top modes is most active.
""")

code("""
def get_aucs(activity, labels):
    return torch.tensor([roc_auc_score(labels.numpy(), activity[:, m].numpy()) if activity[:, m].any() else 0.5 for m in range(activity.shape[1])])


def show_modes(shown, labels):
    fig, axes = plt.subplots(len(shown), 8, figsize=(16, 2.2 * len(shown)), squeeze=False)
    for row, mode, label in zip(axes, shown, labels):
        order = z6[:, mode].abs().argsort(descending=True)
        for ax, index in zip(row, order[~hidden[order]][:8].tolist()):
            image = images[index]["image"].convert("RGB")
            image.thumbnail((256, 256))
            ax.imshow(np.asarray(image))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.set_xlabel(class_names[val_y[index]][:22], fontsize=8, color=INK)
        row[0].set_ylabel(f"mode {mode}\\n{label}\\non {usage[mode].item():.0%}, |λ| {size[mode]:.1f}, {angle[mode]:+.0f}°", fontsize=9, color=INK)
    fig.tight_layout()
    plt.show()


snow_activity = snow_z.abs()
aucs = get_aucs(snow_activity, is_snowy)
eligible = usage < 0.9
SNOW_MODES = [m for m in aucs.argsort(descending=True).tolist() if eligible[m]][:NUM_SNOW_MODES]

# Pick on one half of the snow test, score on the other, so the AUC isn't measured where the mode was picked
fold = torch.arange(len(y)) % 2
for half in [0, 1]:
    pick, test = fold == half, fold != half
    best = int(torch.where(eligible, get_aucs(snow_activity[pick], is_snowy[pick]), 0.0).argmax())
    print(f"Half {half}: picks mode {best}, AUC on the other half {roc_auc_score(is_snowy[test].numpy(), snow_activity[test, best].numpy()):.3f}")

show_modes(SNOW_MODES[:4], [f"AUC {aucs[m]:.2f}" for m in SNOW_MODES[:4]])
pd.DataFrame(
    {
        "AUC": aucs[SNOW_MODES].numpy(),
        "on for snowy": (snow_activity[is_snowy][:, SNOW_MODES] > 0).double().mean(dim=0).numpy(),
        "on for normal": (snow_activity[~is_snowy][:, SNOW_MODES] > 0).double().mean(dim=0).numpy(),
        "on for validation": usage[SNOW_MODES].numpy(),
        "|λ|": size[SNOW_MODES].numpy(),
        "angle": angle[SNOW_MODES].numpy(),
    },
    index=pd.Index(SNOW_MODES, name="mode"),
).round(3)
""")

md("""
##### Editing one snow mode: size or turn

For each of the top three snow modes, the KAE's accuracy on the 129 snowy images as we turn the mode's eigenvalue by an angle, keeping its size. A turn of 0° is K as trained, and a turn of 180° is the same as multiplying `λ` by -1. The orange line is the mode removed (`λ` to 0).

Then a check that the best turn isn't picked where it is scored: we pick it on the even snowy images and score it on the odd ones.
""")

code("""
TURNS = list(range(-180, 181, 30))
snowy_z, snowy_truth = snow_z[is_snowy], y[is_snowy]
as_trained = kae_right[is_snowy]
even = torch.arange(len(snowy_truth)) % 2 == 0

fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), sharey=True)
rows = []
for ax, mode in zip(axes, SNOW_MODES[:3]):
    right = {degrees: head.classify(edit(snowy_z, {mode: turn(degrees)})) == snowy_truth for degrees in TURNS}
    removed = head.classify(edit(snowy_z, {mode: 0.0})) == snowy_truth
    accuracy = [right[degrees].double().mean().item() for degrees in TURNS]
    ax.plot(TURNS, accuracy, color=BLUE, lw=2, marker="o", ms=4)
    ax.axhline(removed.double().mean().item(), color=ORANGE, lw=1.5, ls="--")
    ax.axhline(as_trained.double().mean().item(), color=MUTED, lw=1)
    ax.annotate("removed", (180, removed.double().mean().item()), xytext=(-2, 4), textcoords="offset points", fontsize=8, color=INK, ha="right")
    ax.annotate("as trained", (-180, as_trained.double().mean().item()), xytext=(2, -10), textcoords="offset points", fontsize=8, color=INK)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xlabel("turn applied to λ (degrees)")
    ax.set_title(f"mode {mode}: |λ| {size[mode]:.1f}, angle {angle[mode]:+.0f}°", loc="left", color=INK)

    best = max(TURNS[:-1], key=lambda degrees: right[degrees][even].double().mean().item())
    rows.append(
        {
            "mode": mode,
            "best turn, picked on even images": best,
            "odd images: as trained": as_trained[~even].double().mean().item(),
            "odd images: removed": removed[~even].double().mean().item(),
            "odd images: best turn": right[best][~even].double().mean().item(),
            "all snowy: fixed by the best turn": int((~as_trained & right[best]).sum()),
            "all snowy: broken by the best turn": int((as_trained & ~right[best]).sum()),
            "all snowy: fixed by removal": int((~as_trained & removed).sum()),
            "all snowy: broken by removal": int((as_trained & ~removed).sum()),
        }
    )
axes[0].set_ylabel("accuracy on the snowy images")
fig.tight_layout()
plt.show()

pd.DataFrame(rows).set_index("mode").round(3)
""")

md("""
##### Editing several snow modes at once

Snow is spread over several modes, so we edit the top 1 to 6 of them together. An image is fixed if its answer goes from wrong to right, and broken if it goes from right to wrong. On the left, the 129 snowy images. On the right, every validation image that keeps at least one of the edited modes, which is what the edit costs on ordinary images, as K is shared by every image.
""")

code("""
EDITS = {"removed": 0.0, "turned +90°": turn(90), "turned -90°": turn(-90), "turned 180°": -1.0}
rows = []
for count in range(1, NUM_SNOW_MODES + 1):
    chosen = SNOW_MODES[:count]
    touched = kept[:, chosen].any(dim=1)
    val_before = answers[touched] == val_y[touched]
    for label, factor in EDITS.items():
        changes = {mode: factor for mode in chosen}
        right = head.classify(edit(snow_z, changes)) == y
        val_right = head.classify(edit(z6[touched], changes)) == val_y[touched]
        rows.append(
            {
                "modes edited": count,
                "edit": label,
                "accuracy, snowy": right[is_snowy].double().mean().item(),
                "accuracy, normal": right[~is_snowy].double().mean().item(),
                "snowy: fixed": int((~kae_right & right & is_snowy).sum()),
                "snowy: broken": int((kae_right & ~right & is_snowy).sum()),
                "validation images touched": int(touched.sum()),
                "validation: fixed": int((~val_before & val_right).sum()),
                "validation: broken": int((val_before & ~val_right).sum()),
            }
        )
several = pd.DataFrame(rows)

removal = several[several["edit"] == "removed"].set_index("modes edited")
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
for ax, prefix, title in [(axes[0], "snowy", "the 129 snowy images"), (axes[1], "validation", "validation images that keep an edited mode")]:
    for column, label, color in [(f"{prefix}: fixed", "fixed", BLUE), (f"{prefix}: broken", "broken", ORANGE)]:
        ax.plot(removal.index, removal[column], color=color, lw=2, marker="o", ms=5)
        ax.annotate(label, (removal.index[-1], removal[column].iloc[-1]), xytext=(6, 0), textcoords="offset points", color=INK, va="center")
    ax.set_xlabel("snow modes removed")
    ax.set_ylabel("images")
    ax.set_title(title, loc="left", color=INK)
fig.tight_layout()
plt.show()

print(f"As trained: snowy {kae_right[is_snowy].double().mean():.3f}, normal {kae_right[~is_snowy].double().mean():.3f}; DINOv3: snowy {true_right[is_snowy].double().mean():.3f}, normal {true_right[~is_snowy].double().mean():.3f}")
several.set_index(["modes edited", "edit"]).round(3)
""")

md("""
The modes above were picked on the images they are scored on. As a check, we rank the modes by AUC on one half of the snow test, remove the top 4, and score the snowy images of the other half.
""")

code("""
rows = []
for half in [0, 1]:
    pick, test = fold == half, (fold != half) & is_snowy
    half_aucs = torch.where(eligible, get_aucs(snow_activity[pick], is_snowy[pick]), 0.0)
    chosen = half_aucs.argsort(descending=True)[:4].tolist()
    right = head.classify(edit(snow_z[test], {mode: 0.0 for mode in chosen})) == y[test]
    rows.append(
        {
            "picked on half": half,
            "modes": chosen,
            "other half, as trained": kae_right[test].double().mean().item(),
            "other half, 4 modes removed": right.double().mean().item(),
            "fixed": int((~kae_right[test] & right).sum()),
            "broken": int((kae_right[test] & ~right).sum()),
        }
    )
pd.DataFrame(rows).set_index("picked on half").round(3)
""")

md("""
Which snowy images removing the top 4 snow modes fixes, with the KAE's answer before and after, and the classes it breaks most on validation.
""")

code("""
changes = {mode: 0.0 for mode in SNOW_MODES[:4]}
after_answers = head.classify(edit(snow_z, changes))
fixed = (is_snowy & ~kae_right & (after_answers == y)).nonzero()[:, 0]
first_snowy = int((~is_snowy).sum())

shown = fixed[torch.randperm(len(fixed))[:8]]
if len(shown):
    fig, axes = plt.subplots(1, len(shown), figsize=(2 * len(shown), 3.2), squeeze=False)
    for ax, index in zip(axes[0], shown.tolist()):
        image = Image.open(paths[index - first_snowy]).convert("RGB")
        image.thumbnail((256, 256))
        ax.imshow(np.asarray(image))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_xlabel(f"before: {class_names[snow_answers[index]][:18]}\\nafter: {class_names[after_answers[index]][:18]}", fontsize=8, color=INK)
    fig.tight_layout()
    plt.show()

touched = kept[:, SNOW_MODES[:4]].any(dim=1)
truth, before = val_y[touched], answers[touched]
after = head.classify(edit(z6[touched], changes))
broken = (before == truth) & (after != truth)
print(f"Snowy images still wrong after the edit: {Counter(class_names[c] for c in after_answers[is_snowy & (after_answers != y)].tolist()).most_common(4)}")
pd.Series(Counter(class_names[c] for c in truth[broken].tolist()), dtype=int).sort_values(ascending=False).head(10).rename("validation images broken").to_frame()
""")

md("""
##### Is K wrong about the snow modes?

Removing the snow modes fixes snowy images. That would be a repair if K overdid snow on these images. We check it the same way as above, on the snow test: the top 4 snow modes take their true block-12 value while K handles the rest, and the other way round. Then how the snow modes really move from block 6 to block 12 on the snowy vehicles, against the snowplows and snowmobiles of validation.
""")

code("""
def get_snow_scores(z_end):
    right = head.classify(decode(z_end)) == y
    return {"accuracy, snowy": right[is_snowy].double().mean().item(), "accuracy, normal": right[~is_snowy].double().mean().item()}


chosen = SNOW_MODES[:4]
predicted = eigenvalues * snow_z
only_snow, all_but_snow = predicted.clone(), snow_z12.clone()
only_snow[:, chosen] = snow_z12[:, chosen]
all_but_snow[:, chosen] = predicted[:, chosen]
display(
    pd.DataFrame(
        {
            "K as trained": get_snow_scores(predicted),
            "true values for the top 4 snow modes, K for the rest": get_snow_scores(only_snow),
            "K for the top 4 snow modes, true values for the rest": get_snow_scores(all_but_snow),
            "true values for every mode (no K)": get_snow_scores(snow_z12),
            "K's size, true turn, every mode": get_snow_scores(mix(snow_z, snow_z12, "turn")),
            "K's turn, true size, every mode": get_snow_scores(mix(snow_z, snow_z12, "size")),
        }
    ).T.round(3)
)

assert (class_names[802], class_names[803]) == ("snowmobile", "snowplow")
snow_classes = (val_y == 802) | (val_y == 803)
image_groups = {
    "snowy vehicles": (snow_z[is_snowy], snow_z12[is_snowy]),
    "snowplows and snowmobiles in validation": (z6[snow_classes], z12[snow_classes]),
    "all validation": (z6, z12),
}
rows = []
for mode in chosen:
    for label, (start, end) in image_groups.items():
        on = start[:, mode] != 0
        a, b = start[on, mode], end[on, mode]
        pooled = (b * a.conj()).sum()
        rows.append(
            {
                "mode": mode,
                "images": label,
                "keep the mode": int(on.sum()),
                "|λ|": size[mode].item(),
                "angle": angle[mode].item(),
                "size at block 6": a.abs().median().item(),
                "measured gain": (b.abs() / a.abs()).median().item(),
                "measured turn": math.degrees(torch.angle(pooled).item()),
            }
        )
pd.DataFrame(rows).set_index(["mode", "images"]).round(2)
""")

md("""
##### Every mode: coherence, and what removing it does

For each mode kept on at least 0.5% of images:
- **coherence**: how alike its top 16 validation images are, as the mean cosine between their DINOv3 final features. A mode passes if it beats 99% of random sets of 16 images. This is a low bar: 16 images of one class score far higher. So passing means "more alike than chance", not "interpretable".
- **removal**: we set `λ` to 0 and count, over every validation image that keeps the mode, the answers that go from wrong to right (fixed) and from right to wrong (broken).
""")

code("""
true_embeddings = head.embed(val_x[:, 1])


def get_coherence(indices):
    \"\"\"Mean cosine between the DINOv3 features of a set of images, leaving out each image with itself.\"\"\"

    similarity = true_embeddings[indices] @ true_embeddings[indices].T
    n = len(indices)
    return ((similarity.sum() - n) / (n * (n - 1))).item()


random_sets = np.array([get_coherence(torch.randperm(len(val_y))[:16]) for _ in range(1_000)])
class_sets = np.array([get_coherence((val_y == c).nonzero()[:, 0][:16]) for c in range(1_000)])
threshold = np.quantile(random_sets, 0.99)

rows = []
for mode in modes.index.tolist():
    top = z6[:, mode].abs().topk(16).indices
    on = kept[:, mode]
    before, truth = answers[on], val_y[on]
    after = head.classify(edit(z6[on], {mode: 0.0}))
    rows.append(
        {
            "mode": mode,
            "coherence": get_coherence(top),
            "classes in top 16": len(set(val_y[top].tolist())),
            "hidden in top 16": int(hidden[top].sum()),
            "top classes": ", ".join(f"{class_names[c][:16]} {k}" for c, k in Counter(val_y[top].tolist()).most_common(3)),
            "fixed": int(((before != truth) & (after == truth)).sum()),
            "broken": int(((before == truth) & (after != truth)).sum()),
            "answers changed": (before != after).double().mean().item(),
        }
    )
table = modes.join(pd.DataFrame(rows).set_index("mode"))
table["net"] = table["fixed"] - table["broken"]

print(f"Random sets of 16: 99th percentile {threshold:.3f}. Sets of 16 from one class: median {np.median(class_sets):.3f}")
print(f"{(table['coherence'] > threshold).mean():.0%} of {len(table)} modes beat random")
print(f"Summed over modes, removal fixes {table['fixed'].sum()} answers and breaks {table['broken'].sum()}; {int((table['net'] > 0).sum())} modes fix more than they break")

fig, axes = plt.subplots(1, 2, figsize=(14, 3.4))
axes[0].hist(table["coherence"], bins=50, color=BLUE, edgecolor="white", linewidth=0.5)
for value, label, style in [(threshold, "random sets, 99th percentile", "--"), (np.median(class_sets), "one class, median", ":")]:
    axes[0].axvline(value, color=INK, lw=1, ls=style)
    axes[0].annotate(label, (value, axes[0].get_ylim()[1]), xytext=(4, -12), textcoords="offset points", fontsize=8, color=INK)
axes[0].set_xlabel("coherence of a mode's top 16 images")
axes[0].set_ylabel("modes")
feature = table["group"] == "feature"
axes[1].scatter(table.loc[feature, "kept"], table.loc[feature, "net"], s=9, color=BLUE, alpha=0.6, lw=0, label="feature modes")
axes[1].scatter(table.loc[~feature, "kept"], table.loc[~feature, "net"], s=22, color=ORANGE, alpha=0.8, lw=0, label="growth, offset and inert modes")
axes[1].axhline(0, color=INK, lw=0.8)
axes[1].set_xlabel("share of validation images that keep the mode")
axes[1].set_ylabel("fixed minus broken, on removal")
axes[1].legend(fontsize=8, loc="lower left")
fig.tight_layout()
plt.show()

COLUMNS = ["group", "|λ|", "angle", "kept", "coherence", "classes in top 16", "fixed", "broken", "net", "answers changed", "top classes"]
table.groupby(pd.cut(table["classes in top 16"], [0, 3, 8, 16], labels=["1-3 classes", "4-8 classes", "9-16 classes"]), observed=True)["coherence"].agg(modes="size", beat_random=lambda c: (c > threshold).mean()).round(3)
""")

md("""
The modes whose removal breaks the most answers, and the ones whose removal fixes more than it breaks.
""")

code("""
display(table.sort_values("net").head(12)[COLUMNS].round(3))
table.sort_values("net", ascending=False).head(8)[COLUMNS].round(3)
""")

md("""
##### What some modes fire on

The 8 validation images where each mode is most active (by size at block 6), with their ImageNet class underneath. First the most coherent feature modes whose top 16 images span at least 9 classes, picked automatically, so these are modes that group images across classes. Then modes drawn at random from the feature modes, so that the gallery isn't only the best picks.
""")

code("""
showable = table["hidden in top 16"] < 4
print(f"{int((~showable).sum())} modes mostly fire on hidden images and are left out of the galleries")
cross_class = table[(table["classes in top 16"] >= 9) & feature & showable].sort_values("coherence", ascending=False)
picked = cross_class.index[:NUM_COHERENT].tolist()
show_modes(picked, [f"coherence {cross_class.loc[m, 'coherence']:.2f}" for m in picked])
""")

code("""
candidates = table.index[feature & showable].to_numpy()
sample = candidates[torch.randperm(len(candidates))[:NUM_RANDOM].numpy()].tolist()
show_modes(sample, ["randomly sampled"] * NUM_RANDOM)
""")

md("""
The feature modes whose removal breaks the most answers, then the growth modes and the two offset modes with the most energy at block 6. The last four are on for every image, so their top images only say where they are largest.
""")

code("""
needed = table[feature & showable].sort_values("net").index[:6].tolist()
show_modes(needed, [f"removal breaks {table.loc[m, 'broken']}" for m in needed])

constant = [int(m) for m in np.flatnonzero(group == "growth")] + [int(m) for m in share6.argsort(descending=True).tolist() if group[m] == "offset"][:2]
show_modes(constant, [group[m] for m in constant])
""")

md("""
##### Size against turn, over many modes

Is turning a mode a different edit from changing its size? For 40 feature modes drawn at random from those kept on 3% to 30% of images, we apply each edit on up to 2,000 images that keep the mode and measure: how many answers change, the change in accuracy, how far the DINOv3 feature moves (1 - cosine), and the direction of that move against the move made by doubling the mode's size (cosine: 1 means the edit acts like scaling up, -1 like scaling down, 0 means a new direction). Each number is the median over the 40 modes.
""")

code("""
MANY_EDITS = {
    "size x0 (removed)": 0.0,
    "size x0.5": 0.5,
    "size x2": 2.0,
    "turn +45°": turn(45),
    "turn -45°": turn(-45),
    "turn +90°": turn(90),
    "turn -90°": turn(-90),
    "turn 180° (x-1)": -1.0,
}
pool = table.index[feature & (table["kept"] > 0.03) & (table["kept"] < 0.3)].to_numpy()
sampled = pool[torch.randperm(len(pool))[:40].numpy()].tolist()
per_edit = {label: [] for label in MANY_EDITS}
for mode in sampled:
    on = kept[:, mode].nonzero()[:, 0]
    on = on[torch.randperm(len(on))[:2_000]]
    start, truth = z6[on], val_y[on]
    before = head.embed(decode(eigenvalues * start))
    doubled = head.embed(edit(start, {mode: 2.0})) - before
    for label, factor in MANY_EDITS.items():
        prediction = edit(start, {mode: factor})
        after, new_answers = head.embed(prediction), head.classify(prediction)
        per_edit[label].append(
            {
                "answers changed": (new_answers != answers[on]).double().mean().item(),
                "accuracy change": ((new_answers == truth).double().mean() - (answers[on] == truth).double().mean()).item(),
                "feature move (1 - cosine)": (1 - (after * before).sum(dim=-1)).median().item(),
                "direction against size x2 (cosine)": torch.nn.functional.cosine_similarity(after - before, doubled, dim=-1).median().item(),
            }
        )
pd.DataFrame({label: pd.DataFrame(rows).median() for label, rows in per_edit.items()}).T.round(3)
""")

md("""
##### By angle

The modes kept on at least 0.5% of images, grouped by the size of their angle. Medians within each group.
""")

code("""
bins = pd.cut(table["angle"].abs(), [0, 5, 15, 45, 180], include_lowest=True)
by_angle = table.groupby(bins, observed=True).agg(
    modes=("kept", "size"),
    kept=("kept", "median"),
    size=("|λ|", "median"),
    coherence=("coherence", "median"),
    r_squared=("R²", "median"),
    answers_changed_on_removal=("answers changed", "median"),
)
by_angle.round(3)
""")

# The always-on modes and DINOv3's own mistakes
exec(Path(__file__).with_name("dino_extra_cells.py").read_text())

nb = new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbformat.validate(nb)
out = f"/home/nsa325/work/koopmann/second_analysis/dino_modes_{name}.ipynb"
nbformat.write(nb, out)
print(out)
