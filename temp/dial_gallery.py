"""
Pictures for the dial test, with retrieval as the decoder: what does the KAE's prediction look like after a mode is turned?

No picture is painted. The KAE's (edited) prediction of block 12 is read through DINOv3's final layer norm, and the
validation photos whose real DINOv3 embedding is closest to it are shown (the photo itself is excluded). One sheet per
case (mode, source direction, destination direction): a row per source photo with
  the photo | nearest photos to the unedited prediction | to the prediction with the mode turned to the destination's
  direction and made stronger | to the prediction with the mode made as strong without a turn (the control).
Images of insects, arachnids and their relatives are never shown (label or one of DINOv3's top 5 answers is such a class).

DINOv3 run im_dino_w21_wd, ImageNet validation. Usage: dial_gallery.py <gain> <mode:source degrees:destination degrees,...>
"""

import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import open_clip
import torch
from datasets import load_dataset
from torchvision import transforms

import inversion
from inversion import to_complex, to_latent, train

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
torch.set_grad_enabled(False)
INK, MUTED = "#52514e", "#898781"
SLICES, PHOTOS, NEAREST = 6, 4, 3
HIDDEN_CLASSES = torch.tensor([*range(69, 80), 110, 111, *range(113, 117), *range(118, 127), *range(300, 329), 410, 599, 815])
gain = float(sys.argv[1])
cases = [tuple(int(v) for v in case.split(":")) for case in sys.argv[2].split(",")]
out = Path("dial")
out.mkdir(exist_ok=True)
names = open_clip.IMAGENET_CLASSNAMES
normalize = torch.nn.functional.normalize

kae, stats = inversion.load_kae()
states, labels = train.get_states(inversion.ENCODER, "imagenet", "val", "cpu", [5, 11], tokens="cls_patch_std", extract=False)
states = (states.float() - stats["mean"]) / stats["std"]
head = train.DinoHead(inversion.ENCODER, stats["mean"], stats["std"], "cpu", zero_shot=True)
hidden = torch.isin(labels, HIDDEN_CLASSES) | torch.isin(head.get_log_probs(states[:, 1]).topk(5).indices, HIDDEN_CLASSES).any(dim=1)
z6 = torch.cat([to_complex(train.encode_pairs(kae, states[first : first + 5000])[:, 0]) for first in range(0, len(states), 5000)])
lam = torch.cat([kae.real_eigenvalue.to(torch.cfloat), torch.complex(kae.block_real, kae.block_imag)])
# shape: [images, width], unit length; DINOv3's final embedding of the true block-12 state
true_embed = torch.cat([normalize(head.features(states[first : first + 5000, 1]), dim=-1) for first in range(0, len(states), 5000)])
images = load_dataset("ILSVRC/imagenet-1k", split="validation", cache_dir=os.getenv("DATASETS_CACHE"))
to_thumb = transforms.Compose([transforms.Resize(160), transforms.CenterCrop(160), train.to_rgb])


def get_family(mode, degrees):
    """Rows of the showable images that keep the mode with a direction in the 60° slice around `degrees`, strongest first."""

    rows = ((z6[:, mode] != 0) & ~hidden).nonzero()[:, 0]
    size, phase = z6[rows, mode].abs(), torch.rad2deg(torch.angle(z6[rows, mode])) % 360
    slice_of = ((phase + 180 / SLICES) % 360 / (360 / SLICES)).long()
    pick = slice_of == degrees // (360 // SLICES)
    return rows[pick][size[pick].argsort(descending=True)]


def get_nearest(latent, own):
    """The showable validation photos closest to the prediction decoded from `latent`, the photo itself excluded."""

    embed = normalize(head.features(kae.decode(to_latent(latent[None]))), dim=-1)[0]
    scores = true_embed @ embed
    scores[hidden] = -1
    scores[own] = -1
    return scores.topk(NEAREST).indices.tolist()


for mode, source, destination in cases:
    family, target = get_family(mode, source), get_family(mode, destination)
    described = ", ".join(f"{names[c]} ({n} of 30)" for c, n in Counter(labels[target[:30]].tolist()).most_common(3))
    from_described = ", ".join(f"{names[c]} ({n} of 30)" for c, n in Counter(labels[family[:30]].tolist()).most_common(3))
    step = (destination - source) % 360
    groups = ["nearest photos to the KAE's prediction", f"mode turned by {step}° and made {gain:g}x stronger", f"mode made {gain:g}x stronger, no turn (control)"]
    fig, axes = plt.subplots(PHOTOS, 1 + NEAREST * len(groups), figsize=(1.45 * (1 + NEAREST * len(groups)), 1.45 * PHOTOS + 1.6), squeeze=False)
    report = []
    for r, row in enumerate(family[:PHOTOS].tolist()):
        latent = lam * z6[row]
        turned, stronger = latent.clone(), latent.clone()
        turned[mode] = gain * inversion.turn(step) * latent[mode]
        stronger[mode] = gain * latent[mode]
        shown = [[row], get_nearest(latent, row), get_nearest(turned, row), get_nearest(stronger, row)]
        report.append(f"    photo {row} ({names[labels[row]]}): unedited -> {[names[labels[i]] for i in shown[1]]}; turned -> {[names[labels[i]] for i in shown[2]]}; control -> {[names[labels[i]] for i in shown[3]]}")
        for c, index in enumerate([i for group in shown for i in group]):
            ax = axes[r, c]
            ax.imshow(to_thumb(images[int(index)]["image"]))
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_visible(False)
            if r == 0 and c == 0:
                ax.set_title("the photo", fontsize=8.5, color=INK)
            if r == 0 and c > 0 and (c - 1) % NEAREST == 1:
                ax.set_title(groups[(c - 1) // NEAREST], fontsize=8.5, color=INK)
    fig.suptitle(
        f"Mode {mode}: photos whose direction in the mode's plane is {source}° ({from_described}),\n"
        f"turned to {destination}°, where the mode holds: {described}",
        fontsize=9.5, color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / f"dial_{mode}_{source}_{destination}_x{gain:g}.png", dpi=100, facecolor="white")
    plt.close(fig)
    print(f"  mode {mode}: {source}° ({from_described}) to {destination}° ({described}), gain {gain:g}")
    print("\n".join(report), flush=True)
print("done")
