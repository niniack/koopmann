"""Helpers for demo.ipynb: load a trained KAE, encode images into modes, edit modes, read the prediction, show images."""

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")  # Library notices that would clutter the notebook

import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
from dotenv import load_dotenv
from PIL import Image
from safetensors.torch import load_file
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "tokae"))
load_dotenv(REPO / ".env")
import train  # noqa: E402

torch.set_grad_enabled(False)

ENCODER = "ViT-B-32"
CACHE = Path(__file__).resolve().parent / "cache"
DATA = Path(os.getenv("DATASETS_CACHE"))
CLASS_NAMES = open_clip.IMAGENET_CLASSNAMES
# The snow test: folders of snowy photos, and the ImageNet class each one belongs to
SNOW_FOLDERS = {"armytank": 847, "carwheel": 479, "firetruck": 555, "motorscooter": 670, "racingcar": 751, "schoolbus": 779, "trafficlight": 920}
# Classes no gallery shows: insects, arachnids and their relatives, worms, snails, starfish, sea urchins, honeycombs, apiaries, webs
HIDDEN_CLASSES = torch.tensor([*range(69, 80), 110, 111, *range(113, 117), *range(118, 127), *range(300, 329), 410, 599, 815])
HIDDEN_IMAGES = [22037]

BLUE, ORANGE, MUTED, INK = "#2a78d6", "#eb6834", "#898781", "#52514e"
plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": "#e8e7e3",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
    }
)


############# DATA #############


def load_val():
    """ImageNet validation: states [50000, 2 (after block 6, after block 12), 2304] and labels.

    A state is the CLS token, the mean of the patch tokens and their std.
    """

    val = torch.load(DATA / "imagenet" / f"clip_cls_patch_std_{ENCODER}_val_blocks_5_11.pt")
    return val["states"].float(), val["labels"]


def load_images():
    """ImageNet's validation images, in the order of `load_val`."""

    from datasets import load_dataset

    return load_dataset("ILSVRC/imagenet-1k", split="validation", cache_dir=os.getenv("DATASETS_CACHE"))


def load_snowy():
    """The snow test: states and labels of 129 snowy photos of 7 vehicle classes, and the photos' paths."""

    path = CACHE / "snowy_states.pt"
    if not path.exists():
        build_snowy(path)
    snowy = torch.load(path)
    return snowy["states"].float(), snowy["labels"], snowy["paths"]


def build_snowy(path):
    """Runs CLIP on the snowy photos and keeps the states after blocks 6 and 12, built the way the training cache was."""

    clip, _, preprocess = open_clip.create_model_and_transforms(ENCODER, pretrained="openai", force_quick_gelu=True, cache_dir=f"{os.getenv('HF_HOME')}/open_clip")
    clip.eval()
    states = []

    def keep(_, __, output):
        output = output[0] if isinstance(output, tuple) else output
        patches = output[:, 1:]
        states.append(torch.cat([output[:, 0], patches.mean(dim=1), patches.std(dim=1)], dim=-1))

    for block in (5, 11):
        clip.visual.transformer.resblocks[block].register_forward_hook(keep)
    paths = sorted((DATA / "snowy_imagenet").glob("*/*.jpg"))
    for batch in torch.stack([preprocess(train.to_rgb(Image.open(p))) for p in paths]).split(32):
        clip.encode_image(batch)
    states = torch.cat([torch.stack(states[i : i + 2], dim=1) for i in range(0, len(states), 2)])
    path.parent.mkdir(exist_ok=True)
    torch.save({"states": states, "labels": torch.tensor([SNOW_FOLDERS[p.parent.name] for p in paths]), "paths": [str(p) for p in paths]}, path)


############# MODEL #############


class Kae:
    """A trained Jordan KAE with CLIP's own head on top.

    A mode is one 2x2 block of K, written as one complex number per image. Its eigenvalue λ is shared by all images,
    and K's step from block 6 to block 12 is z12 = λ·z6. Mode 0 is the single real mode.
    """

    def __init__(self, name):
        folder = Path(os.getenv("WEIGHTS_CACHE")) / "tokae"
        stats = torch.load(folder / f"{name}_stats.pt")
        self.mean, self.std = stats["mean"], stats["std"]
        self.model = train.JordanKoopmanAutoencoder(
            256,
            shared_modes=True,
            real_start=stats.get("real_start", False),
            in_features=2304,
            latent_features=2049,
            hidden_config=stats["hidden_config"],
            bias=True,
            batchnorm=False,
            nonlinearity="leaky_relu",
        )
        self.model.load_state_dict(load_file(folder / f"{name}.safetensors"))
        self.model.eval()
        self.eigenvalues = torch.cat([self.model.real_eigenvalue.to(torch.cfloat), torch.complex(self.model.block_real, self.model.block_imag)])

        # CLIP's final layer norm, projection and zero-shot classifier. The classifier is built once (slow on a CPU) and cached
        cached = CACHE / "zero_shot_classes.pt"
        self.head = train.ClipHead(ENCODER, self.mean, self.std, "cpu", zero_shot=not cached.exists())
        if cached.exists():
            self.head.classes = torch.load(cached)
        else:
            cached.parent.mkdir(exist_ok=True)
            torch.save(self.head.classes, cached)

    def encode(self, states):
        """States [images, 2, 2304] -> every mode's value at block 6, complex [images, 1025]. Modes TopK drops are 0."""

        x = (states - self.mean) / self.std
        latent = torch.cat([train.encode_pairs(self.model, batch)[:, 0] for batch in x.split(5_000)])
        return torch.cat([latent[:, :1].to(torch.cfloat), torch.complex(latent[:, 1::2], latent[:, 2::2])], dim=1)

    def answers(self, z6, modes=(), factor=0.0):
        """The KAE's ImageNet answer per image: K's step, decoder, CLIP's head.

        The edit: the eigenvalues of `modes` are multiplied by `factor` (0 removes the modes, 1 leaves them alone).
        """

        eigenvalues = self.eigenvalues.clone()
        eigenvalues[list(modes)] *= factor
        z12 = eigenvalues * z6
        latent = torch.cat([z12[:, :1].real, torch.view_as_real(z12[:, 1:]).flatten(1)], dim=1)
        return torch.cat([self.head.classify(self.model.decode(batch)) for batch in latent.split(5_000)])

    def clip_answers(self, states, top=1):
        """Real CLIP's answer per image, from its own block-12 state (the `top` best answers if top > 1)."""

        log_probs = self.head.get_log_probs((states[:, 1] - self.mean) / self.std)
        return log_probs.argmax(dim=1) if top == 1 else log_probs.topk(top).indices


def load_clip():
    """The real CLIP model (only the demo's timing uses it)."""

    model, _, _ = open_clip.create_model_and_transforms(ENCODER, pretrained="openai", force_quick_gelu=True, cache_dir=f"{os.getenv('HF_HOME')}/open_clip")
    return model.eval()


def get_hidden(kae, states, labels):
    """Images no gallery shows: a hidden class by label or among real CLIP's 5 best answers, plus a few picked by hand."""

    hidden = torch.isin(labels, HIDDEN_CLASSES) | torch.isin(kae.clip_answers(states, top=5), HIDDEN_CLASSES).any(dim=1)
    hidden[HIDDEN_IMAGES] = True
    return hidden


############# STATISTICS #############


def rank_correlation(a, b, fixed=()):
    """Spearman correlation of a and b. With `fixed`, what those variables explain is first taken out of both."""

    a, b = [torch.tensor(rankdata(v)).float() for v in (a, b)]
    if fixed:
        design = torch.stack([torch.ones(len(a)), *[torch.tensor(rankdata(v)).float() for v in fixed]], dim=1)
        a = a - design @ torch.linalg.lstsq(design, a[:, None]).solution[:, 0]
        b = b - design @ torch.linalg.lstsq(design, b[:, None]).solution[:, 0]
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def match_modes(z_a, z_b, min_images=200):
    """Pairs the modes of two KAEs that fire on the same images.

    A pair is kept if each mode is the other's best partner, by the correlation over images of the modes' sizes.
    Only modes on for at least `min_images` images are used. Returns (modes of A, their partners in B, correlation).
    """

    live_a, live_b = [((z != 0).sum(dim=0) >= min_images).nonzero()[1:, 0] for z in (z_a, z_b)]
    sizes = torch.cat([z_a[:, live_a].abs(), z_b[:, live_b].abs()], dim=1)
    alike = torch.corrcoef(sizes.T)[: len(live_a), len(live_a) :]
    best = alike.argmax(dim=1)
    mutual = alike.argmax(dim=0)[best] == torch.arange(len(live_a))
    return live_a[mutual], live_b[best[mutual]], alike.max(dim=1).values[mutual]


def class_tie(values, labels):
    """Per mode: the share of its image-to-image variation that the class label explains (0 none, 1 all).

    NOTE: Assumes 50 images per class, as in ImageNet validation.
    """

    class_means = torch.zeros(1_000, values.shape[1]).index_add_(0, labels, values) / 50
    return class_means.var(dim=0, unbiased=False) / values.var(dim=0, unbiased=False).clamp_min(1e-12)


############# PICTURES #############


def show_images(pictures, captions, title=None):
    """One row of pictures (PIL images or paths) with a caption under each."""

    fig, axes = plt.subplots(1, len(pictures), figsize=(2 * len(pictures), 2.5), squeeze=False)
    for ax, picture, caption in zip(axes[0], pictures, captions):
        picture = (Image.open(picture) if isinstance(picture, (str, Path)) else picture).convert("RGB")
        picture.thumbnail((256, 256))
        ax.imshow(np.asarray(picture))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_xlabel(caption, fontsize=8, color=INK)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    plt.show()
