import json
import sys

models = sys.argv[1]


def cell(kind, text):
    new = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        new.update({"outputs": [], "execution_count": None})
    return new


intro = """### Are sparse modes better modes?

Every model goes through the same code here. A mode is one real eigenvalue or one complex pair. Nothing is trained.

- **Editing:** remove the top `m` class-separating modes, and measure specificity, as in `15_data_svd`.
- **Concepts:** the 20 neutral concepts from `17_rotation`, measured on training images within each class.

For reference, ICA on the linear map's top 20 directions reached specificity .519, .685 and .603 at `m` of 1, 2 and 5, and a neutral concept score of .303.
"""

setup = f'''import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from safetensors.torch import load_file
from torchvision import transforms
from torchvision.datasets import Imagenette

sys.path.insert(0, "../scripts/tokae")
from train import JordanKoopmanAutoencoder, SparseKoopmanAutoencoder

from koopmann.utils import set_seed

load_dotenv("../.env")
torch.set_grad_enabled(False)
set_seed(0)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 200)

# NOTE: On CPU, so this can run next to training
device = torch.device("cpu")
CLASSES = ["tench", "springer", "cassette", "chain saw", "church", "horn", "truck", "pump", "golf", "parachute"]
NEUTRAL = [
    "the sky", "grass", "water", "snow", "night", "an indoor room", "the outdoors", "a crowd", "a face", "hands",
    "text", "a black and white image", "a close-up", "a landscape", "a white background", "a building", "a vehicle",
    "an animal", "a machine", "metal",
]
NUM_REMOVED = [1, 2, 5, 20]
NUM_SHOWN = 7
NUM_IMAGES = 6
NUM_TOP = 50
NUM_CANDIDATES = 30

MODELS = {models}
'''

load = '''save_dir = Path(os.getenv("WEIGHTS_CACHE")) / "tokae"
root = Path(os.getenv("DATASETS_CACHE")) / "imagenette"

stats = torch.load(save_dir / "b6_novc_stats.pt", map_location=device)


def load_split(split):
    data = torch.load(root / f"clip_cls_ViT-B-32_{split}.pt")
    states = (data["states"][:, stats["blocks"]] - stats["mean"]) / stats["std"]
    return states, data["labels"], data["states"][:, -1]


train_x, train_y, train_final = load_split("train")
val_x, val_y, _ = load_split("val")

thumbnail = transforms.Compose([transforms.Resize(128), transforms.CenterCrop(128)])
dataset = Imagenette(root, split="val", size="full", transform=thumbnail)
assert [label for _, label in dataset._samples] == val_y.tolist()


def fit_linear(keys, values, reg=100.0):
    keys = F.pad(keys, (0, 1), value=1).double()
    gram = keys.T @ keys
    eye = torch.eye(len(gram), dtype=gram.dtype)
    return torch.linalg.solve(gram + reg * eye, keys.T @ values.double())


classifier = fit_linear(train_x[:, 1], F.one_hot(train_y, 10).float())


def get_class_accs(states):
    predictions = (F.pad(states.double(), (0, 1), value=1) @ classifier).argmax(dim=1)
    correct = (predictions == val_y).double()
    return torch.stack([correct[val_y == c].mean() for c in range(10)])
'''

concepts = '''clip, _, _ = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai",
    force_quick_gelu=True,
    cache_dir=f"{os.getenv('HF_HOME')}/open_clip",
)
clip = clip.to(device).eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32")

embeddings = F.normalize(clip.visual.ln_post(train_final) @ clip.visual.proj, dim=-1)
plain = F.normalize(clip.encode_text(tokenizer(["a photo"])), dim=-1)[0]
prompts = tokenizer([f"a photo of {concept}" for concept in NEUTRAL])
neutral_values = (embeddings @ (F.normalize(clip.encode_text(prompts), dim=-1) - plain).T).double()


def remove_class_means(x):
    x = x.clone()
    for c in range(10):
        x[train_y == c] -= x[train_y == c].mean(dim=0)
    return x


neutral_centered = F.normalize(remove_class_means(neutral_values), dim=0)


def get_neutral_best(features):
    """The best absolute correlation with each neutral concept, among images of the same class."""

    return (F.normalize(remove_class_means(features), dim=0).T @ neutral_centered).abs().amax(dim=0)
'''

modes = '''def load_model(name, top_k, jordan):
    kwargs = {
        "in_features": 768,
        "latent_features": 1025 if jordan else 1024,
        "hidden_config": [1024],
        "bias": True,
        "batchnorm": False,
        "nonlinearity": "leaky_relu",
    }
    if jordan:
        kae = JordanKoopmanAutoencoder(top_k, **kwargs)
    else:
        kae = SparseKoopmanAutoencoder(top_k=top_k, k_steps=1, **kwargs)

    kae.load_state_dict(load_file(save_dir / f"{name}.safetensors"))
    return kae.to(device).eval()


def get_jordan_modes(kae):
    """K is already in modal form, so each coordinate or pair of coordinates is a mode."""

    eigenvalues = torch.cat(
        [
            kae.real_eigenvalue.double().to(torch.cdouble),
            torch.complex(kae.block_real.double(), kae.block_imag.double()),
        ]
    )

    def encode(x):
        phi = kae.encode(x).double()
        blocks = phi[:, 1:].unflatten(-1, (-1, 2))
        return torch.cat([phi[:, :1].to(torch.cdouble), torch.complex(blocks[..., 0], blocks[..., 1])], dim=1)

    def decode(modes):
        blocks = torch.stack([modes[:, 1:].real, modes[:, 1:].imag], dim=-1).flatten(1)
        return kae.decode(torch.cat([modes[:, :1].real, blocks], dim=1).float())

    return eigenvalues, encode, decode


def get_dense_modes(kae):
    """Eigenmodes of a full K. A complex pair is one mode, so we keep one and count it twice."""

    eigenvalues, vectors = torch.linalg.eig(kae.koopman_weights.T.double().contiguous())
    inverse = torch.linalg.inv(vectors)
    keep = eigenvalues.imag > -1e-9
    counts = 1.0 + (eigenvalues[keep].imag.abs() > 1e-9).double()

    def encode(x):
        return kae.encode(x).double().to(vectors.dtype) @ vectors[:, keep]

    def decode(modes):
        return kae.decode(((modes * counts) @ inverse[keep]).real.float())

    return eigenvalues[keep], encode, decode


def get_scores(modes, c):
    """How well each mode separates class `c`, on training images."""

    is_class = train_y == c
    gap = modes[is_class].mean(dim=0) - modes[~is_class].mean(dim=0)
    spread = (modes - modes.mean(dim=0)).abs().pow(2).mean(dim=0).sqrt()
    return gap.abs() / (spread + 1e-12)


def get_feature(mode):
    """One real feature for a mode: the main direction of its coefficients."""

    coordinates = torch.stack([mode.real, mode.imag], dim=1)
    coordinates = coordinates - coordinates.mean(dim=0)
    return coordinates @ torch.linalg.svd(coordinates, full_matrices=False)[2][0]


def get_class_share(feature):
    x = feature - feature.mean()
    between = sum((val_y == c).sum() * x[val_y == c].mean().pow(2) for c in range(10))
    return (between / x.pow(2).sum()).item()


def get_class_counts(indices):
    counts = val_y[indices].bincount(minlength=10)
    top = counts.argsort(descending=True)[:3].tolist()
    return ", ".join(f"{CLASSES[c]} {counts[c].item()}" for c in top if counts[c] > 0)


def analyze(name, top_k, jordan):
    kae = load_model(name, top_k, jordan)
    eigenvalues, encode, decode = (get_jordan_modes if jordan else get_dense_modes)(kae)

    train_modes, val_modes = encode(train_x[:, 0]), encode(val_x[:, 0])
    mean = train_modes.mean(dim=0)
    energy = (train_modes - mean).abs().pow(2).mean(dim=0) * eigenvalues.abs().pow(2)
    order = energy.argsort(descending=True) if jordan else eigenvalues.abs().argsort(descending=True)

    # With nothing removed, modes must rebuild the model's own prediction
    base_prediction = kae.decode(kae.koopman_forward(kae.encode(val_x[:, 0]), 1))
    assert (decode(val_modes * eigenvalues) - base_prediction).abs().max() < 1e-3
    base = get_class_accs(base_prediction)

    def edit(removed, fill):
        modes = val_modes.clone()
        modes[:, removed] = fill[removed]
        return decode(modes * eigenvalues)

    def run(m, fill, random=False):
        target_drops, other_drops = [], []
        for c in range(10):
            if random:
                removed = order[:20][torch.randperm(20)[:m]]
            else:
                removed = get_scores(train_modes, c).argsort(descending=True)[:m]

            drops = base - get_class_accs(edit(removed, fill))
            target_drops.append(drops[c])
            other_drops.append((drops.sum() - drops[c]) / 9)

        target, other = torch.stack(target_drops).mean().item(), torch.stack(other_drops).mean().item()
        return {"target drop": target, "other drop": other, "specificity": target - other}

    editing = {}
    for m in NUM_REMOVED:
        editing[(m, "zeroed")] = run(m, torch.zeros_like(mean))
        editing[(m, "set to mean")] = run(m, mean)
        editing[(m, "random of top 20")] = run(m, mean, random=True)

    per_class = {}
    for c, class_name in enumerate(CLASSES):
        top = get_scores(train_modes, c).argmax()
        drops = base - get_class_accs(edit(top[None], mean))
        per_class[class_name] = {
            "mode": top.item(),
            "|eigenvalue|": eigenvalues[top].abs().item(),
            "target": drops[c].item(),
            "other": ((drops.sum() - drops[c]) / 9).item(),
        }

    train_features = torch.stack([get_feature(train_modes[:, i]) for i in order[:20]], dim=1)
    val_features = torch.stack([get_feature(val_modes[:, i]) for i in order[:NUM_SHOWN]], dim=1)

    summary = {}
    for rank, i in enumerate(order[:NUM_SHOWN].tolist()):
        feature = val_features[:, rank]
        ranked = feature.argsort(descending=True)
        summary[i] = {
            "|eigenvalue|": eigenvalues[i].abs().item(),
            "angle": eigenvalues[i].angle().abs().item(),
            "active": (val_modes[:, i].abs() > 0).double().mean().item(),
            "class share": get_class_share(feature),
            "high end": get_class_counts(ranked[:NUM_TOP]),
            "low end": get_class_counts(ranked[-NUM_TOP:]),
        }

    return {
        "editing": pd.DataFrame(editing).T,
        "per class": pd.DataFrame(per_class).T,
        "summary": pd.DataFrame(summary).T,
        "neutral top 7": get_neutral_best(train_features[:, :7]),
        "neutral top 20": get_neutral_best(train_features),
        "val features": val_features,
        "val modes": val_modes[:, order[:NUM_CANDIDATES]],
        "order": order,
    }


results = {name: analyze(name, **config) for name, config in MODELS.items()}
'''

summary = '''pd.DataFrame({name: result["editing"]["specificity"] for name, result in results.items()}).unstack().round(3)
'''

neutral = '''pd.DataFrame(
    {
        name: {
            "neutral, top 7": result["neutral top 7"].mean().item(),
            "neutral, top 20": result["neutral top 20"].mean().item(),
        }
        for name, result in results.items()
    }
).T.round(3)
'''

details = '''for name, result in results.items():
    print(name)
    print(result["summary"].to_string())
    print(result["per class"].astype(float).round(3).to_string())
    print()
'''

images = '''def show_modes(name):
    """Top images at each end of a model's main modes."""

    features = results[name]["val features"]
    columns = 2 * NUM_IMAGES + 1
    fig, axes = plt.subplots(NUM_SHOWN, columns, figsize=(columns, 1.25 * NUM_SHOWN))

    for row in range(NUM_SHOWN):
        ranked = features[:, row].argsort(descending=True).tolist()
        picks = ranked[:NUM_IMAGES] + [None] + ranked[-NUM_IMAGES:][::-1]

        for column, index in enumerate(picks):
            ax = axes[row, column]
            ax.set_xticks([])
            ax.set_yticks([])
            if index is None:
                ax.axis("off")
                continue
            image, label = dataset[index]
            ax.imshow(image)
            ax.set_title(CLASSES[label], fontsize=7, pad=2)

        axes[row, 0].set_ylabel(f"mode {results[name]['order'][row].item()}", fontsize=8)

    fig.suptitle(f"{name}. Left: high end. Right: low end.", fontsize=9)
    fig.tight_layout()
    plt.show()


for name, config in MODELS.items():
    if config["top_k"]:
        show_modes(name)
'''

seeds = '''def get_coordinates(mode):
    \"\"\"One real coordinate for a real mode, two for a complex mode.\"\"\"

    coordinates = torch.stack([mode.real, mode.imag], dim=1)
    coordinates = coordinates - coordinates.mean(dim=0)
    return coordinates if mode.imag.abs().max() > 1e-9 else coordinates[:, :1]


def get_match(x, y):
    \"\"\"Largest canonical correlation between two modes, across validation images.\"\"\"

    return torch.linalg.svdvals(torch.linalg.qr(x)[0].T @ torch.linalg.qr(y)[0]).max().item()


def compare_seeds(first, second):
    first_modes = [get_coordinates(mode) for mode in results[first]["val modes"].T]
    second_modes = [get_coordinates(mode) for mode in results[second]["val modes"].T]
    matches = torch.tensor([[get_match(x, y) for y in second_modes] for x in first_modes])
    return {"best match, top 7": matches[:NUM_SHOWN].amax(dim=1).tolist(), "median": matches.median().item()}


pairs = {
    "dense K": ("b6_novc", "b6_novc_s1"),
    "Jordan K + TopK": ("b6_jordan128", "b6_jordan128_s1"),
    "Jordan K": ("b6_jordan", "b6_jordan_s1"),
}
seed_results = {name: compare_seeds(*pair) for name, pair in pairs.items()}

pd.DataFrame(
    {
        name: {
            **{f"mode {i}": value for i, value in enumerate(result["best match, top 7"])},
            "median over all pairs": result["median"],
        }
        for name, result in seed_results.items()
    }
).round(3)
'''

nb = {
    "cells": [
        cell("markdown", intro),
        cell("code", setup),
        cell("markdown", "##### Load data\n"),
        cell("code", load),
        cell("markdown", "##### Concepts\n"),
        cell("code", concepts),
        cell("markdown", "##### Modes\n"),
        cell("code", modes),
        cell("markdown", "##### Editing specificity\n"),
        cell("code", summary),
        cell("markdown", "##### Neutral concepts\n\nThe mean, over 20 concepts, of the best correlation among each model's top modes. Linear's top 20 directions scored .332.\n"),
        cell("code", neutral),
        cell("markdown", "##### Do the main modes repeat across seeds?\n\nFor each of the first seed's main modes, the best match among the second seed's top 30.\n"),
        cell("code", seeds),
        cell("markdown", "##### Main modes, and the best mode for each class\n"),
        cell("code", details),
        cell("markdown", "##### What the main modes fire on\n"),
        cell("code", images),
    ],
    "metadata": json.load(open("second_analysis/16_mode_images.ipynb"))["metadata"],
    "nbformat": 4,
    "nbformat_minor": 5,
}
json.dump(nb, open("second_analysis/18_sparse_modes.ipynb", "w"), indent=1)
