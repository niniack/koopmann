import json


def cell(kind, source):
    new = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        new.update({"outputs": [], "execution_count": None})
    return new


intro = """### What do K's top eigenmodes fire on?

Uses `b6_novc` and `b6_novc_s1`, two KAEs that differ only in their seed. Nothing is trained here.

Each eigenmode gives every image a coefficient. Two modes from different seeds count as the same mode if those coefficients agree across validation images.

**Match** is the largest canonical correlation between two modes. A complex mode has two real coordinates, so it is a plane rather than a line.

**Class share** is the fraction of a mode's variance explained by the class label.
"""

setup = '''import os
from pathlib import Path

import matplotlib.pyplot as plt
import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Imagenette

from koopmann.models import KoopmanAutoencoder
from koopmann.utils import get_device, set_seed

load_dotenv("../.env")
torch.set_grad_enabled(False)
set_seed(0)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 200)

device = get_device()
CLASSES = ["tench", "springer", "cassette", "chain saw", "church", "horn", "truck", "pump", "golf", "parachute"]
NUM_MODES = 7
NUM_CANDIDATES = 30
NUM_IMAGES = 6
NUM_TOP = 50
'''

load = '''save_dir = Path(os.getenv("WEIGHTS_CACHE")) / "tokae"
root = Path(os.getenv("DATASETS_CACHE")) / "imagenette"

data = torch.load(root / "clip_cls_ViT-B-32_val.pt")
labels = data["labels"].to(device)

thumbnail = transforms.Compose([transforms.Resize(128), transforms.CenterCrop(128)])
dataset = Imagenette(root, split="val", size="full", transform=thumbnail)

# The cached states must follow the dataset's order
assert [label for _, label in dataset._samples] == labels.tolist()
'''

modes = '''def get_modes(name):
    """Eigenvalues, and a function giving each mode's coefficients, largest eigenvalue first."""

    kae, _ = KoopmanAutoencoder.load_model(file_path=save_dir / f"{name}.safetensors")
    kae = kae.to(device).eval()
    stats = torch.load(save_dir / f"{name}_stats.pt", map_location=device)

    # NOTE: On CUDA, `torch.linalg.eig` overwrites transposed inputs, so this must be contiguous
    eigenvalues, eigenvectors = torch.linalg.eig(kae.koopman_weights.T.double().contiguous())

    # Keep one of each conjugate pair
    order = eigenvalues.abs().argsort(descending=True)
    order = order[eigenvalues[order].imag > -1e-9]

    def get_coefficients(states):
        start = (states[:, stats["blocks"][0]].to(device) - stats["mean"]) / stats["std"]
        coefficients = kae.encode(start).double().to(eigenvalues.dtype) @ eigenvectors[:, order]
        return coefficients - coefficients.mean(dim=0)

    return eigenvalues[order], get_coefficients


def get_coordinates(coefficient):
    """One real coordinate for a real mode, two for a complex mode."""

    if coefficient.imag.abs().max() < 1e-9:
        return coefficient.real[:, None]
    return torch.stack([coefficient.real, coefficient.imag], dim=1)


def get_shared(x, y):
    """Weights for the direction in each mode that agrees most across seeds, and how well."""

    (x_basis, x_scale), (y_basis, y_scale) = torch.linalg.qr(x), torch.linalg.qr(y)
    left, values, right = torch.linalg.svd(x_basis.T @ y_basis)
    return torch.linalg.solve(x_scale, left[:, :1]), torch.linalg.solve(y_scale, right[:1].T), values


def get_class_share(x):
    """Fraction of a mode's variance explained by the class label."""

    between = sum((labels == c).sum() * x[labels == c].mean(dim=0).pow(2).sum() for c in range(10))
    return (between / x.pow(2).sum()).item()


def get_class_counts(indices):
    counts = labels[indices].bincount(minlength=10)
    top = counts.argsort(descending=True)[:3].tolist()
    return ", ".join(f"{CLASSES[c]} {counts[c].item()}" for c in top)


values_a, get_coefficients_a = get_modes("b6_novc")
values_b, get_coefficients_b = get_modes("b6_novc_s1")
coefficients_a = get_coefficients_a(data["states"])
coefficients_b = get_coefficients_b(data["states"])

coordinates_a = [get_coordinates(coefficients_a[:, i]) for i in range(NUM_CANDIDATES)]
coordinates_b = [get_coordinates(coefficients_b[:, i]) for i in range(NUM_CANDIDATES)]

matches = torch.tensor(
    [[get_shared(x, y)[2][0].item() for y in coordinates_b] for x in coordinates_a]
)
print(f"Median match over all pairs of the top {NUM_CANDIDATES} modes: {matches.median():.3f}")
'''

table = '''shared = {}
rows = {}
for i in range(NUM_MODES):
    j = matches[i].argmax().item()
    weights_a, weights_b, agreement = get_shared(coordinates_a[i], coordinates_b[j])
    feature_a, feature_b = (coordinates_a[i] @ weights_a)[:, 0], (coordinates_b[j] @ weights_b)[:, 0]
    shared[i] = (j, weights_a, weights_b, feature_a, feature_b)

    high_a, high_b = feature_a.argsort(descending=True)[:NUM_TOP], feature_b.argsort(descending=True)[:NUM_TOP]
    low_a, low_b = feature_a.argsort()[:NUM_TOP], feature_b.argsort()[:NUM_TOP]
    overlap = len(set(high_a.tolist()) & set(high_b.tolist())) + len(set(low_a.tolist()) & set(low_b.tolist()))

    rows[i] = {
        "|eigenvalue| a": values_a[i].abs().item(),
        "|eigenvalue| b": values_b[j].abs().item(),
        "complex": bool(values_a[i].imag.abs() > 1e-9),
        "match": agreement[0].item(),
        "class share": get_class_share(coordinates_a[i]),
        "top overlap": overlap / (2 * NUM_TOP),
        "high end": get_class_counts(high_a),
        "low end": get_class_counts(low_a),
    }

pd.DataFrame(rows).T
'''

images = '''def show_mode(i):
    """Top images at each end of a mode, one row per seed."""

    _, _, _, feature_a, feature_b = shared[i]
    columns = 2 * NUM_IMAGES + 1
    fig, axes = plt.subplots(2, columns, figsize=(columns, 2.6))

    for row, (seed, feature) in enumerate([("seed 0", feature_a), ("seed 1", feature_b)]):
        ranked = feature.argsort(descending=True).tolist()
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

        axes[row, 0].set_ylabel(seed, fontsize=8)

    fig.suptitle(f"Mode {i} (|eigenvalue| {values_a[i].abs():.2f}). Left: high end. Right: low end.", fontsize=9)
    fig.tight_layout()
    plt.show()


for i in range(NUM_MODES):
    show_mode(i)
'''

concepts = '''# NOTE: These concepts were guessed from looking at validation images, so we test on training images
train_data = torch.load(root / "clip_cls_ViT-B-32_train.pt")
train_labels = train_data["labels"].to(device)

resize = transforms.Compose([transforms.Resize(64), transforms.CenterCrop(64), transforms.ToTensor()])
train_images = Imagenette(root, split="train", size="full", transform=resize)
assert [label for _, label in train_images._samples] == train_labels.tolist()


def get_pixel_concepts(images):
    \"\"\"Image statistics that use pixels alone. Images are [batch, 3, height, width], in 0 to 1.\"\"\"

    red, green, blue = images.unbind(dim=1)
    brightest, darkest = images.amax(dim=1), images.amin(dim=1)
    border = torch.cat(
        [darkest[:, :4].flatten(1), darkest[:, -4:].flatten(1), darkest[:, :, :4].flatten(1), darkest[:, :, -4:].flatten(1)],
        dim=1,
    )
    return torch.stack(
        [
            ((brightest - darkest) / (brightest + 1e-6)).mean(dim=[1, 2]),
            ((red - blue > 0.3) & (red > green)).float().mean(dim=[1, 2]),
            (border > 0.9).float().mean(dim=1),
        ],
        dim=1,
    )


pixels = torch.cat([get_pixel_concepts(images) for images, _ in DataLoader(train_images, batch_size=256, num_workers=8)])

# Concepts that pixels can't measure use CLIP's text encoder
clip, _, _ = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai",
    force_quick_gelu=True,
    cache_dir=f"{os.getenv('HF_HOME')}/open_clip",
)
clip = clip.to(device).eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32")

final_cls = train_data["states"][:, -1].to(device)
embeddings = F.normalize(clip.visual.ln_post(final_cls) @ clip.visual.proj, dim=-1)


def get_text_concept(positive, negative):
    text = F.normalize(clip.encode_text(tokenizer([positive, negative]).to(device)), dim=-1)
    return embeddings @ (text[0] - text[1])


concepts = {
    "colourfulness": pixels[:, 0],
    "warm colour": pixels[:, 1],
    "white background": pixels[:, 2],
    "people": get_text_concept("a photo of a group of people", "a photo of an object").cpu(),
    "wide scene": get_text_concept("a wide photo of a landscape", "a close-up photo of an object").cpu(),
}
concept_values = torch.stack(list(concepts.values()), dim=1).double().to(device)
'''

correlations = '''def remove_class_means(x):
    x = x.clone()
    for c in range(10):
        x[train_labels == c] -= x[train_labels == c].mean(dim=0)
    return x


def get_correlations(features):
    \"\"\"Correlation of each feature with each concept, among images of the same class.\"\"\"

    return F.normalize(remove_class_means(features), dim=0).T @ F.normalize(remove_class_means(concept_values), dim=0)


train_coefficients_a = get_coefficients_a(train_data["states"])
train_coefficients_b = get_coefficients_b(train_data["states"])

features_a = torch.cat(
    [get_coordinates(train_coefficients_a[:, i]) @ shared[i][1] for i in range(NUM_MODES)], dim=1
)
features_b = torch.cat(
    [get_coordinates(train_coefficients_b[:, shared[i][0]]) @ shared[i][2] for i in range(NUM_MODES)], dim=1
)

mode_names = [f"mode {i}" for i in range(NUM_MODES)]
correlations_a = get_correlations(features_a).abs()
correlations_b = get_correlations(features_b).abs()

print("Seed 0")
print(pd.DataFrame(correlations_a.cpu(), index=mode_names, columns=concepts).round(2))
print("Seed 1")
print(pd.DataFrame(correlations_b.cpu(), index=mode_names, columns=concepts).round(2))
'''

baselines = '''def fit_linear(keys, values, reg=100.0):
    keys = F.pad(keys, (0, 1), value=1).double()
    gram = keys.T @ keys
    eye = torch.eye(len(gram), device=device, dtype=gram.dtype)
    return torch.linalg.solve(gram + reg * eye, keys.T @ values.double())


def get_principal_directions(x):
    return torch.linalg.svd(x - x.mean(dim=0), full_matrices=False)[2].T


stats = torch.load(save_dir / "b6_novc_stats.pt", map_location=device)
train_x = (train_data["states"].to(device)[:, stats["blocks"]] - stats["mean"]) / stats["std"]
val_x = (data["states"].to(device)[:, stats["blocks"]] - stats["mean"]) / stats["std"]

linear = fit_linear(train_x[:, 0], train_x[:, 1])
train_outputs = F.pad(train_x[:, 0], (0, 1), value=1).double() @ linear
val_outputs = F.pad(val_x[:, 0], (0, 1), value=1).double() @ linear
linear_directions = get_principal_directions(train_outputs)
pca_directions = get_principal_directions(train_x[:, 0].double())

linear_features = (train_outputs - train_outputs.mean(dim=0)) @ linear_directions
pca_features = (train_x[:, 0].double() - train_x[:, 0].double().mean(dim=0)) @ pca_directions

pd.DataFrame(
    {
        "K top 7, seed 0": correlations_a.amax(dim=0).cpu(),
        "K top 7, seed 1": correlations_b.amax(dim=0).cpu(),
        "linear top 7": get_correlations(linear_features[:, :NUM_MODES]).abs().amax(dim=0).cpu(),
        "linear top 20": get_correlations(linear_features[:, :20]).abs().amax(dim=0).cpu(),
        "PCA top 7": get_correlations(pca_features[:, :NUM_MODES]).abs().amax(dim=0).cpu(),
        "PCA top 20": get_correlations(pca_features[:, :20]).abs().amax(dim=0).cpu(),
    },
    index=concepts,
).round(2)
'''

novelty = '''def get_captured(feature, directions):
    \"\"\"Share of a feature's variance that a set of directions can explain, on validation images.\"\"\"

    basis = torch.linalg.qr(directions)[0]
    feature = feature - feature.mean()
    return ((basis.T @ feature).pow(2).sum() / feature.pow(2).sum()).item()


val_linear = (val_outputs - val_outputs.mean(dim=0)) @ linear_directions
val_pca = (val_x[:, 0].double() - val_x[:, 0].double().mean(dim=0)) @ pca_directions

pd.DataFrame(
    {
        mode_names[i]: {
            "linear top 7": get_captured(shared[i][3], val_linear[:, :NUM_MODES]),
            "linear top 20": get_captured(shared[i][3], val_linear[:, :20]),
            "linear top 100": get_captured(shared[i][3], val_linear[:, :100]),
            "PCA top 20": get_captured(shared[i][3], val_pca[:, :20]),
            "PCA top 100": get_captured(shared[i][3], val_pca[:, :100]),
        }
        for i in range(NUM_MODES)
    }
).T.round(2)
'''

nb = {
    "cells": [
        cell("markdown", intro),
        cell("code", setup),
        cell("markdown", "##### Load data\n"),
        cell("code", load),
        cell("markdown", "##### Modes from both seeds\n"),
        cell("code", modes),
        cell("markdown", "##### Do the top modes repeat?\n\n`top overlap` is the share of each end's top 50 images that both seeds agree on. By chance it would be about 0.01.\n"),
        cell("code", table),
        cell("markdown", "##### What each mode fires on\n"),
        cell("code", images),
        cell("markdown", "##### Measuring what the images suggest\n\nFrom the images, these modes may track concepts that cut across classes. We measure each concept on training images, which we didn't look at.\n"),
        cell("code", concepts),
        cell("markdown", "##### Do modes track these concepts within a class?\n\nEach number is the absolute correlation between a mode and a concept, among images of the same class. So class identity can't explain it.\n"),
        cell("code", correlations),
        cell("markdown", "##### Can linear directions find the same concepts?\n\nThe best absolute correlation for each concept, among each method's top directions.\n"),
        cell("code", baselines),
        cell("markdown", "##### How much of each mode do linear directions capture?\n\nA value near 1 means the mode is nothing new. Each cell is the share of the mode's variance explained by that set of directions together.\n"),
        cell("code", novelty),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
json.dump(nb, open("second_analysis/16_mode_images.ipynb", "w"), indent=1)
