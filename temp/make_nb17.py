import json

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


md("""
### Do KAEs trained on ImageNet separate features?

On Imagenette, KAEs memorized 9,469 images. These models trained on 1.28M. We ask three things, on ImageNet's own validation images, which no model saw:

1. **Prediction:** how well does each model predict block 12 from block 6?
2. **Black-and-white:** does the steering result from Imagenette hold up?
3. **Snow:** does any feature separate the 129 snowy images from normal images of the same seven classes? Does removing it fix their predictions?

Features are read two ways: raw modes, and ICA on the operator's outputs. A plain linear map gets ICA too, which shows what lifting adds. Nothing is trained here.
""")

code('''
import os
import sys
from pathlib import Path

import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from PIL import Image
from safetensors.torch import load_file
from sklearn.decomposition import FastICA
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

sys.path.insert(0, "../scripts/tokae")
from train import ImageNet, JordanKoopmanAutoencoder, SparseKoopmanAutoencoder

from koopmann.utils import set_seed

load_dotenv("../.env")
torch.set_grad_enabled(False)
set_seed(0)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

device = torch.device("cpu")
NUM_CLASSES = 1_000
NUM_FEATURES = 100
NUM_CONTROLS = 20
SAMPLE_STRIDE = 25
GREY_THRESHOLD = 0.02

MODELS = {
    "im_dense": {"top_k": 0, "jordan": False},
    "im_jordan": {"top_k": 0, "jordan": True},
    "im_jordan128": {"top_k": 128, "jordan": True},
}
PANEL = [
    "the sky", "grass", "water", "night", "an indoor room", "the outdoors", "a crowd", "a face", "hands",
    "text", "a close-up", "a landscape", "a white background", "a building", "a vehicle", "an animal", "a machine",
    "metal",
]
SNOW_FOLDERS = {
    "armytank": 847,
    "carwheel": 479,
    "firetruck": 555,
    "motorscooter": 670,
    "racingcar": 751,
    "schoolbus": 779,
    "trafficlight": 920,
}
''')

md("##### Load data\n\nFeatures are fit on every 25th training image, about 51,000. The classifier and the linear map use all of them.")

code('''
save_dir = Path(os.getenv("WEIGHTS_CACHE")) / "tokae"
root = Path(os.getenv("DATASETS_CACHE")) / "imagenet"
stats = torch.load(save_dir / "im_dense_stats.pt", map_location=device)


def load_split(split):
    data = torch.load(root / f"clip_cls_ViT-B-32_{split}_blocks_5_11.pt")
    return (data["states"] - stats["mean"]) / stats["std"], data["labels"]


def fit_linear(keys, values, reg=100.0):
    keys = F.pad(keys, (0, 1), value=1).double()
    gram = keys.T @ keys
    eye = torch.eye(len(gram), dtype=gram.dtype)
    return torch.linalg.solve(gram + reg * eye, keys.T @ values.double())


def apply_linear(operator, x):
    return F.pad(x.double(), (0, 1), value=1) @ operator


all_x, all_y = load_split("train")
val_x, val_y = load_split("val")

classifier = fit_linear(all_x[:, 1], F.one_hot(all_y, NUM_CLASSES).float())
linear = fit_linear(all_x[:, 0], all_x[:, 1])

sample = torch.arange(0, len(all_x), SAMPLE_STRIDE)
train_x, train_y = all_x[sample], all_y[sample]
del all_x, all_y


def predict_class(states):
    return apply_linear(classifier, states).argmax(dim=1)
''')

md("##### Models")

code('''
def load_model(name, top_k, jordan):
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


def get_mode_bundle(name, top_k, jordan):
    """A model's top modes, each as one feature, and how to remove each one."""

    kae = load_model(name, top_k, jordan)
    eigenvalues, encode, decode = (get_jordan_modes if jordan else get_dense_modes)(kae)
    train_modes = encode(train_x[:, 0])

    mean = train_modes.mean(dim=0)
    energy = (train_modes - mean).abs().pow(2).mean(dim=0) * eigenvalues.abs().pow(2)
    order = energy.argsort(descending=True) if jordan else eigenvalues.abs().argsort(descending=True)
    order = order[:NUM_FEATURES]

    planes = torch.stack([train_modes[:, order].real, train_modes[:, order].imag], dim=-1)
    centers = planes.mean(dim=0)
    directions = torch.stack([torch.linalg.svd(planes[:, i] - centers[i], full_matrices=False)[2][0] for i in range(len(order))])

    def features(x):
        modes = encode(x)[:, order]
        return ((torch.stack([modes.real, modes.imag], dim=-1) - centers) * directions).sum(dim=-1)

    def edit(x, i, strength=0.0):
        modes = encode(x)
        modes[:, order[i]] = mean[order[i]] + strength * (modes[:, order[i]] - mean[order[i]])
        return decode(modes * eigenvalues)

    def predict(x):
        return decode(encode(x) * eigenvalues)

    return features, edit, predict, [f"mode {i}" for i in order.tolist()]


def get_ica_bundle(forward, finish):
    """ICA on an operator's outputs. The operator still predicts, and ICA reads features from it."""

    train_out = forward(train_x[:, 0])
    mean = train_out.mean(dim=0)
    directions = torch.linalg.svd(train_out - mean, full_matrices=False)[2].T[:, :NUM_FEATURES]

    ica = FastICA(whiten="unit-variance", max_iter=2000, random_state=0).fit(((train_out - mean) @ directions).numpy())
    unmixing = torch.tensor(ica.components_, dtype=torch.double).T
    mixing = torch.linalg.inv(unmixing)

    def features(x):
        return ((forward(x) - mean) @ directions) @ unmixing

    def edit(x, i, strength=0.0):
        change = (features(x)[:, i : i + 1] @ mixing[i : i + 1]) @ directions.T
        return finish(forward(x) - (1 - strength) * change)

    def predict(x):
        return finish(forward(x))

    return features, edit, predict, [f"ica {i}" for i in range(NUM_FEATURES)]


def get_kae_ica(name, top_k, jordan):
    kae = load_model(name, top_k, jordan)
    return get_ica_bundle(lambda x: kae.koopman_forward(kae.encode(x), 1).double(), lambda x: kae.decode(x.float()))


bundles = {name: get_mode_bundle(name, **config) for name, config in MODELS.items()}
bundles.update({f"{name} ICA": get_kae_ica(name, **config) for name, config in MODELS.items()})
bundles["linear ICA"] = get_ica_bundle(lambda x: apply_linear(linear, x), lambda x: x)
''')

md("##### Prediction\n\nFVU is the fraction of block 12's variance a model misses, so lower is better.")

code('''
def get_fvu(prediction, target):
    return ((prediction.double() - target.double()).pow(2).sum() / (target.double() - target.double().mean(dim=0)).pow(2).sum()).item()


rows = {}
for name in [*MODELS, "linear ICA"]:
    prediction = bundles[name][2](val_x[:, 0])
    rows[name.replace(" ICA", "")] = {"FVU": get_fvu(prediction, val_x[:, 1]), "accuracy": (predict_class(prediction) == val_y).double().mean().item()}
rows["true block 12"] = {"FVU": 0.0, "accuracy": (predict_class(val_x[:, 1]) == val_y).double().mean().item()}

pd.DataFrame(rows).T.round(4)
''')

md("##### Which images are black-and-white?\n\nMeasured from pixels alone.")

code('''
def to_rgb(image):
    return image.convert("RGB")


resize = transforms.Compose([to_rgb, transforms.Resize(64), transforms.CenterCrop(64), transforms.ToTensor()])


def get_colour_amount(images):
    """How far pixels are from grey. Black-and-white images are close to zero."""

    amounts = []
    for batch, _ in DataLoader(images, batch_size=256, num_workers=16):
        red, green, blue = batch.unbind(dim=1)
        amounts.append(((red - green).abs() + (green - blue).abs()).mean(dim=[1, 2]))
    return torch.cat(amounts)


train_colour = get_colour_amount(Subset(ImageNet("train", resize), sample.tolist()))
val_colour = get_colour_amount(ImageNet("val", resize))
train_grey, val_grey = train_colour < GREY_THRESHOLD, val_colour < GREY_THRESHOLD

print(f"Black-and-white: {train_grey.sum()} of {len(train_grey)} training, {val_grey.sum()} of {len(val_grey)} validation")
''')

md("##### Measures\n\n`gap closed` is how far the target images move toward the other images on a CLIP text score. 1 means the gap is gone. Side effects are measured only on images an edit changes: the target images, and the 10% of other images it changes most.")

code('''
clip, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai",
    force_quick_gelu=True,
    cache_dir=f"{os.getenv('HF_HOME')}/open_clip",
)
clip = clip.to(device).eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32")


def encode_text(prompts):
    return F.normalize(clip.encode_text(tokenizer(prompts)), dim=-1)


grey_direction = (encode_text(["a black and white photo"]) - encode_text(["a color photo"]))[0]
snow_direction = (encode_text(["a photo in the snow"]) - encode_text(["a photo"]))[0]
panel_directions = (encode_text([f"a photo of {concept}" for concept in PANEL]) - encode_text(["a photo"])).T


def to_embeddings(states):
    raw = states.float() * stats["std"] + stats["mean"]
    return F.normalize(clip.visual.ln_post(raw) @ clip.visual.proj, dim=-1)


def remove_class_means(x, labels):
    sums = torch.zeros(NUM_CLASSES, x.shape[1], dtype=x.dtype).index_add_(0, labels, x)
    counts = torch.bincount(labels, minlength=NUM_CLASSES).clamp_min(1)[:, None]
    return x - (sums / counts)[labels]


def get_auc(features, is_positive):
    """For each feature, the chance a positive image ranks above a negative one, in either direction."""

    ranks = features.argsort(dim=0).argsort(dim=0).double() + 1
    positives, negatives = is_positive.sum(), (~is_positive).sum()
    auc = (ranks[is_positive].sum(dim=0) - positives * (positives + 1) / 2) / (positives * negatives)
    return torch.maximum(auc, 1 - auc)


def measure(base, edited, labels, target, rest, direction):
    """How much of the gap closes, and what else moves on the images that change."""

    base_embeddings, edited_embeddings = to_embeddings(base), to_embeddings(edited)
    base_scores, edited_scores = base_embeddings @ direction, edited_embeddings @ direction
    gap = base_scores[target].mean() - base_scores[rest].mean()
    closed = (base_scores[target].mean() - edited_scores[target].mean()) / gap

    base_panel = base_embeddings @ panel_directions
    change = (edited_embeddings @ panel_directions - base_panel).pow(2)
    variance = remove_class_means(base_panel, labels).pow(2).mean(dim=0)

    # The other images this edit changes most
    size = (edited.double() - base.double()).norm(dim=1) / base.double().norm(dim=1)
    others = rest.nonzero()[:, 0]
    touched = others[size[others].argsort(descending=True)[: max(len(others) // 10, 1)]]

    before, after = predict_class(base), predict_class(edited)
    flips = before != after

    def accuracy(predictions, mask):
        return (predictions[mask] == labels[mask]).double().mean().item()

    return {
        "gap closed": closed.item(),
        "size ratio": (size[target].mean() / size[rest].mean().clamp_min(1e-12)).item(),
        "other, target": (change[target].mean(dim=0) / variance).median().item(),
        "other, rest": (change[touched].mean(dim=0) / variance).median().item(),
        "flips, target": 100 * flips[target].double().mean().item(),
        "flips, rest": 100 * flips[touched].double().mean().item(),
        "accuracy, target": accuracy(before, target),
        "edited, target": accuracy(after, target),
        "accuracy, rest": accuracy(before, rest),
        "edited, rest": accuracy(after, rest),
    }
''')

md("##### Black-and-white\n\nThe best feature of each method is chosen on training images, then scored on validation images.")

code('''
rows, controls = {}, {}
for name, (features, edit, predict, labels) in bundles.items():
    auc = get_auc(features(train_x[:, 0]), train_grey)
    best = auc.argmax().item()
    base = predict(val_x[:, 0])
    validation = get_auc(features(val_x[:, 0])[:, best : best + 1], val_grey)[0].item()

    def score(i):
        return measure(base, edit(val_x[:, 0], i), val_y, val_grey, ~val_grey, grey_direction)

    rows[name] = {"feature": labels[best], "train AUC": auc.max().item(), "validation AUC": validation, **score(best)}

    others = [i for i in torch.randperm(NUM_FEATURES)[:NUM_CONTROLS].tolist() if i != best]
    other_scores = pd.DataFrame([score(i) for i in others])
    controls[name] = {
        "gap closed": rows[name]["gap closed"],
        "next best of sample": other_scores["gap closed"].max(),
        "median of sample": other_scores["gap closed"].median(),
    }

pd.DataFrame(rows).T
''')

code('''
pd.DataFrame(controls).T.round(3)
''')

md("##### Snow\n\nThe 129 snowy images against ImageNet's normal validation images of the same seven classes. With so few images, each feature is chosen on one half and scored on the other, then the halves swap.")

code('''
snow_root = Path(os.getenv("DATASETS_CACHE")) / "snowy_imagenet"
paths = sorted(snow_root.glob("*/*.jpg"))
snow_y = torch.tensor([SNOW_FOLDERS[path.parent.name] for path in paths])

states = []
hooks = [
    clip.visual.transformer.resblocks[block].register_forward_hook(lambda _, __, output: states.append(output[:, 0]))
    for block in stats["blocks"]
]
clip.encode_image(torch.stack([preprocess(Image.open(path)) for path in paths]))
for hook in hooks:
    hook.remove()

snow_x = (torch.stack(states, dim=1) - stats["mean"]) / stats["std"]

normal = torch.isin(val_y, torch.tensor(list(SNOW_FOLDERS.values()))).nonzero()[:, 0]
x = torch.cat([val_x[normal], snow_x])
y = torch.cat([val_y[normal], snow_y])
is_snow = torch.cat([torch.zeros(len(normal), dtype=torch.bool), torch.ones(len(snow_x), dtype=torch.bool)])
fold = torch.randperm(len(x)) % 2

true_correct = predict_class(x[:, 1]) == y
true_scores = to_embeddings(x[:, 1]) @ snow_direction
print(f"{len(paths)} snowy images, {len(normal)} normal images")
print(f"True block 12 accuracy: snowy {true_correct[is_snow].double().mean():.3f}, normal {true_correct[~is_snow].double().mean():.3f}")
print(f"True block 12 snow gap: {true_scores[is_snow].mean() - true_scores[~is_snow].mean():.3f}")
''')

code('''
rows = {}
for name, (features, edit, predict, labels) in bundles.items():
    scores = features(x[:, 0])
    base = predict(x[:, 0])
    halves = []
    for half in [0, 1]:
        pick, test = fold == half, fold != half
        best = get_auc(scores[pick], is_snow[pick]).argmax().item()
        auc = get_auc(scores[test][:, best : best + 1], is_snow[test])[0].item()
        result = measure(base[test], edit(x[:, 0], best)[test], y[test], is_snow[test], ~is_snow[test], snow_direction)
        halves.append({"AUC": auc, **result})
        print(name, "half", half, labels[best], f"AUC {auc:.3f}")
    rows[name] = pd.DataFrame(halves).mean().to_dict()

pd.DataFrame(rows).T.round(3)
''')

notebook = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
json.dump(notebook, open("/home/nsa325/work/koopmann/second_analysis/21_imagenet_features.ipynb", "w"), indent=1)
print("written", len(cells))
