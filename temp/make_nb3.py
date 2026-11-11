import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

nb.cells = [
md("""### Is snow concentrated in a few Koopman modes?

Uses the KAE trained on CLIP ViT-B/32 CLS tokens from Imagenette (one shared K, per-layer standardization). Nothing is trained here.

We take the snowy minus clean difference in observable space, and ask how few modes of K can explain it.

NOTE: Snowy images come from Flickr and clean images from ImageNet, so this difference can also contain dataset style."""),
code("""import io
import os
from pathlib import Path

import open_clip
import pandas as pd
import pyarrow as pa
import torch
from dotenv import load_dotenv
from PIL import Image
from sklearn.metrics import roc_auc_score

from koopmann.models import KoopmanAutoencoder
from koopmann.utils import get_device, set_seed

load_dotenv("../.env")
torch.set_grad_enabled(False)
set_seed(0)

device = get_device()
ENCODER = "ViT-B-32"
RUN = f"clip_cls_{ENCODER}_layer_norm_seed_0\""""),
md("##### Load data"),
code("""snow_indices = [847, 479, 555, 670, 751, 779, 920]

# Snowy images, without the two exact duplicates
data_root = Path(os.getenv("DATASETS_CACHE")) / "snowy_imagenet"
folder_to_index = dict(zip(sorted(p.name for p in data_root.iterdir()), snow_indices))
paths = [path for path in sorted(data_root.glob("*/*.jpg")) if "(1)" not in path.name]
snowy_images = [Image.open(path).convert("RGB") for path in paths]
snowy_labels = torch.tensor([folder_to_index[path.parent.name] for path in paths])


def load_imagenet(wanted):
    \"\"\"Collects validation images from `wanted` classes.\"\"\"

    root = Path(os.getenv("DATASETS_CACHE")) / "ILSVRC___imagenet-1k"
    files = sorted(root.glob("default/0.0.0/*/imagenet-1k-validation-*.arrow"))

    images, labels = [], []
    for file in files:
        with pa.memory_map(str(file)) as source:
            for batch in pa.ipc.open_stream(source):
                for image, label in zip(batch["image"].to_pylist(), batch["label"].to_pylist()):
                    if label in wanted:
                        images.append(Image.open(io.BytesIO(image["bytes"])).convert("RGB"))
                        labels.append(label)

    return images, torch.tensor(labels)


clean_images, clean_labels = load_imagenet(set(snow_indices))
print(f"{len(snowy_images)} snowy, {len(clean_images)} clean")"""),
md("##### Get CLS states"),
code("""model, _, preprocess = open_clip.create_model_and_transforms(
    ENCODER,
    pretrained="openai",
    force_quick_gelu=True,  # OpenAI's weights were trained with QuickGELU
    cache_dir=f"{os.getenv('HF_HOME')}/open_clip",
)
model = model.to(device).eval()


def get_states(images, batch_size=64):
    \"\"\"The CLS token after every block.\"\"\"

    states = []
    hooks = [
        block.register_forward_hook(lambda _, __, output: states.append(output[:, 0]))
        for block in model.visual.transformer.resblocks
    ]

    batches = []
    for i in range(0, len(images), batch_size):
        states.clear()
        batch = torch.stack([preprocess(image) for image in images[i : i + batch_size]])
        model.encode_image(batch.to(device))
        batches.append(torch.stack(states, dim=1))

    for hook in hooks:
        hook.remove()

    # shape: [samples, layers, neurons]
    return torch.cat(batches)


snowy_states = get_states(snowy_images)
clean_states = get_states(clean_images)

del model
torch.cuda.empty_cache()"""),
md("##### Load the KAE"),
code("""save_dir = Path(os.getenv("WEIGHTS_CACHE")) / "tokae"
kae, _ = KoopmanAutoencoder.load_model(file_path=save_dir / f"kae_{RUN}.safetensors")
kae = kae.to(device).eval()

stats = torch.load(save_dir / f"stats_{RUN}.pt")
mean, std = stats["mean"].to(device), stats["std"].to(device)

imagenette_path = Path(os.getenv("DATASETS_CACHE")) / "imagenette" / f"clip_cls_{ENCODER}_train.pt"
imagenette_states = torch.load(imagenette_path)["states"].to(device)


def get_observables(states):
    # shape: [samples, layers, latent]
    states = (states - mean) / std
    return kae.encode(states.flatten(0, 1)).unflatten(0, states.shape[:2]).double()


snowy_phi = get_observables(snowy_states)
clean_phi = get_observables(clean_states)
imagenette_phi = get_observables(imagenette_states)"""),
md("##### Spectrum of K"),
code("""# NOTE: `koopman_forward` multiplies rows by K^T, so these are its modes
operator = kae.koopman_weights.T.double()
eigenvalues, eigenvectors = torch.linalg.eig(operator)
magnitudes = eigenvalues.abs()

print(f"condition number of the eigenvectors: {torch.linalg.cond(eigenvectors).item():.1f}")
print(f"real eigenvalues: {(eigenvalues.imag.abs() < 1e-8).sum().item()} of {len(eigenvalues)}")
pd.Series(magnitudes.cpu().numpy()).describe(percentiles=[0.01, 0.1, 0.5, 0.9, 0.99]).round(3)"""),
md("""##### How few modes explain the snow difference?

Each mode's size is its contribution to the difference. We keep the largest `k` modes and measure how much of the difference they explain.

Modes of K come in conjugate pairs, so `k` is even."""),
code("""TOP_K = [2, 10, 20, 50, 100, 200]
LAYERS = [2, 5, 8, 11]  # after blocks 3, 6, 9 and 12


def get_difference(layer):
    \"\"\"Snowy minus clean, per class, then averaged so the object cancels out.\"\"\"

    differences = [
        snowy_phi[snowy_labels.to(device) == index, layer].mean(dim=0)
        - clean_phi[clean_labels.to(device) == index, layer].mean(dim=0)
        for index in snow_indices
    ]
    return torch.stack(differences).mean(dim=0)


def get_explained(difference, basis):
    \"\"\"Share of the difference explained by its `k` largest modes.\"\"\"

    inverse = torch.linalg.inv(basis)
    coefficients = difference.cdouble() @ basis
    sizes = coefficients.abs() * inverse.norm(dim=1)
    order = sizes.argsort(descending=True)

    explained = {}
    for k in TOP_K:
        top = order[:k]
        part = (coefficients[top] @ inverse[top]).real
        explained[k] = 1 - (difference - part).pow(2).sum().item() / difference.pow(2).sum().item()

    return explained


def get_participation(difference, basis):
    \"\"\"Effective number of modes. 1 means one mode, `size` means all of them equally.\"\"\"

    inverse = torch.linalg.inv(basis)
    sizes = ((difference.cdouble() @ basis).abs() * inverse.norm(dim=1)).pow(2)
    return (sizes.sum().pow(2) / sizes.pow(2).sum()).item()


size = eigenvectors.shape[0]
rows = {}
for layer in LAYERS:
    difference = get_difference(layer)
    _, _, principal = torch.linalg.svd(
        imagenette_phi[:, layer] - imagenette_phi[:, layer].mean(dim=0), full_matrices=False
    )
    random, _ = torch.linalg.qr(torch.randn(size, size, device=device, dtype=torch.double))

    bases = {
        "K modes": eigenvectors,
        "PCA": principal.T.cdouble(),
        "random": random.cdouble(),
    }
    for name, basis in bases.items():
        row = {f"top {k}": v for k, v in get_explained(difference, basis).items()}
        row["effective modes"] = get_participation(difference, basis)
        rows[(f"block {layer + 1}", name)] = row

pd.DataFrame(rows).T.round(3)"""),
md("""##### Can one mode detect snow?

We pick the top mode on half of the images and test it on the other half. An AUC of 1 is perfect and 0.5 is chance. The last column uses the full difference direction, as a reference."""),
code("""def get_auc(scores, labels):
    return roc_auc_score(labels.cpu().numpy(), scores.cpu().numpy())


def get_half_difference(phi_snowy, labels_snowy, phi_clean, labels_clean):
    differences = [
        phi_snowy[labels_snowy == index].mean(dim=0) - phi_clean[labels_clean == index].mean(dim=0)
        for index in snow_indices
    ]
    return torch.stack(differences).mean(dim=0)


def get_detection(layer, basis, k=2):
    \"\"\"Fits on even images, tests on odd images.\"\"\"

    snowy, clean = snowy_phi[:, layer], clean_phi[:, layer]
    s_labels, c_labels = snowy_labels.to(device), clean_labels.to(device)

    difference = get_half_difference(snowy[::2], s_labels[::2], clean[::2], c_labels[::2])
    inverse = torch.linalg.inv(basis)
    coefficients = difference.cdouble() @ basis
    top = (coefficients.abs() * inverse.norm(dim=1)).argsort(descending=True)[:k]

    # Project onto the part of the difference held by the top modes
    direction = (coefficients[top] @ inverse[top]).real

    test = torch.cat([snowy[1::2], clean[1::2]])
    labels = torch.cat([torch.ones(len(snowy[1::2])), torch.zeros(len(clean[1::2]))])

    return get_auc(test @ direction, labels), get_auc(test @ difference, labels)


rows = {}
for layer in LAYERS:
    _, _, principal = torch.linalg.svd(
        imagenette_phi[:, layer] - imagenette_phi[:, layer].mean(dim=0), full_matrices=False
    )
    random, _ = torch.linalg.qr(torch.randn(size, size, device=device, dtype=torch.double))

    k_modes, full = get_detection(layer, eigenvectors)
    rows[f"block {layer + 1}"] = {
        "K modes": k_modes,
        "PCA": get_detection(layer, principal.T.cdouble())[0],
        "random": get_detection(layer, random.cdouble())[0],
        "full difference": full,
    }

pd.DataFrame(rows).T.round(3)"""),
md("""##### Singular directions of K

Singular vectors are orthogonal, so they can't cancel each other like the eigenmodes do. They come in pairs: K sends each input direction to a different output direction.

Same two tests as above."""),
code("""# NOTE: `koopman_forward` multiplies rows by K^T, so `left` holds the input directions
left, singular_values, right = torch.linalg.svd(operator)

print(f"singular values: max {singular_values.max():.3f}, median {singular_values.median():.3f}")
print(f"condition number of K: {torch.linalg.cond(operator).item():.1f}")

rows = {}
for layer in LAYERS:
    difference = get_difference(layer)
    _, _, principal = torch.linalg.svd(
        imagenette_phi[:, layer] - imagenette_phi[:, layer].mean(dim=0), full_matrices=False
    )
    random, _ = torch.linalg.qr(torch.randn(size, size, device=device, dtype=torch.double))

    bases = {
        "K singular, input": left.cdouble(),
        "K singular, output": right.T.cdouble(),
        "PCA": principal.T.cdouble(),
        "random": random.cdouble(),
    }
    for name, basis in bases.items():
        row = {f"top {k}": v for k, v in get_explained(difference, basis).items()}
        row["effective modes"] = get_participation(difference, basis)
        row["detection"] = get_detection(layer, basis)[0]
        rows[(f"block {layer + 1}", name)] = row

pd.DataFrame(rows).T.round(3)"""),
]
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "second_analysis/08_snow_modes.ipynb")
