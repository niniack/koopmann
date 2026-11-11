import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

nb.cells = [
md("""### Removing a snow direction from CLIP embeddings

A baseline with no KAE and no training. We estimate a snow direction from a few images, project it out of the image embedding, and classify as usual.

Clean images and guardrails come from the ImageNet validation set."""),
code("""import io
import os
from pathlib import Path

import open_clip
import pandas as pd
import pyarrow as pa
import torch
from dotenv import load_dotenv
from PIL import Image

from koopmann.utils import get_device, set_seed

load_dotenv("../.env")
torch.set_grad_enabled(False)
set_seed(0)

device = get_device()"""),
md("##### Load data"),
code("""class_names = list(open_clip.IMAGENET_CLASSNAMES)
snow_indices = [847, 479, 555, 670, 751, 779, 920]
guard_indices = {"snowmobile": 802, "snowplow": 803}
assert all(class_names[index] == name for name, index in guard_indices.items())

# Snowy images, without the two exact duplicates
data_root = Path(os.getenv("DATASETS_CACHE")) / "snowy_imagenet"
folder_to_index = dict(zip(sorted(p.name for p in data_root.iterdir()), snow_indices))
paths = [path for path in sorted(data_root.glob("*/*.jpg")) if "(1)" not in path.name]
snowy_images = [Image.open(path).convert("RGB") for path in paths]
snowy_labels = torch.tensor([folder_to_index[path.parent.name] for path in paths])

print(f"{len(snowy_images)} snowy images")
print({folder: class_names[index] for folder, index in folder_to_index.items()})"""),
code("""def load_imagenet(wanted, random_every=25):
    \"\"\"Collects validation images from `wanted` classes, plus a random subset.\"\"\"

    root = Path(os.getenv("DATASETS_CACHE")) / "ILSVRC___imagenet-1k"
    files = sorted(root.glob("default/0.0.0/*/imagenet-1k-validation-*.arrow"))

    images, labels, is_random = [], [], []
    count = 0
    for file in files:
        with pa.memory_map(str(file)) as source:
            for batch in pa.ipc.open_stream(source):
                for image, label in zip(batch["image"].to_pylist(), batch["label"].to_pylist()):
                    count += 1
                    random = count % random_every == 0
                    if label in wanted or random:
                        images.append(Image.open(io.BytesIO(image["bytes"])).convert("RGB"))
                        labels.append(label)
                        is_random.append(random)

    return images, torch.tensor(labels), torch.tensor(is_random)


wanted = set(snow_indices) | set(guard_indices.values())
imagenet_images, imagenet_labels, is_random = load_imagenet(wanted)

print(f"{len(imagenet_images)} ImageNet images, {is_random.sum()} of them random")"""),
md("##### Get embeddings"),
code("""ENCODERS = ["ViT-B-32", "ViT-B-16", "ViT-L-14"]


def get_features(model, preprocess, images, batch_size=64):
    features = []
    for i in range(0, len(images), batch_size):
        batch = torch.stack([preprocess(image) for image in images[i : i + batch_size]])
        features.append(model.encode_image(batch.to(device)).float())

    features = torch.cat(features)
    return features / features.norm(dim=-1, keepdim=True)


def get_head(model, tokenizer):
    weights = []
    for name in class_names:
        templates = open_clip.OPENAI_IMAGENET_TEMPLATES
        tokens = tokenizer([template(name) for template in templates]).to(device)
        embeddings = model.encode_text(tokens).float()
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        mean_embedding = embeddings.mean(dim=0)
        weights.append(mean_embedding / mean_embedding.norm())

    return torch.stack(weights)


embeddings = {}
for encoder in ENCODERS:
    model, _, preprocess = open_clip.create_model_and_transforms(
        encoder,
        pretrained="openai",
        force_quick_gelu=True,  # OpenAI's weights were trained with QuickGELU
        cache_dir=f"{os.getenv('HF_HOME')}/open_clip",
    )
    model = model.to(device).eval()

    embeddings[encoder] = {
        "snowy": get_features(model, preprocess, snowy_images).cpu(),
        "imagenet": get_features(model, preprocess, imagenet_images).cpu(),
        "head": get_head(model, open_clip.get_tokenizer(encoder)).cpu(),
    }

    del model
    torch.cuda.empty_cache()"""),
md("""##### Remove the snow direction

For each class we take the difference between its snowy and clean means, using 5 images of each. The top singular vectors of those differences give the snow subspace."""),
code("""NUM_REFERENCES = 5
NUM_DRAWS = 20
RANKS = [1, 3, 7]


def pick(labels, index, count):
    (candidates,) = torch.where(labels == index)
    return candidates[torch.randperm(len(candidates))[:count]]


def get_basis(snowy, clean, clean_labels, snowy_refs):
    differences = []
    for index in snow_indices:
        clean_mean = clean[pick(clean_labels, index, NUM_REFERENCES)].mean(dim=0)
        differences.append(snowy[snowy_refs[index]].mean(dim=0) - clean_mean)

    _, _, vectors = torch.linalg.svd(torch.stack(differences), full_matrices=False)
    return vectors


def project_out(features, basis):
    return features - (features @ basis.T) @ basis


def get_acc(features, labels, head):
    return ((features @ head.T).argmax(dim=1) == labels).float()


def get_class_mean(correct, labels):
    return torch.stack([correct[labels == index].mean() for index in snow_indices]).mean()


def run_draw(encoder):
    snowy = embeddings[encoder]["snowy"]
    imagenet = embeddings[encoder]["imagenet"]
    head = embeddings[encoder]["head"]

    snowy_refs = {index: pick(snowy_labels, index, NUM_REFERENCES) for index in snow_indices}
    is_test = torch.ones(len(snowy), dtype=torch.bool)
    is_test[torch.cat(list(snowy_refs.values()))] = False

    basis = get_basis(snowy, imagenet[~is_random], imagenet_labels[~is_random], snowy_refs)

    rows = {}
    for rank in [0] + RANKS:
        def edit(features):
            return project_out(features, basis[:rank]) if rank else features

        correct = get_acc(edit(snowy[is_test]), snowy_labels[is_test], head)
        row = {"snowy": get_class_mean(correct, snowy_labels[is_test]).item() * 100}

        for name, index in guard_indices.items():
            is_guard = (imagenet_labels == index) & ~is_random
            correct = get_acc(edit(imagenet[is_guard]), imagenet_labels[is_guard], head)
            row[f"real {name}"] = correct.mean().item() * 100

        correct = get_acc(edit(imagenet[is_random]), imagenet_labels[is_random], head)
        row["random ImageNet"] = correct.mean().item() * 100
        rows[rank] = row

    return rows, basis


results = {}
top_classes = {}
for encoder in ENCODERS:
    draws = [run_draw(encoder) for _ in range(NUM_DRAWS)]
    for rank in [0] + RANKS:
        frame = pd.DataFrame([rows[rank] for rows, _ in draws])
        results[(encoder, "before" if rank == 0 else f"rank {rank}")] = frame.mean().round(1)

    # Which classes does the top direction point towards?
    scores = embeddings[encoder]["head"] @ draws[0][1][0]
    scores = scores * torch.sign(scores.max() + scores.min())
    top_classes[encoder] = [class_names[i] for i in scores.topk(8).indices]

pd.DataFrame(results).T"""),
md("##### What does the top direction look like?"),
code("""pd.DataFrame(top_classes)"""),
md("""##### Compare to the paper

Their numbers are for reference only, since the test images may differ. The ceiling is hand-fixing every snow-themed mistake."""),
code("""reference = {
    "ViT-B-32": {"class vector, latent": 72.2, "class vector, mapped": 76.2, "dirmatch": 73.9, "ceiling": 90.2},
    "ViT-B-16": {"class vector, latent": 69.7, "class vector, mapped": 72.7, "dirmatch": 72.0, "ceiling": 88.4},
    "ViT-L-14": {"class vector, latent": 71.3, "class vector, mapped": 78.3, "dirmatch": 74.6, "ceiling": 85.4},
}
table = pd.DataFrame(reference).T
for rank in ["before"] + [f"rank {rank}" for rank in RANKS]:
    table[rank] = [results[(encoder, rank)]["snowy"] for encoder in table.index]

table"""),
]
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "second_analysis/07_snow_direction.ipynb")
