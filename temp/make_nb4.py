import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

nb.cells = [
md("""### Are any modes class relevant?

Compares the trained KAEs on Imagenette. Nothing is trained here.

For each class, we pick the single best direction on the training set, then measure its one-vs-rest AUC on the validation set. 0.5 is chance and 1 is perfect.

Every basis offers about the same number of candidates, as the best of many random directions can look good by chance."""),
code("""import os
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv
from scipy.stats import rankdata

from koopmann.models import KoopmanAutoencoder
from koopmann.utils import set_seed

load_dotenv("../.env")
torch.set_grad_enabled(False)
set_seed(0)

device = "cpu"  # the GPU is busy with training
ENCODER = "ViT-B-32\""""),
md("##### Load data"),
code("""root = Path(os.getenv("DATASETS_CACHE")) / "imagenette"
train = torch.load(root / f"clip_cls_{ENCODER}_train.pt")
val = torch.load(root / f"clip_cls_{ENCODER}_val.pt")

train_states, train_labels = train["states"].to(device), train["labels"]
val_states, val_labels = val["states"].to(device), val["labels"]

print(tuple(train_states.shape), tuple(val_states.shape))"""),
md("##### Load models"),
code("""MODELS = {
    "dense": ("layer_norm_seed_0", 0),
    "dense + state": ("state_1.0_dim_1024_k_0_seed_0", 0),
    "top 32 of 1024": ("state_1.0_dim_1024_k_32_seed_0", 32),
    "top 1024 of 4096": ("state_0.0_dim_4096_k_1024_seed_0", 1024),
}
save_dir = Path(os.getenv("WEIGHTS_CACHE")) / "tokae"


def sparsify(phi, top_k):
    if not top_k:
        return phi

    threshold = phi.abs().topk(top_k, dim=-1).values[:, -1:]
    return phi * (phi.abs() >= threshold)


def get_observables(kae, states, mean, std, top_k):
    # shape: [samples, layers, latent]
    states = (states - mean) / std
    phi = sparsify(kae.encode(states.flatten(0, 1)), top_k)
    return phi.unflatten(0, states.shape[:2]).double()"""),
md("##### Best single direction per class"),
code("""LAYERS = [2, 5, 8, 11]  # after blocks 3, 6, 9 and 12


def get_ranks(scores):
    # NOTE: Average ranks, as sparse latents are mostly tied at zero
    return torch.from_numpy(rankdata(scores.cpu().numpy(), axis=0))


def get_aucs(ranks, is_class):
    \"\"\"One-vs-rest AUC of every candidate.\"\"\"

    positives = is_class.sum()
    negatives = len(is_class) - positives
    return (ranks[is_class].sum(dim=0) - positives * (positives + 1) / 2) / (positives * negatives)


def get_best_auc(train_scores, val_scores):
    \"\"\"Picks a direction per class on train, returns its mean AUC on val.\"\"\"

    train_ranks, val_ranks = get_ranks(train_scores), get_ranks(val_scores)

    aucs = []
    for c in range(10):
        train_aucs = get_aucs(train_ranks, train_labels == c)
        best = (train_aucs - 0.5).abs().argmax()

        auc = get_aucs(val_ranks[:, best : best + 1], val_labels == c).item()
        aucs.append(auc if train_aucs[best] > 0.5 else 1 - auc)

    return sum(aucs) / len(aucs)


def get_mean_direction_auc(train_phi, val_phi):
    \"\"\"Reference: the full class mean direction, not a single coordinate.\"\"\"

    aucs = []
    for c in range(10):
        is_class = (train_labels == c).to(device)
        direction = train_phi[is_class].mean(dim=0) - train_phi[~is_class].mean(dim=0)
        scores = (val_phi @ direction).unsqueeze(1)
        aucs.append(get_aucs(get_ranks(scores), val_labels == c).item())

    return sum(aucs) / len(aucs)


rows = {}
for name, (suffix, top_k) in MODELS.items():
    run = f"clip_cls_{ENCODER}_{suffix}"
    path = save_dir / f"kae_{run}.safetensors"
    if not path.exists():
        print(f"skipping {name}, not saved yet")
        continue

    kae, _ = KoopmanAutoencoder.load_model(file_path=path)
    kae = kae.to(device).eval()

    stats = torch.load(save_dir / f"stats_{run}.pt")
    mean, std = stats["mean"].to(device), stats["std"].to(device)

    train_phi = get_observables(kae, train_states, mean, std, top_k)
    val_phi = get_observables(kae, val_states, mean, std, top_k)

    size = train_phi.shape[-1]
    _, eigenvectors = torch.linalg.eig(kae.koopman_weights.T.double())

    for layer in LAYERS:
        x, y = train_phi[:, layer], val_phi[:, layer]

        center = x.mean(dim=0)
        _, _, principal = torch.linalg.svd(x - center, full_matrices=False)
        random = torch.randn(size, size, device=device, dtype=torch.double)

        # Real and imaginary parts are separate candidates, so we keep half of each
        modes, val_modes = x.cdouble() @ eigenvectors, y.cdouble() @ eigenvectors
        half = torch.randperm(size, device=device)[: size // 2]

        rows[(name, f"block {layer + 1}")] = {
            "latents": get_best_auc(x, y),
            "K modes": get_best_auc(
                torch.cat([modes[:, half].real, modes[:, half].imag], dim=1),
                torch.cat([val_modes[:, half].real, val_modes[:, half].imag], dim=1),
            ),
            "PCA": get_best_auc((x - center) @ principal.T, (y - center) @ principal.T),
            "random": get_best_auc(x @ random, y @ random),
            "full class direction": get_mean_direction_auc(x, y),
        }

    del kae
    torch.cuda.empty_cache()

pd.DataFrame(rows).T.round(3)"""),
md("""##### Are the sparse latents class specific?

Purity is the share of a latent's activations that come from its most common class. Imagenette is balanced, so 0.1 is chance."""),
code("""rows = {}
for name, (suffix, top_k) in MODELS.items():
    run = f"clip_cls_{ENCODER}_{suffix}"
    path = save_dir / f"kae_{run}.safetensors"
    if not top_k or not path.exists():
        continue

    kae, _ = KoopmanAutoencoder.load_model(file_path=path)
    kae = kae.to(device).eval()

    stats = torch.load(save_dir / f"stats_{run}.pt")
    mean, std = stats["mean"].to(device), stats["std"].to(device)
    val_phi = get_observables(kae, val_states, mean, std, top_k)

    for layer in LAYERS:
        active = val_phi[:, layer] != 0

        # shape: [classes, latents]
        counts = torch.stack([active[val_labels.to(device) == c].sum(dim=0) for c in range(10)])
        totals = counts.sum(dim=0)
        alive = totals >= 20
        purity = counts[:, alive].max(dim=0).values / totals[alive]

        rows[(name, f"block {layer + 1}")] = {
            "dead latents": (totals == 0).float().mean().item(),
            "latents with 20+ uses": alive.sum().item(),
            "mean purity": purity.double().mean().item(),
            "latents over 0.5 purity": (purity > 0.5).sum().item(),
            "latents over 0.9 purity": (purity > 0.9).sum().item(),
        }

pd.DataFrame(rows).T.round(3)"""),
md("""##### A stricter test

Best-of-many AUC is easy on these classes, so here we ask two harder questions.

1. **Few-direction classification.** Pick the `k` most class-informative directions on the training set, then classify all 10 classes with a nearest-centroid rule. Chance is 0.1.
2. **Selectivity.** For the top 50 directions, the share of between-class signal that comes from one class. About 0.1 means every class contributes equally, and about 0.9 means one class only."""),
code("""TOP_K = [5, 10, 20, 50]


def get_class_means(scores, labels):
    # shape: [classes, candidates]
    return torch.stack([scores[labels == c].mean(dim=0) for c in range(10)])


def get_informative(scores, labels):
    \"\"\"Candidates ordered by between-class variance, as a share of total variance.\"\"\"

    means = get_class_means(scores, labels)
    between = (means - scores.mean(dim=0)).pow(2).mean(dim=0)
    return (between / (scores.var(dim=0) + 1e-12)).argsort(descending=True)


def get_accuracy(train_scores, val_scores, order, k, train_y, val_y):
    \"\"\"Nearest-centroid accuracy using the top `k` candidates.\"\"\"

    top = order[:k]
    center, scale = train_scores[:, top].mean(dim=0), train_scores[:, top].std(dim=0) + 1e-12
    x, y = (train_scores[:, top] - center) / scale, (val_scores[:, top] - center) / scale

    predictions = torch.cdist(y, get_class_means(x, train_y)).argmin(dim=1)
    return (predictions == val_y).double().mean().item()


def get_selectivity(val_scores, order, val_y, k=50):
    top = order[:k]
    effects = get_class_means(val_scores[:, top], val_y) - val_scores[:, top].mean(dim=0)
    effects = effects.pow(2)
    return (effects.max(dim=0).values / effects.sum(dim=0)).mean().item()


train_y, val_y = train_labels.to(device), val_labels.to(device)

accuracy_rows, selectivity_rows = {}, {}
for name, (suffix, top_k) in MODELS.items():
    run = f"clip_cls_{ENCODER}_{suffix}"
    path = save_dir / f"kae_{run}.safetensors"
    if not path.exists():
        continue

    kae, _ = KoopmanAutoencoder.load_model(file_path=path)
    kae = kae.to(device).eval()

    stats = torch.load(save_dir / f"stats_{run}.pt")
    mean, std = stats["mean"].to(device), stats["std"].to(device)

    train_phi = get_observables(kae, train_states, mean, std, top_k)
    val_phi = get_observables(kae, val_states, mean, std, top_k)

    size = train_phi.shape[-1]
    _, eigenvectors = torch.linalg.eig(kae.koopman_weights.T.double())

    for layer in LAYERS:
        x, y = train_phi[:, layer], val_phi[:, layer]

        center = x.mean(dim=0)
        _, _, principal = torch.linalg.svd(x - center, full_matrices=False)
        random = torch.randn(size, size, device=device, dtype=torch.double)

        modes, val_modes = x.cdouble() @ eigenvectors, y.cdouble() @ eigenvectors
        half = torch.randperm(size, device=device)[: size // 2]

        candidates = {
            "latents": (x, y),
            "K modes": (
                torch.cat([modes[:, half].real, modes[:, half].imag], dim=1),
                torch.cat([val_modes[:, half].real, val_modes[:, half].imag], dim=1),
            ),
            "PCA": ((x - center) @ principal.T, (y - center) @ principal.T),
            "random": (x @ random, y @ random),
        }

        selectivity = {}
        for basis, (train_scores, val_scores) in candidates.items():
            order = get_informative(train_scores, train_y)
            accuracy_rows[(name, f"block {layer + 1}", basis)] = {
                f"top {k}": get_accuracy(train_scores, val_scores, order, k, train_y, val_y)
                for k in TOP_K
            }
            selectivity[basis] = get_selectivity(val_scores, order, val_y)

        selectivity_rows[(name, f"block {layer + 1}")] = selectivity

    del kae
    torch.cuda.empty_cache()

pd.DataFrame(accuracy_rows).T.round(3)"""),
code("""pd.DataFrame(selectivity_rows).T.round(3)"""),
]
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "second_analysis/09_class_modes.ipynb")
