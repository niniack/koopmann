import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

nb.cells = [
md("""### Does editing K do something targeted?

Uses `b6_30v`, which predicts block 12 from block 6. Nothing is trained here.

For each class, we rank directions of the operator by how well they separate that class on training images. We then remove the top `m` and measure the change in validation accuracy.

**Specificity** is the target class's accuracy drop minus the mean drop of the other classes."""),
code("""import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from dotenv import load_dotenv

from koopmann.models import KoopmanAutoencoder
from koopmann.utils import get_device, set_seed

load_dotenv("../.env")
torch.set_grad_enabled(False)
set_seed(0)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

device = get_device()
CLASSES = ["tench", "springer", "cassette", "chain saw", "church", "horn", "truck", "pump", "golf", "parachute"]
NUM_REMOVED = [1, 5, 20, 50, 100]"""),
md("##### Load model and data"),
code("""save_dir = Path(os.getenv("WEIGHTS_CACHE")) / "tokae"
root = Path(os.getenv("DATASETS_CACHE")) / "imagenette"

kae, _ = KoopmanAutoencoder.load_model(file_path=save_dir / "b6_30v.safetensors")
kae = kae.to(device).eval()
stats = torch.load(save_dir / "b6_30v_stats.pt", map_location=device)


def load_split(split):
    data = torch.load(root / f"clip_cls_ViT-B-32_{split}.pt")
    states = data["states"].to(device)[:, stats["blocks"]]
    return (states - stats["mean"]) / stats["std"], data["labels"].to(device)


train_x, train_y = load_split("train")
val_x, val_y = load_split("val")"""),
md("##### References"),
code("""def fit_linear(keys, values, reg=100.0):
    keys = F.pad(keys, (0, 1), value=1).double()
    gram = keys.T @ keys
    eye = torch.eye(len(gram), device=device, dtype=gram.dtype)
    return torch.linalg.solve(gram + reg * eye, keys.T @ values.double())


def classify(states):
    return (F.pad(states.double(), (0, 1), value=1) @ classifier).argmax(dim=1)


def get_class_accs(states):
    correct = (classify(states) == val_y).double()
    return torch.stack([correct[val_y == c].mean() for c in range(10)])


classifier = fit_linear(train_x[:, 1], F.one_hot(train_y, 10).float())

# Linear map from start to end
linear = fit_linear(train_x[:, 0], train_x[:, 1])
linear_weight, linear_bias = linear[:-1], linear[-1]

# Observables at the start
train_phi = kae.encode(train_x[:, 0]).double()
val_phi = kae.encode(val_x[:, 0]).double()
operator = kae.koopman_weights.T.double()


def decode(phi):
    return kae.decode(phi.float())


kae_accs = get_class_accs(decode(val_phi @ operator))
linear_accs = get_class_accs(val_x[:, 0].double() @ linear_weight + linear_bias)

pd.DataFrame(
    {"KAE": kae_accs.cpu(), "linear": linear_accs.cpu(), "true end": get_class_accs(val_x[:, 1]).cpu()},
    index=CLASSES,
).round(3)"""),
md("##### Directions"),
code("""k_left, k_values, k_right = torch.linalg.svd(operator)
eigenvalues, eigenvectors = torch.linalg.eig(operator)
eigenvectors_inverse = torch.linalg.inv(eigenvectors)
l_left, l_values, l_right = torch.linalg.svd(linear_weight)


def get_scores(coefficients, c):
    \"\"\"How well each direction separates class `c`, on training images.\"\"\"

    is_class = train_y == c
    gap = coefficients[is_class].mean(dim=0) - coefficients[~is_class].mean(dim=0)
    spread = (coefficients - coefficients.mean(dim=0)).abs().pow(2).mean(dim=0).sqrt()
    return gap.abs() / (spread + 1e-12)


def get_keep(num_directions, removed):
    keep = torch.ones(num_directions, device=device, dtype=torch.double)
    keep[removed] = 0
    return keep


def edit_kae_svd(removed):
    keep = get_keep(len(k_values), removed)
    return decode((((val_phi @ k_left) * k_values * keep) @ k_right))


def edit_kae_eigen(removed):
    keep = get_keep(len(eigenvalues), removed).to(eigenvalues.dtype)
    modes = val_phi.to(eigenvalues.dtype) @ eigenvectors
    return decode(((modes * eigenvalues * keep) @ eigenvectors_inverse).real)


def edit_linear_svd(removed):
    keep = get_keep(len(l_values), removed)
    coefficients = val_x[:, 0].double() @ l_left
    return (coefficients * l_values * keep) @ l_right + linear_bias


def run(edit, coefficients, base_accs, m, random=False):
    \"\"\"Mean target drop and mean other-class drop, over all classes.\"\"\"

    target_drops, other_drops = [], []
    for c in range(10):
        if random:
            removed = torch.randperm(coefficients.shape[1], device=device)[:m]
        else:
            removed = get_scores(coefficients, c).argsort(descending=True)[:m]

        drops = base_accs - get_class_accs(edit(removed))
        target_drops.append(drops[c])
        other_drops.append((drops.sum() - drops[c]) / 9)

    target, other = torch.stack(target_drops).mean(), torch.stack(other_drops).mean()
    return {"target drop": target.item(), "other drop": other.item(), "specificity": (target - other).item()}"""),
md("##### Results"),
code("""k_coefficients = train_phi @ k_left
eigen_coefficients = train_phi.to(eigenvalues.dtype) @ eigenvectors
l_coefficients = train_x[:, 0].double() @ l_left

rows = {}
for m in NUM_REMOVED:
    rows[(m, "K singular")] = run(edit_kae_svd, k_coefficients, kae_accs, m)
    rows[(m, "K eigen")] = run(edit_kae_eigen, eigen_coefficients, kae_accs, m)
    rows[(m, "K random")] = run(edit_kae_svd, k_coefficients, kae_accs, m, random=True)
    rows[(m, "linear singular")] = run(edit_linear_svd, l_coefficients, linear_accs, m)
    rows[(m, "linear random")] = run(edit_linear_svd, l_coefficients, linear_accs, m, random=True)

pd.DataFrame(rows).T.round(3)"""),
md("##### Per class, removing the top 20 singular directions"),
code("""rows = {}
for c, name in enumerate(CLASSES):
    removed = get_scores(k_coefficients, c).argsort(descending=True)[:20]
    k_drops = kae_accs - get_class_accs(edit_kae_svd(removed))

    removed = get_scores(l_coefficients, c).argsort(descending=True)[:20]
    l_drops = linear_accs - get_class_accs(edit_linear_svd(removed))

    rows[name] = {
        "K target": k_drops[c].item(),
        "K other": ((k_drops.sum() - k_drops[c]) / 9).item(),
        "linear target": l_drops[c].item(),
        "linear other": ((l_drops.sum() - l_drops[c]) / 9).item(),
    }

pd.DataFrame(rows).T.round(3)"""),
]
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "second_analysis/13_editing.ipynb")
