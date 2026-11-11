import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

nb.cells = [
md("""### Which sparse latents are shared between classes?

A latent is **used** by a class if it's active in at least `threshold` of that class's validation images. It's **distinct** if no other class uses it, and **shared** otherwise. Nothing is trained here."""),
code("""import os
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv

from koopmann.models import KoopmanAutoencoder

load_dotenv("../.env")
torch.set_grad_enabled(False)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

ENCODER = "ViT-B-32"
CLASSES = ["tench", "springer", "cassette", "chain saw", "church", "horn", "truck", "pump", "golf", "parachute"]
MODELS = {
    "top 32 of 1024": ("state_1.0_dim_1024_k_32_seed_0", 32),
    "top 1024 of 4096": ("state_0.0_dim_4096_k_1024_seed_0", 1024),
}
LAYERS = [5, 8, 11]  # after blocks 6, 9 and 12
THRESHOLDS = [0.5, 0.2]"""),
md("##### Load data"),
code("""val = torch.load(Path(os.getenv("DATASETS_CACHE")) / "imagenette" / f"clip_cls_{ENCODER}_val.pt")
states, labels = val["states"], val["labels"]
save_dir = Path(os.getenv("WEIGHTS_CACHE")) / "tokae"


def get_active(suffix, top_k):
    \"\"\"Which latents are active for each image. Shape: [samples, layers, latent].\"\"\"

    run = f"clip_cls_{ENCODER}_{suffix}"
    kae, _ = KoopmanAutoencoder.load_model(file_path=save_dir / f"kae_{run}.safetensors")
    stats = torch.load(save_dir / f"stats_{run}.pt", map_location="cpu")

    x = (states - stats["mean"]) / stats["std"]
    phi = kae.eval().encode(x.flatten(0, 1))
    threshold = phi.abs().topk(top_k, dim=-1).values[:, -1:]

    return (phi.abs() >= threshold).unflatten(0, x.shape[:2])


active = {name: get_active(*model) for name, model in MODELS.items()}"""),
md("##### Shared and distinct latents, per class"),
code("""def get_usage(active_layer, threshold):
    # shape: [classes, latents]
    rates = torch.stack([active_layer[labels == c].float().mean(dim=0) for c in range(10)])
    return rates >= threshold


rows = {}
for name, layers in active.items():
    for layer in LAYERS:
        for threshold in THRESHOLDS:
            used = get_usage(layers[:, layer], threshold)
            num_classes = used.sum(dim=0)

            distinct = (used & (num_classes == 1)).sum(dim=1).float()
            total = used.sum(dim=1).float()

            rows[(name, f"block {layer + 1}", threshold)] = {
                "used per class": total.mean().item(),
                "% distinct": (100 * distinct / total.clamp(min=1)).mean().item(),
                "% shared": (100 * (total - distinct) / total.clamp(min=1)).mean().item(),
                "classes per used latent": num_classes[num_classes > 0].float().mean().item(),
                "used by all 10": (num_classes == 10).sum().item(),
                "unused": (num_classes == 0).sum().item(),
            }

pd.DataFrame(rows).T.round(1)"""),
md("##### Per class, at block 12"),
code("""rows = {}
for name, layers in active.items():
    used = get_usage(layers[:, 11], 0.5)
    num_classes = used.sum(dim=0)

    for c, label in enumerate(CLASSES):
        total = used[c].sum().item()
        distinct = (used[c] & (num_classes == 1)).sum().item()
        rows[(name, label)] = {
            "used": total,
            "distinct": distinct,
            "% distinct": 100 * distinct / max(total, 1),
            "% shared": 100 * (total - distinct) / max(total, 1),
        }

pd.DataFrame(rows).T.round(1)"""),
md("##### Which classes overlap most? (block 12, top 1024 of 4096)"),
code("""used = get_usage(active["top 1024 of 4096"][:, 11], 0.5).float()
overlap = used @ used.T
union = used.sum(dim=1, keepdim=True) + used.sum(dim=1) - overlap

# Jaccard overlap of the used latents
pd.DataFrame((overlap / union).numpy(), index=CLASSES, columns=CLASSES).round(2)"""),
]
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "second_analysis/10_shared_latents.ipynb")
