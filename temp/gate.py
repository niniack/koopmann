"""
A cascade: answer with the KAE (blocks 1-6 of DINOv3, then the KAE) when the KAE is sure, and run DINOv3's blocks 7-12 when it is not.
Scores how well a few "am I sure" signals of the KAE pick the images to pass on, on ImageNet's validation images.
"""

import os
from pathlib import Path

import torch

import inversion
from inversion import to_latent, train

torch.set_num_threads(8)
torch.set_grad_enabled(False)
kae, stats = inversion.load_kae()
head = train.DinoHead(inversion.ENCODER, stats["mean"], stats["std"], torch.device("cpu"), zero_shot=True)
root = Path(os.getenv("DATASETS_CACHE"))
data = torch.load(root / "imagenet" / f"clip_cls_patch_std_{inversion.ENCODER}_val_blocks_5_11.pt")
print({key: (tuple(value.shape), value.dtype) for key, value in data.items() if torch.is_tensor(value)})
labels = data["labels"]
states = (data["states"].float() - stats["mean"]) / stats["std"]
cache = torch.load("dino_explore.pt", weights_only=False)
z6, lam = cache["z"][:, 0], cache["eigenvalues"]

kae_log_probs, dino_log_probs, start_error = [], [], []
for first in range(0, len(states), 5000):
    rows = slice(first, first + 5000)
    kae_log_probs.append(head.get_log_probs(kae.decode(to_latent(lam * z6[rows]))))
    dino_log_probs.append(head.get_log_probs(states[rows, 1]))
    # How badly the KAE reconstructs the block-6 state it was given, as a share of that state's size
    recon = kae.decode(to_latent(z6[rows]))
    start_error.append((recon - states[rows, 0]).pow(2).sum(dim=1) / states[rows, 0].pow(2).sum(dim=1))
kae_log_probs, dino_log_probs, start_error = torch.cat(kae_log_probs), torch.cat(dino_log_probs), torch.cat(start_error)
kae_right = kae_log_probs.argmax(dim=1) == labels
dino_right = dino_log_probs.argmax(dim=1) == labels
print(f"KAE {kae_right.float().mean():.4f} | DINOv3 {dino_right.float().mean():.4f} | same answer {(kae_log_probs.argmax(dim=1) == dino_log_probs.argmax(dim=1)).float().mean():.4f}")

top = kae_log_probs.exp().topk(2).values
# Each signal is larger when the KAE should be trusted more
signals = {
    "top probability": top[:, 0],
    "margin between top two": top[:, 0] - top[:, 1],
    "minus entropy": (kae_log_probs.exp() * kae_log_probs).sum(dim=1),
    "minus block-6 reconstruction error": -start_error,
    "random": torch.rand(len(labels), generator=torch.Generator().manual_seed(0)),
    # NOTE: A ceiling that uses the labels: pass on exactly the images the KAE gets wrong
    "oracle": kae_right.float() + 1e-3 * torch.rand(len(labels), generator=torch.Generator().manual_seed(0)),
}
SHARES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
print("accuracy of the cascade by share of images passed on to blocks 7-12:", SHARES)
curves = {}
for name, signal in signals.items():
    order = signal.argsort()  # least sure first
    row = []
    for share in SHARES:
        passed = torch.zeros(len(labels), dtype=torch.bool)
        passed[order[: int(share * len(labels))]] = True
        row.append(torch.where(passed, dino_right, kae_right).float().mean().item())
    curves[name] = row
    # How well the signal separates the KAE's right answers from its wrong ones (0.5 = chance)
    rank = signal.argsort().argsort().float()
    auc = (rank[kae_right].mean() - (kae_right.sum() - 1) / 2) / (~kae_right).sum()
    print(f"  {name:38s} {[round(v, 4) for v in row]} | separates right from wrong: {auc:.3f}")
torch.save({"shares": SHARES, "curves": curves}, "gate.pt")
# The share to pass on for a wanted accuracy, with the best signal
best = signals["top probability"].argsort()
for wanted in [0.75, 0.80, 0.82, 0.83]:
    for share in torch.linspace(0, 1, 201):
        passed = torch.zeros(len(labels), dtype=torch.bool)
        passed[best[: int(share * len(labels))]] = True
        if torch.where(passed, dino_right, kae_right).float().mean() >= wanted:
            print(f"  {wanted:.2f} accuracy needs {share:.3f} of the images passed on (blocks 7-12 run on that share)")
            break
