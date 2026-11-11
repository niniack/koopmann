"""
Does the eigenvalue angle say which images the KAE gets wrong? (DINOv3 run im_dino_w21_wd, ImageNet validation)

K multiplies a mode's block-6 value z by λ. The prediction λz splits into a kept part Re(λ)·z (block 6, rescaled)
and a built part i·Im(λ)·z (what blocks 7-12 add that block 6 did not hold in this mode).
An image's "built share" is the share of its predicted latent energy that is built: Σ (Im λ)²|z|² / Σ |λ|²|z|².
It needs no label and no block 12, so it can serve as a second "am I sure" signal for the cascade
(answer with the KAE when sure, run blocks 7-12 when not), next to the KAE's top probability.
"""

import os
from pathlib import Path

import torch

import inversion
from inversion import to_latent, train

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
kae, stats = inversion.load_kae()
head = train.DinoHead(inversion.ENCODER, stats["mean"], stats["std"], torch.device("cpu"), zero_shot=True)
root = Path(os.getenv("DATASETS_CACHE"))
data = torch.load(root / "imagenet" / f"clip_cls_patch_std_{inversion.ENCODER}_val_blocks_5_11.pt")
labels = data["labels"]
states = (data["states"].float() - stats["mean"]) / stats["std"]
cache = torch.load("dino_explore.pt", weights_only=False)
z6, lam = cache["z"][:, 0], cache["eigenvalues"]

with torch.no_grad():
    kae_log_probs, dino_log_probs = [], []
    for first in range(0, len(states), 5000):
        rows = slice(first, first + 5000)
        kae_log_probs.append(head.get_log_probs(kae.decode(to_latent(lam * z6[rows]))))
        dino_log_probs.append(head.get_log_probs(states[rows, 1]))
    kae_log_probs, dino_log_probs = torch.cat(kae_log_probs), torch.cat(dino_log_probs)
kae_right = kae_log_probs.argmax(dim=1) == labels
dino_right = dino_log_probs.argmax(dim=1) == labels
agrees = kae_log_probs.argmax(dim=1) == dino_log_probs.argmax(dim=1)
print(f"KAE {kae_right.float().mean():.4f} | DINOv3 {dino_right.float().mean():.4f} | same answer {agrees.float().mean():.4f}")

kept = z6 != 0
always = kept.float().mean(dim=0) > 0.99
energy = (lam.abs() * z6.abs()).pow(2)  # predicted block-12 energy of every mode. shape: [images, modes]
built = (lam.imag * z6.abs()).pow(2)
feature_energy, feature_built = energy[:, ~always], built[:, ~always]
angle = torch.angle(lam).abs()
top = kae_log_probs.exp().topk(2).values
# Each signal is larger when the KAE should be trusted more
signals = {
    "top probability": top[:, 0],
    "minus entropy": (kae_log_probs.exp() * kae_log_probs).sum(dim=1),
    "minus built share, all modes": -built.sum(dim=1) / energy.sum(dim=1),
    "minus built share, feature modes": -feature_built.sum(dim=1) / feature_energy.sum(dim=1),
    "minus mean |angle|, feature modes": -(feature_energy * angle[~always]).sum(dim=1) / feature_energy.sum(dim=1),
    "mean log|λ|, feature modes": (feature_energy * lam.abs().log()[~always]).sum(dim=1) / feature_energy.sum(dim=1),
    "share of energy in feature modes": feature_energy.sum(dim=1) / energy.sum(dim=1),
    "minus size of the block-6 latent": -z6.abs().pow(2).sum(dim=1).sqrt(),
}


def get_auc(signal, positive):
    """How well a signal ranks `positive` images above the rest (0.5 = chance)."""

    rank = signal.argsort().argsort().float()
    return ((rank[positive].mean() - (positive.sum() - 1) / 2) / (~positive).sum()).item()


def get_curve(signal, rows, shares):
    order = signal[rows].argsort()  # least sure first
    out = []
    for share in shares:
        passed = torch.zeros(len(order), dtype=torch.bool)
        passed[order[: int(share * len(order))]] = True
        out.append(torch.where(passed, dino_right[rows], kae_right[rows]).float().mean().item())
    return out


print("\nhow well each signal separates images (0.5 = chance): KAE right from wrong | KAE agreeing with DINOv3 from not | DINOv3 right from wrong")
for name, signal in signals.items():
    print(f"  {name:36s} {get_auc(signal, kae_right):.3f} | {get_auc(signal, agrees):.3f} | {get_auc(signal, dino_right):.3f}")

share = -signals["minus built share, feature modes"]
print(f"\nbuilt share of the feature modes at 10/50/90%: {[round(v, 3) for v in share.quantile(torch.tensor([0.1, 0.5, 0.9])).tolist()]}")
order = share.argsort()
for part, label in enumerate(["lowest fifth", "second", "third", "fourth", "highest fifth"]):
    chunk = order[part * len(order) // 5 : (part + 1) * len(order) // 5]
    print(f"  {label:14s} built share {share[chunk].mean():.3f}: KAE right {kae_right[chunk].float().mean():.3f}, agrees with DINOv3 {agrees[chunk].float().mean():.3f}, DINOv3 right {dino_right[chunk].float().mean():.3f}")

# Does it add to the top probability? A gate fitted on half of the images (logistic regression), scored on the other half.
half = torch.arange(len(labels)) % 2 == 0
SHARES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def fit_gate(names):
    x = torch.stack([signals[name] for name in names], dim=1).double()
    x = (x - x[half].mean(dim=0)) / x[half].std(dim=0)
    weight = torch.zeros(len(names), dtype=torch.double, requires_grad=True)
    bias = torch.zeros(1, dtype=torch.double, requires_grad=True)
    optimizer = torch.optim.LBFGS([weight, bias], max_iter=200)
    target = kae_right[half].double()

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(x[half] @ weight + bias, target)
        loss.backward()
        return loss

    optimizer.step(closure)
    return (x @ weight + bias).detach().float(), weight.detach().tolist()


print(f"\ncascade accuracy on the held-out half, by share of images passed on to blocks 7-12: {SHARES}")
gates = {
    "top probability": ["top probability"],
    "top probability + entropy": ["top probability", "minus entropy"],
    "top probability + entropy + built share": ["top probability", "minus entropy", "minus built share, feature modes"],
    "top probability + entropy + all spectral signals": list(signals),
    "spectral signals only (no head needed)": [name for name in signals if name not in ("top probability", "minus entropy")],
}
for label, names in gates.items():
    score, weights = fit_gate(names)
    print(f"  {label:50s} {[round(v, 4) for v in get_curve(score, ~half, SHARES)]} | separates right from wrong: {get_auc(score[~half], kae_right[~half]):.3f}")
print("done")
