"""
For every mode: does its block-6 phase or its block-6 size carry the information?
On the validation images that keep a mode we (a) give it the mode's average phase, keeping each image's size,
(b) give it the mode's average size, keeping each image's phase, (c) remove it, then push through K and the decoder and score the probe.
"""
import math, os, sys
from pathlib import Path
import pandas as pd
import torch
import inversion
from inversion import train, to_latent

torch.set_num_threads(16)
torch.set_grad_enabled(False)
SAMPLE = 1500
kae, stats = inversion.load_kae()
head = train.DinoHead(inversion.ENCODER, stats["mean"], stats["std"], torch.device("cpu"), zero_shot=True)
root = Path(os.getenv("DATASETS_CACHE"))
val_y = torch.load(root / "imagenet" / f"clip_cls_patch_std_{inversion.ENCODER}_val_blocks_5_11.pt")["labels"]
cache = torch.load("dino_explore.pt", weights_only=False)
z6, lam = cache["z"][:, 0], cache["eigenvalues"]
kept = z6 != 0
usage = kept.double().mean(dim=0)


def score(z_start, labels):
    """Log probabilities of the probe on the KAE's prediction from these block-6 modes."""
    return head.get_log_probs(kae.decode(to_latent(lam * z_start)))


rows = []
picker = torch.Generator().manual_seed(0)
modes = [m for m in range(1, 1025) if kept[:, m].sum() >= 250]
for count, mode in enumerate(modes):
    on = kept[:, mode].nonzero()[:, 0]
    values = z6[on, mode]
    mean_phase = torch.angle(torch.exp(1j * torch.angle(values)).mean())
    together = torch.exp(1j * torch.angle(values)).mean().abs().item()
    mean_size = values.abs().mean()
    chosen = on[torch.randperm(len(on), generator=picker)[:SAMPLE]]
    start, labels = z6[chosen].clone(), val_y[chosen]
    edits = {
        "base": start[:, mode],
        "mean phase": start[:, mode].abs() * torch.exp(1j * mean_phase),
        "mean size": mean_size * start[:, mode] / start[:, mode].abs(),
        "removed": torch.zeros_like(start[:, mode]),
    }
    row = {
        "mode": mode,
        "usage": usage[mode].item(),
        "|λ|": lam[mode].abs().item(),
        "angle°": math.degrees(torch.angle(lam[mode]).item()),
        "phase spread°": math.degrees(math.sqrt(max(-2 * math.log(together), 0))),
        "size spread": (values.abs().std() / mean_size).item(),
        "images": len(chosen),
    }
    base = None
    for name, value in edits.items():
        edited = start.clone()
        edited[:, mode] = value
        log_probs = score(edited, labels)
        answers = log_probs.argmax(dim=1)
        if base is None:
            base = (answers, log_probs)
        row[f"accuracy, {name}"] = (answers == labels).double().mean().item()
        row[f"changed, {name}"] = (answers != base[0]).double().mean().item()
        # How far the whole answer moves: KL from the unedited answer to the edited one
        row[f"kl, {name}"] = (base[1].exp() * (base[1] - log_probs)).sum(dim=1).mean().item()
    rows.append(row)
    if count % 40 == 0:
        print(count, len(modes), row, flush=True)

table = pd.DataFrame(rows)
table.to_csv("phase_size_all.csv", index=False)
