"""
First look at im_dino_w21_wd (DINOv3 ViT-B/16, blocks 7-12) on the CPU: the spectrum, whether each mode behaves as its eigenvalue says,
and the snow test. Saves what it computes to dino_explore.pt for follow-ups.
Run with HF_HUB_OFFLINE=1 PYTHONPATH=<scratchpad>/tf_target.
"""
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from PIL import Image
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score
from torchvision import transforms

sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
import train  # noqa: E402

load_dotenv("/home/nsa325/work/koopmann/.env")
torch.set_grad_enabled(False)
torch.manual_seed(0)
torch.set_num_threads(32)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)

SP = Path(__file__).parent
NAME, ENCODER = "im_dino_w21_wd", "dinov3-vitb16"
EXPECTED = {"accuracy": 0.61778, "agreement": 0.66488}
FOLDERS = {"armytank": 847, "carwheel": 479, "firetruck": 555, "motorscooter": 670, "racingcar": 751, "schoolbus": 779, "trafficlight": 920}
cpu = torch.device("cpu")
root, ckpt = Path(os.getenv("DATASETS_CACHE")), Path(os.getenv("WEIGHTS_CACHE")) / "tokae"

stats = torch.load(ckpt / f"{NAME}_stats.pt")
mean, std = stats["mean"], stats["std"]
val = torch.load(root / "imagenet" / f"clip_cls_patch_std_{ENCODER}_val_blocks_5_11.pt")
val_x, val_y = (val["states"].float() - mean) / std, val["labels"]

kae = train.JordanKoopmanAutoencoder(
    256, shared_modes=True, in_features=2304, latent_features=2049, hidden_config=stats["hidden_config"], bias=True, batchnorm=False, nonlinearity="leaky_relu"
)
kae.load_state_dict(load_file(ckpt / f"{NAME}.safetensors"))
kae.eval()
head = train.DinoHead(ENCODER, mean, std, cpu, zero_shot=True)
eigenvalues = torch.cat([kae.real_eigenvalue.to(torch.cfloat), torch.complex(kae.block_real, kae.block_imag)])


def to_complex(phi):
    """Each mode as one complex number (mode 0 is real)."""

    return torch.cat([phi[..., :1].to(torch.cfloat), torch.view_as_complex(phi[..., 1:].unflatten(-1, (-1, 2)).contiguous())], dim=-1)


def to_latent(z):
    return torch.cat([z[..., :1].real, torch.view_as_real(z[..., 1:]).flatten(-2)], dim=-1)


def decode(z):
    return torch.cat([kae.decode(to_latent(batch)) for batch in z.split(5_000)])


# shape: [images, 2, modes], complex
z = to_complex(torch.cat([train.encode_pairs(kae, batch) for batch in val_x.split(2_500)]))
z6, z12 = z[:, 0], z[:, 1]
check = to_complex(kae.koopman_forward(to_latent(z6[:100]), 1))
assert torch.allclose(check, eigenvalues * z6[:100], atol=1e-4), "a mode's step is not its eigenvalue times its block-6 value"

prediction = decode(eigenvalues * z6)
answers, true_answers = head.classify(prediction), head.classify(val_x[:, 1])
scores = {"accuracy": (answers == val_y).double().mean().item(), "agreement": (answers == true_answers).double().mean().item()}
print(f"KAE: {scores}; real DINOv3 + probe {(true_answers == val_y).double().mean():.4f}")
for key, value in scores.items():
    assert abs(value - EXPECTED[key]) < 0.002, f"{key}: {value:.4f} here, {EXPECTED[key]} on wandb"

############# SPECTRUM #############

kept = z6 != 0
usage = kept.double().mean(dim=0)
size, angle = eigenvalues.abs(), torch.rad2deg(torch.angle(eigenvalues))
share6 = z6.abs().pow(2).sum(dim=0) / z6.abs().pow(2).sum()
share12 = z12.abs().pow(2).sum(dim=0) / z12.abs().pow(2).sum()

print(f"\n== Spectrum: {int((usage > 0).sum())} of {len(usage)} modes are ever kept; {int((usage >= 0.99).sum())} on at least 99% of images; {int((usage >= 0.5).sum())} on at least half")
used = usage > 0
print(f"Never-kept modes: |λ| median {size[~used].median():.3f}, |angle| median {angle[~used].abs().median():.2f}°")
print("Used modes, |λ| quantiles (5, 25, 50, 75, 95%):", [round(q, 3) for q in size[used].quantile(torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])).tolist()])
print("Used modes, |angle| quantiles (5, 25, 50, 75, 95%):", [round(q, 1) for q in angle[used].abs().quantile(torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])).tolist()])
for name, weight in [("usage", usage), ("block-6 latent energy", share6), ("block-12 latent energy", share12)]:
    w = (weight / weight.sum()).float()
    bins = [(0, 5), (5, 15), (15, 45), (45, 90), (90, 181)]
    print(f"Share of {name} by |angle|:", {f"{lo}-{hi}°": round(w[(angle.abs() >= lo) & (angle.abs() < hi)].sum().item(), 3) for lo, hi in bins})
    bins = [(0, 0.3), (0.3, 0.7), (0.7, 1.1), (1.1, 2), (2, 100)]
    print(f"Share of {name} by |λ|:", {f"{lo}-{hi}": round(w[(size >= lo) & (size < hi)].sum().item(), 3) for lo, hi in bins})

rows = []
for mode in used.nonzero()[:, 0].tolist():
    on = kept[:, mode]
    a, b = z6[on, mode], z12[on, mode]
    rows.append(
        {
            "mode": mode,
            "|λ|": size[mode].item(),
            "angle": angle[mode].item(),
            "kept": usage[mode].item(),
            "share 6": share6[mode].item(),
            "share 12": share12[mode].item(),
            "gain": (b.abs() / a.abs()).median().item(),
            # The turn the data makes, pooled over images: the angle of sum(z12 * conj(z6))
            "turn": math.degrees(torch.angle((b * a.conj()).sum()).item()),
            # How aligned the per-image turns are: 1 means every image turns by the same angle
            "turn agreement": ((b * a.conj()).sum().abs() / (b.abs() * a.abs()).sum()).item(),
            "R²": 1 - ((b - eigenvalues[mode] * a).abs().pow(2).sum() / (b - b.mean()).abs().pow(2).sum().clamp_min(1e-12)).item(),
        }
    )
modes = pd.DataFrame(rows).set_index("mode")
print("\nTop 25 modes by block-12 latent energy")
print(modes.sort_values("share 12", ascending=False).head(25).round(3).to_string())
print("\nTop 25 modes by usage")
print(modes.sort_values("kept", ascending=False).head(25).round(3).to_string())
common = modes[modes["kept"] >= 0.005]
print(f"\n{len(common)} modes kept on at least 0.5% of images: R² median {common['R²'].median():.3f}, turn agreement median {common['turn agreement'].median():.3f}")
print("gain vs |λ|: correlation", round(np.corrcoef(common["gain"], common["|λ|"])[0, 1], 3), "| turn vs angle: median absolute gap", round((common["turn"] - common["angle"]).abs().median(), 2), "°")
print(common.groupby(pd.cut(common["angle"].abs(), [0, 5, 15, 45, 90, 180], include_lowest=True), observed=True)[["kept", "share 6", "share 12", "|λ|", "gain", "R²", "turn agreement"]].agg(["count", "median"]).round(3).to_string())

############# SNOW #############

from transformers import AutoModel  # noqa: E402

dino = AutoModel.from_pretrained(train.DINO_MODELS[ENCODER], cache_dir=f"{os.getenv('HF_HOME')}/hub").eval()
first_patch = 1 + dino.config.num_register_tokens
preprocess = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        train.to_rgb,
        transforms.ToTensor(),
        transforms.Normalize(*train.IMAGENET_MEAN_STD),
    ]
)


def get_states(pixels):
    """Unstandardized states after blocks 6 and 12, built the way the training cache was."""

    states = []

    def keep(_, __, output):
        output = output[0] if isinstance(output, tuple) else output
        patches = output[:, first_patch:]
        states.append(torch.cat([output[:, 0], patches.mean(dim=1), patches.std(dim=1)], dim=-1))

    hooks = [dino.model.layer[block].register_forward_hook(keep) for block in stats["blocks"]]
    for batch in pixels.split(32):
        dino(pixel_values=batch)
    for hook in hooks:
        hook.remove()
    # NOTE: The hooks fire block by block within a batch, so states alternate block 6, block 12 per batch
    return torch.cat([torch.stack(states[i : i + 2], dim=1) for i in range(0, len(states), 2)])


images = load_dataset("ILSVRC/imagenet-1k", split="validation", cache_dir=os.getenv("DATASETS_CACHE"))
fresh = get_states(torch.stack([preprocess(images[i]["image"]) for i in range(5)]))
cached = val["states"][:5].float()
gap = ((fresh - cached).norm(dim=-1) / cached.norm(dim=-1)).max().item()
print(f"\n== Snow. Largest relative gap between fresh and cached states on 5 val images: {gap:.2e}")
assert gap < 1e-2, "snowy states would not be comparable to the cached ones"

paths = sorted((root / "snowy_imagenet").glob("*/*.jpg"))
snowy_raw = get_states(torch.stack([preprocess(Image.open(path)) for path in paths]))
snowy_x, snowy_y = (snowy_raw - mean) / std, torch.tensor([FOLDERS[path.parent.name] for path in paths])
normal = torch.isin(val_y, torch.tensor(list(FOLDERS.values())))
x, y = torch.cat([val_x[normal], snowy_x]), torch.cat([val_y[normal], snowy_y])
is_snowy = torch.cat([torch.zeros(int(normal.sum()), dtype=torch.bool), torch.ones(len(snowy_x), dtype=torch.bool)])

zs = to_complex(train.encode_pairs(kae, x))
base = decode(eigenvalues * zs[:, 0])
kae_right, true_right = head.classify(base) == y, head.classify(x[:, 1]) == y
print(f"{int(is_snowy.sum())} snowy and {int((~is_snowy).sum())} normal images of the same 7 classes")
print(f"Real DINOv3 + probe: snowy {true_right[is_snowy].double().mean():.3f}, normal {true_right[~is_snowy].double().mean():.3f}")
print(f"KAE:                 snowy {kae_right[is_snowy].double().mean():.3f}, normal {kae_right[~is_snowy].double().mean():.3f}")
by_class = pd.DataFrame(
    {
        name: {
            "snowy images": int((is_snowy & (y == c)).sum()),
            "real, snowy": true_right[is_snowy & (y == c)].double().mean().item(),
            "real, normal": true_right[~is_snowy & (y == c)].double().mean().item(),
            "KAE, snowy": kae_right[is_snowy & (y == c)].double().mean().item(),
            "KAE, normal": kae_right[~is_snowy & (y == c)].double().mean().item(),
        }
        for name, c in FOLDERS.items()
    }
).T
print(by_class.round(3).to_string())

activity = zs[:, 0].abs()
aucs = torch.tensor([roc_auc_score(is_snowy.numpy(), activity[:, m].numpy()) if activity[:, m].any() else 0.5 for m in range(activity.shape[1])])
snow = pd.DataFrame(
    {
        "AUC": aucs.numpy(),
        "on for snowy": (activity[is_snowy] > 0).double().mean(dim=0).numpy(),
        "on for normal": (activity[~is_snowy] > 0).double().mean(dim=0).numpy(),
        "on for val": usage.numpy(),
        "|λ|": size.numpy(),
        "angle": angle.numpy(),
    }
)
print("\nModes whose block-6 size best separates snowy from normal images")
print(snow.sort_values("AUC", ascending=False).head(12).round(3).to_string())
fold = torch.arange(len(y)) % 2
for half in [0, 1]:
    pick, test = fold == half, fold != half
    half_aucs = torch.tensor([roc_auc_score(is_snowy[pick].numpy(), activity[pick, m].numpy()) if activity[pick, m].any() else 0.5 for m in range(activity.shape[1])])
    best = int(half_aucs.argmax())
    print(f"Half {half}: picks mode {best}, AUC on the other half {roc_auc_score(is_snowy[test].numpy(), activity[test, best].numpy()):.3f}")

EDITS = {
    "size x0": 0.0,
    "size x0.5": 0.5,
    "size x2": 2.0,
    "turn +45°": complex(math.cos(math.pi / 4), math.sin(math.pi / 4)),
    "turn -45°": complex(math.cos(math.pi / 4), -math.sin(math.pi / 4)),
    "turn +90°": 1j,
    "turn -90°": -1j,
    "turn 180°": -1.0,
}


def edit(z_start, mode, factor):
    """The prediction with one eigenvalue multiplied by a complex factor."""

    lam = eigenvalues.clone()
    lam[mode] = lam[mode] * factor
    return decode(lam * z_start)


for mode in snow.sort_values("AUC", ascending=False).index[:4].tolist():
    on = kept[:, mode]
    val_start, val_truth = z6[on], val_y[on]
    val_before = answers[on] == val_truth
    rows = {}
    for label, factor in EDITS.items():
        if mode == 0 and isinstance(factor, complex) and factor.imag != 0:
            continue
        right = head.classify(edit(zs[:, 0], mode, factor)) == y
        val_right = head.classify(edit(val_start, mode, factor)) == val_truth
        rows[label] = {
            "snowy acc": right[is_snowy].double().mean().item(),
            "normal acc": right[~is_snowy].double().mean().item(),
            "snowy fixed": int((~kae_right & right & is_snowy).sum()),
            "snowy broken": int((kae_right & ~right & is_snowy).sum()),
            "normal fixed": int((~kae_right & right & ~is_snowy).sum()),
            "normal broken": int((kae_right & ~right & ~is_snowy).sum()),
            "val fixed": int((~val_before & val_right).sum()),
            "val broken": int((val_before & ~val_right).sum()),
        }
    lam = eigenvalues[mode]
    print(f"\nMode {mode}: λ = {lam.real:.2f}{lam.imag:+.2f}i, |λ| {lam.abs():.2f}, angle {angle[mode]:+.0f}°, AUC {aucs[mode]:.3f}, kept on {int(on.sum())} val images")
    print(f"as trained: snowy acc {kae_right[is_snowy].double().mean():.3f}, normal acc {kae_right[~is_snowy].double().mean():.3f}")
    print(pd.DataFrame(rows).T.round(3).to_string())

torch.save(
    {"z": z, "eigenvalues": eigenvalues, "answers": answers, "true_answers": true_answers, "modes": modes, "snowy_raw": snowy_raw, "snowy_y": snowy_y, "aucs": aucs, "paths": [str(p) for p in paths]},
    SP / "dino_explore.pt",
)
print("\nsaved", SP / "dino_explore.pt")
