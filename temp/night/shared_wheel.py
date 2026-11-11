"""Gallery of a growth mode by phase: val images split into sectors by the angle of their block-6 coefficient, the largest ones in each sector. Plus a gallery of the small modes' gain extremes."""
import os, sys, torch, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import open_clip
from pathlib import Path
from collections import Counter
from datasets import load_dataset
from dotenv import load_dotenv
from safetensors.torch import load_file
sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
from train import JordanKoopmanAutoencoder, encode_pairs
load_dotenv("/home/nsa325/work/koopmann/.env")
torch.set_grad_enabled(False); torch.set_num_threads(8)
ck, name, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
val = torch.load(Path(os.getenv("DATASETS_CACHE")) / "imagenet/clip_cls_patch_std_ViT-B-32_val_blocks_5_11.pt")
y = val["labels"]
images = load_dataset("ILSVRC/imagenet-1k", split="validation", cache_dir=os.getenv("DATASETS_CACHE"))
names = open_clip.IMAGENET_CLASSNAMES
stats = torch.load(ck / f"{name}_stats.pt"); mean, std = stats["mean"].expand(2, -1), stats["std"].expand(2, -1)
kae = JordanKoopmanAutoencoder(256, shared_modes=stats["shared_modes"], in_features=2304, latent_features=2049, hidden_config=stats["hidden_config"], bias=True, batchnorm=False, nonlinearity="leaky_relu")
kae.load_state_dict(load_file(ck / f"{name}.safetensors")); kae.eval()
x = (val["states"].float() - mean) / std
phi = torch.cat([encode_pairs(kae, b) for b in x.split(5000)])
z = torch.cat([phi[..., :1].to(torch.cfloat), torch.view_as_complex(phi[..., 1:].unflatten(-1, (-1, 2)).contiguous())], dim=-1)
SECTORS, PER = 8, 7
def thumb(ax, i):
    im = images[int(i)]["image"].convert("RGB"); im.thumbnail((160, 160))
    ax.imshow(np.asarray(im)); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(names[y[i]][:18], fontsize=7)
for mode in [int(m) for m in sys.argv[4:]]:
    c = z[:, 0, mode]
    on = (c != 0).nonzero()[:, 0]
    v = c[on]
    # Main axis of the coefficients in the mode's plane, then each image's signed position along it
    theta = torch.angle((v ** 2).sum()) / 2
    pos = (v * torch.exp(-1j * theta)).real
    along = pos.pow(2).sum() / v.abs().pow(2).sum()
    hist = np.histogram(np.degrees(torch.angle(v).numpy()), bins=12, range=(-180, 180))[0]
    print(f"\nmode {mode}: axis at {np.degrees(theta.item()):+.0f}°, {along:.1%} of its energy lies along that axis; {(pos > 0).double().mean():.1%} of images on the + side")
    print(f"   phase histogram (30° bins from -180°): {hist.tolist()}")
    order = pos.argsort()
    rows = [("most negative", order[:PER]), ("middle", order[len(order) // 2 - PER // 2:][:PER]), ("most positive", order.flip(0)[:PER])]
    for label, sel in [("negative end", order[:200]), ("positive end", order.flip(0)[:200])]:
        common = Counter(y[on[sel]].tolist()).most_common(6)
        print(f"   {label}: " + ", ".join(f"{names[k]} ({n})" for k, n in common))
    fig, axes = plt.subplots(len(rows), PER, figsize=(PER * 1.7, len(rows) * 1.9))
    for r, (label, sel) in enumerate(rows):
        for j, ax in enumerate(axes[r]):
            thumb(ax, on[sel[j]])
        axes[r, 0].set_ylabel(label, fontsize=8)
    fig.suptitle(f"{name}: growth mode {mode} (|λ| {abs(complex(kae.block_real[mode - 1].item(), kae.block_imag[mode - 1].item())):.1f}), images along its main axis", fontsize=10)
    fig.tight_layout(); fig.savefig(out / f"axis_{name}_{mode}.jpg", dpi=80); plt.close(fig)
