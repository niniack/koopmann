"""Do eigenvalues describe each mode's change from block 6 to block 12 on held-out images, or are they arbitrary?"""
import os, sys, torch, numpy as np, pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from safetensors.torch import load_file
sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
from train import JordanKoopmanAutoencoder
load_dotenv("/home/nsa325/work/koopmann/.env")
torch.set_grad_enabled(False); torch.set_num_threads(16)
name, tokens = sys.argv[1], sys.argv[2]
ckpt = Path(sys.argv[3])
stats = torch.load(ckpt / f"{name}_stats.pt")
val = torch.load(Path(os.getenv("DATASETS_CACHE")) / "imagenet" / f"clip_{tokens}_ViT-B-32_val_blocks_5_11.pt")
x = (val["states"] - stats["mean"]) / stats["std"]
kae = JordanKoopmanAutoencoder(256, in_features=x.shape[-1], latent_features=2049, hidden_config=[2048], bias=True, batchnorm=False, nonlinearity="leaky_relu")
kae.load_state_dict(load_file(ckpt / f"{name}.safetensors")); kae.eval()

def modes(states):
    phi = torch.cat([kae.encode(b) for b in states.split(5000)])
    return torch.view_as_complex(phi[:, 1:].unflatten(-1, (-1, 2)).contiguous())  # [images, modes], complex
z6, z12 = modes(x[:, 0]), modes(x[:, 1])
lam = torch.complex(kae.block_real, kae.block_imag)
on6, on12 = z6 != 0, z12 != 0

rows = []
for m in range(len(lam)):
    o = on6[:, m]
    if o.sum() < 50:
        rows.append({"usage": o.float().mean().item(), "abs_lambda": lam[m].abs().item()}); continue
    a, b = z6[o, m], z12[o, m]
    fit = (b * a.conj()).sum() / (a.abs() ** 2).sum()                     # best single gain on held-out images
    r2_lam = 1 - (b - lam[m] * a).abs().pow(2).sum() / b.abs().pow(2).sum()  # how much of the end value the eigenvalue explains
    both = o & on12[:, m]
    rows.append({
        "usage": o.float().mean().item(), "abs_lambda": lam[m].abs().item(), "abs_fit": fit.abs().item(),
        "angle_lambda": lam[m].angle().item(), "angle_fit": fit.angle().item(), "r2": r2_lam.item(),
        "still_on_at_12": (both.sum() / o.sum()).item(),
        "median_ratio_when_on_both": (z12[both, m].abs() / z6[both, m].abs()).median().item() if both.sum() > 10 else np.nan,
    })
r = pd.DataFrame(rows)
used = r[r.usage >= 0.01]
print(f"{name}: {len(used)} modes on for 1%+ of val images (of {len(r)})")
print(f"|lambda|: median {used.abs_lambda.median():.3f}, 10-90% {used.abs_lambda.quantile(.1):.3f}-{used.abs_lambda.quantile(.9):.3f}")
print(f"held-out fitted |gain| vs |lambda|: spearman {used[['abs_lambda','abs_fit']].corr('spearman').iloc[0,1]:.3f}, median |fit|/|lambda| {(used.abs_fit / used.abs_lambda).median():.3f}")
print(f"angle: corr {np.corrcoef(used.angle_lambda, used.angle_fit)[0,1]:.3f}")
print(f"per-mode R2 of end = lambda x start, on held-out images: median {used.r2.median():.3f}, 10-90% {used.r2.quantile(.1):.3f}-{used.r2.quantile(.9):.3f}")
print(f"mode still kept by TopK at block 12 when kept at block 6: median {used.still_on_at_12.median():.3f}")
print(f"|lambda| vs median ratio when on at both ends: spearman {used[['abs_lambda','median_ratio_when_on_both']].corr('spearman').iloc[0,1]:.3f}")
print(f"|lambda| vs still-on-at-12: spearman {used[['abs_lambda','still_on_at_12']].corr('spearman').iloc[0,1]:.3f}")
print(f"|lambda| vs usage (all modes): spearman {r[['abs_lambda','usage']].corr('spearman').iloc[0,1]:.3f}")
rare = r[r.usage < 0.001]
print(f"rarely used modes (<0.1% of images): {len(rare)}, their |lambda| median {rare.abs_lambda.median():.3f} (init was 1.0)")
r.to_pickle(Path(sys.argv[3]) / f"{name}_eigen.pkl")
