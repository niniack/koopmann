"""Are eigenvalues each mode's gain from block 6 to block 12 on held-out images? One summary per checkpoint."""
import os, sys, torch, numpy as np, pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from safetensors.torch import load_file
sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
from train import JordanKoopmanAutoencoder, encode_pairs
load_dotenv("/home/nsa325/work/koopmann/.env")
torch.set_grad_enabled(False); torch.set_num_threads(16)
name, ckpt = sys.argv[1], Path(sys.argv[2])
stats = torch.load(ckpt / f"{name}_stats.pt")
val = torch.load(Path(os.getenv("DATASETS_CACHE")) / "imagenet" / "clip_cls_patch_std_ViT-B-32_val_blocks_5_11.pt")
x = (val["states"] - stats["mean"]) / stats["std"]
kae = JordanKoopmanAutoencoder(256, shared_modes=stats.get("shared_modes", False), in_features=x.shape[-1], latent_features=2049,
                               hidden_config=stats.get("hidden_config", [2048]), bias=True, batchnorm=False, nonlinearity="leaky_relu")
kae.load_state_dict(load_file(ckpt / f"{name}.safetensors")); kae.eval()
phi = torch.cat([encode_pairs(kae, b) for b in x.split(5000)])
z = torch.view_as_complex(phi[..., 1:].unflatten(-1, (-1, 2)).contiguous())
z6, z12 = z[:, 0], z[:, 1]
lam = torch.complex(kae.block_real, kae.block_imag)
on6, on12 = z6 != 0, z12 != 0
energy12 = z12.abs().pow(2).sum(0)
err = (z12 - lam * z6).abs().pow(2).sum(0)
usage6 = on6.float().mean(0)
print(f"== {name} (standardization {stats.get('standardization', 'global')}, shared modes {stats.get('shared_modes', False)})")
print(f"real mode: lambda {kae.real_eigenvalue.item():.3f}, share of block-12 latent energy {(phi[:, 1, 0].pow(2).sum() / phi[:, 1].pow(2).sum()).item():.4f}")
order = energy12.argsort(descending=True)
share = energy12[order].cumsum(0) / energy12.sum()
print("modes holding 50/80/95% of block-12 latent energy:", [(share < q).sum().item() + 1 for q in [0.5, 0.8, 0.95]])
rows = []
for m in order[:8].tolist():
    o6, o12 = on6[:, m], on12[:, m]
    rows.append({"mode": m + 1, "on at 6": o6.float().mean().item(), "on at 12": o12.float().mean().item(),
                 "|lambda|": lam[m].abs().item(), "R2": (1 - err[m] / energy12[m]).item(), "energy share": (energy12[m] / energy12.sum()).item()})
print(pd.DataFrame(rows).round(3).to_string(index=False))
for lo, hi in [(0.5, 1.01), (0.1, 0.5), (0.01, 0.1), (0.0, 0.01)]:
    sel = (usage6 >= lo) & (usage6 < hi)
    print(f"on for {lo:.0%}-{hi:.0%} at block 6: {int(sel.sum()):4d} modes, block-12 energy {(energy12[sel].sum() / energy12.sum()):.3f}, "
          f"pooled R2 {(1 - err[sel].sum() / energy12[sel].sum()):.3f}, median |lambda| {lam[sel].abs().median():.3f}")
rows = []
for m in torch.nonzero(usage6 >= 0.01).flatten().tolist():
    o = on6[:, m]; a, b = z6[o, m], z12[o, m]
    fit = (b * a.conj()).sum() / (a.abs() ** 2).sum()
    rows.append({"usage": usage6[m].item(), "abs_lambda": lam[m].abs().item(), "abs_fit": fit.abs().item(),
                 "r2": (1 - (b - lam[m] * a).abs().pow(2).sum() / b.abs().pow(2).sum()).item(),
                 "still_on": ((o & on12[:, m]).sum() / o.sum()).item()})
r = pd.DataFrame(rows)
print(f"{len(r)} modes on for 1%+: |lambda| median {r.abs_lambda.median():.3f}, 10-90% {r.abs_lambda.quantile(.1):.3f}-{r.abs_lambda.quantile(.9):.3f}")
print(f"  fitted gain vs |lambda|: spearman {r[['abs_lambda', 'abs_fit']].corr('spearman').iloc[0, 1]:.3f}, median |fit|/|lambda| {(r.abs_fit / r.abs_lambda).median():.3f}")
print(f"  per-mode R2: median {r.r2.median():.3f}; still kept at 12 when kept at 6: median {r.still_on.median():.3f}")
print(f"rare modes (<0.1% of images): {int((usage6 < 0.001).sum())}")
