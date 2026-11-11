import os, sys, torch, numpy as np, pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from safetensors.torch import load_file
sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
from train import JordanKoopmanAutoencoder
load_dotenv("/home/nsa325/work/koopmann/.env")
torch.set_grad_enabled(False); torch.set_num_threads(16)
name, tokens, ckpt = sys.argv[1], sys.argv[2], Path(sys.argv[3])
stats = torch.load(ckpt / f"{name}_stats.pt")
val = torch.load(Path(os.getenv("DATASETS_CACHE")) / "imagenet" / f"clip_{tokens}_ViT-B-32_val_blocks_5_11.pt")
x = (val["states"] - stats["mean"]) / stats["std"]
kae = JordanKoopmanAutoencoder(256, in_features=x.shape[-1], latent_features=2049, hidden_config=[2048], bias=True, batchnorm=False, nonlinearity="leaky_relu")
kae.load_state_dict(load_file(ckpt / f"{name}.safetensors")); kae.eval()
phi6 = torch.cat([kae.encode(b) for b in x[:, 0].split(5000)])
phi12 = torch.cat([kae.encode(b) for b in x[:, 1].split(5000)])
pred = kae.koopman_forward(phi6, 1)
print("latent FVU (as in training logs):", ((pred - phi12).pow(2).sum() / (phi12 - phi12.mean(0)).pow(2).sum()).item())
z6 = torch.view_as_complex(phi6[:, 1:].unflatten(-1, (-1, 2)).contiguous()); z12 = torch.view_as_complex(phi12[:, 1:].unflatten(-1, (-1, 2)).contiguous())
lam = torch.complex(kae.block_real, kae.block_imag)
energy12 = z12.abs().pow(2).sum(0); energy6 = z6.abs().pow(2).sum(0)
print("real eigenvalue mode: lambda", kae.real_eigenvalue.item(), "share of block-12 energy", (phi12[:, 0].pow(2).sum() / phi12.pow(2).sum()).item())
order = energy12.argsort(descending=True)
share = (energy12[order].cumsum(0) / energy12.sum())
print("modes holding 50/80/95% of block-12 energy:", [(share < q).sum().item() + 1 for q in [0.5, 0.8, 0.95]])
err = (z12 - lam * z6).abs().pow(2).sum(0)
rows = []
for m in order[:15].tolist():
    o6, o12 = z6[:, m] != 0, z12[:, m] != 0
    rows.append({"mode": m + 1, "on at 6": o6.float().mean().item(), "on at 12": o12.float().mean().item(),
                 "on at both / on at 6": ((o6 & o12).sum() / o6.sum().clamp(min=1)).item(),
                 "|lambda|": lam[m].abs().item(), "R2": (1 - err[m] / energy12[m]).item(), "energy share": (energy12[m] / energy12.sum()).item()})
print(pd.DataFrame(rows).round(3).to_string(index=False))
usage6 = (z6 != 0).float().mean(0)
for lo, hi in [(0.5, 1.01), (0.1, 0.5), (0.01, 0.1), (0.0, 0.01)]:
    sel = (usage6 >= lo) & (usage6 < hi)
    print(f"modes on for {lo:.0%}-{hi:.0%} at block 6: {int(sel.sum())}, share of block-12 energy {(energy12[sel].sum() / energy12.sum()):.3f}, "
          f"R2 pooled {(1 - err[sel].sum() / energy12[sel].sum()):.3f}, median |lambda| {lam[sel].abs().median():.3f}")
