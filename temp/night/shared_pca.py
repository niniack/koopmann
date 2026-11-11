"""Are the always-on modes the main directions of variation? Compares with the top principal components, and checks whether their information is in size or in phase."""
import os, sys, torch, numpy as np
from pathlib import Path
from dotenv import load_dotenv
from safetensors.torch import load_file
sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
from train import JordanKoopmanAutoencoder, encode_pairs
load_dotenv("/home/nsa325/work/koopmann/.env")
torch.set_grad_enabled(False); torch.set_num_threads(8)
ck, name = Path(sys.argv[1]), sys.argv[2]
val = torch.load(Path(os.getenv("DATASETS_CACHE")) / "imagenet/clip_cls_patch_std_ViT-B-32_val_blocks_5_11.pt")
stats = torch.load(ck / f"{name}_stats.pt"); mean, std = stats["mean"].expand(2, -1), stats["std"].expand(2, -1)
kae = JordanKoopmanAutoencoder(256, shared_modes=stats["shared_modes"], in_features=2304, latent_features=2049, hidden_config=stats["hidden_config"], bias=True, batchnorm=False, nonlinearity="leaky_relu")
kae.load_state_dict(load_file(ck / f"{name}.safetensors")); kae.eval()
x = (val["states"].float() - mean) / std
phi = torch.cat([encode_pairs(kae, b) for b in x.split(5000)])
z = torch.cat([phi[..., :1].to(torch.cfloat), torch.view_as_complex(phi[..., 1:].unflatten(-1, (-1, 2)).contiguous())], dim=-1)
lam = torch.cat([kae.real_eigenvalue.to(torch.cfloat), torch.complex(kae.block_real, kae.block_imag)]).detach()
shared = (z[:, 0] != 0).double().mean(0) > 0.95
n_dims = int(shared[1:].sum()) * 2 + int(shared[0])
for block in []:
    c = x[:, block] - x[:, block].mean(0)
    s = torch.linalg.svdvals(c[::2])
    share = (s.pow(2).cumsum(0) / s.pow(2).sum())
    print(f"block {6 + 6 * block}: top {n_dims} principal components explain {share[n_dims - 1]:.1%} of variance (top 10: {share[9]:.1%}, top 100: {share[99]:.1%})")
# Size vs phase: for each always-on 2x2 mode, spread of size (relative) and of phase (circular), at block 6
idx = shared.nonzero()[:, 0]
idx = idx[idx > 0]
zz = z[:, 0, idx]
on = zz != 0
kept_share = on.double().mean(0)
size_spread, circ_std = [], []
for j in range(len(idx)):
    v = zz[on[:, j], j]
    size_spread.append((v.abs().std() / v.abs().mean()).item())
    R = (v / v.abs()).mean().abs().clamp(1e-6, 1 - 1e-7)  # mean resultant length: 1 = phase fixed, 0 = uniform
    circ_std.append(torch.sqrt(-2 * torch.log(R)).item())
size_spread, circ_std = torch.tensor(size_spread), torch.tensor(circ_std)
order = lam[idx].abs().argsort(descending=True)
print("\nalways-on modes at block 6, by |λ|: size spread (std / mean), phase spread (circular std, degrees)")
for i in order.tolist():
    print(f"   mode {int(idx[i]):4d} |λ| {lam[idx[i]].abs():.2f} rotation {np.degrees(np.angle(lam[idx[i]].item())):+6.1f}°, kept on {kept_share[i]:.1%}: size spread {size_spread[i]:.2f}, phase spread {np.degrees(circ_std[i].item()):5.1f}°")
