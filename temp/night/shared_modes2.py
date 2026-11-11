"""The always-on modes (kept on > 95% of val images), sorted by block-12 energy, plus how energy splits between them and the rest of each image's modes."""
import os, sys, torch, numpy as np, pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from safetensors.torch import load_file
sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
from train import JordanKoopmanAutoencoder, encode_pairs
load_dotenv("/home/nsa325/work/koopmann/.env")
torch.set_grad_enabled(False); torch.set_num_threads(8)
ck = Path(sys.argv[1])
val = torch.load(Path(os.getenv("DATASETS_CACHE")) / "imagenet/clip_cls_patch_std_ViT-B-32_val_blocks_5_11.pt")
pd.set_option("display.width", 250)
for name in sys.argv[2:]:
    stats = torch.load(ck / f"{name}_stats.pt"); mean, std = stats["mean"].expand(2, -1), stats["std"].expand(2, -1)
    kae = JordanKoopmanAutoencoder(256, shared_modes=stats["shared_modes"], in_features=2304, latent_features=2049, hidden_config=stats["hidden_config"], bias=True, batchnorm=False, nonlinearity="leaky_relu")
    kae.load_state_dict(load_file(ck / f"{name}.safetensors")); kae.eval()
    x = (val["states"].float() - mean) / std
    phi = torch.cat([encode_pairs(kae, b) for b in x.split(5000)])
    z = torch.cat([phi[..., :1].to(torch.cfloat), torch.view_as_complex(phi[..., 1:].unflatten(-1, (-1, 2)).contiguous())], dim=-1)
    lam = torch.cat([kae.real_eigenvalue.to(torch.cfloat), torch.complex(kae.block_real, kae.block_imag)]).detach()
    usage = (z[:, 0] != 0).double().mean(0)
    shared = usage > 0.95
    e6 = z[:, 0].abs().pow(2); e12 = z[:, 1].abs().pow(2)
    # Per image: share of its own latent energy in always-on modes
    own6 = (e6[:, shared].sum(1) / e6.sum(1)).median(); own12 = (e12[:, shared].sum(1) / e12.sum(1)).median()
    # How much do the always-on modes vary across images? Spread of each one's coefficient relative to its mean
    spread = (z[:, 0, shared].abs().std(0) / z[:, 0, shared].abs().mean(0)).median()
    E12 = e12.sum(0) / e12.sum()
    E6 = e6.sum(0) / e6.sum()
    idx = shared.nonzero()[:, 0]
    idx = idx[E12[idx].argsort(descending=True)]
    print(f"\n== {name}: {len(idx)} always-on modes; per image they hold a median {own6:.1%} of block-6 and {own12:.1%} of block-12 latent energy; the median always-on mode's size varies by {spread:.0%} across images")
    print(pd.DataFrame({"mode": idx.tolist(), "|λ|": lam[idx].abs().numpy(), "angle°": np.degrees(np.angle(lam[idx].numpy())), "block-6 energy": E6[idx].numpy(), "block-12 energy": E12[idx].numpy()}).head(6).round(3).to_string(index=False))
    print(f"   image-specific modes: {int((usage > 0).sum() - shared.sum())} modes used, each image keeps about {int(129 - shared.sum())} of them")
