"""How many modes do most images share? Per checkpoint: modes kept on many val images, their eigenvalues, and their share of latent energy at each end."""
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
    lam = torch.cat([kae.real_eigenvalue.to(torch.cfloat), torch.complex(kae.block_real, kae.block_imag)])
    usage = (z[:, 0] != 0).double().mean(0)
    e6 = z[:, 0].abs().pow(2).sum(0); e6 = e6 / e6.sum()
    e12 = z[:, 1].abs().pow(2).sum(0); e12 = e12 / e12.sum()
    print(f"\n== {name}: 129 modes kept per image (TopK 256 = 1 real + 128 blocks); {int((usage > 0).sum())} of 1025 modes ever used; median usage {usage[usage > 0].median():.3%}")
    for t in [0.5, 0.8, 0.95]:
        m = usage > t
        print(f"   kept on > {t:.0%} of images: {int(m.sum())} modes, holding {e6[m].sum():.1%} of block-6 and {e12[m].sum():.1%} of block-12 latent energy")
    top = usage.topk(8).indices
    print(pd.DataFrame({"mode": top.tolist(), "kept on": usage[top].numpy(), "|λ|": lam[top].abs().detach().numpy(), "angle°": np.degrees(np.angle(lam[top].detach().numpy())), "block-6 energy": e6[top].numpy(), "block-12 energy": e12[top].numpy()}).round(3).to_string(index=False))
    print(f"   |λ| of all used modes: median {lam[usage > 0].abs().median():.2f}, max {lam[usage > 0].abs().max():.2f}")
