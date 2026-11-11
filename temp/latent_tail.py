"""How unevenly latent energy is spread across images: a few images with huge latents dominate a pooled FVU."""
import os, sys, torch
from pathlib import Path
from dotenv import load_dotenv
from safetensors.torch import load_file
sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
from train import JordanKoopmanAutoencoder, encode_pairs, compute_fvu
load_dotenv("/home/nsa325/work/koopmann/.env")
torch.set_grad_enabled(False); torch.set_num_threads(16)
root = Path(os.getenv("DATASETS_CACHE"))
raw = torch.load(root / "imagenet/clip_cls_patch_std_ViT-B-32_val_blocks_5_11.pt", mmap=True)["states"][::2].float()
for name, ckpt in [("im_std_shared", "ckpt"), ("im_shared_noiso", "ckpt"), ("im_shared_noiso_2x4k", "ckpt"), ("im_cc3m_2x4k_e20", "ckpt_live")]:
    ckpt = Path(sys.argv[1]) / ckpt
    stats = torch.load(ckpt / f"{name}_stats.pt")
    kae = JordanKoopmanAutoencoder(256, shared_modes=stats["shared_modes"], in_features=2304, latent_features=2049,
                                   hidden_config=stats.get("hidden_config", [2048]), bias=True, batchnorm=False, nonlinearity="leaky_relu")
    kae.load_state_dict(load_file(ckpt / f"{name}.safetensors")); kae.eval()
    x = (raw - stats["mean"]) / stats["std"]
    phi = torch.cat([encode_pairs(kae, b) for b in x.split(5000)])
    pred = kae.koopman_forward(phi[:, 0], 1)
    e = phi[:, 1].pow(2).sum(-1)
    err = (pred - phi[:, 1]).pow(2).sum(-1)
    q = e.quantile(torch.tensor([0.5, 0.9, 0.99]))
    top = e.argsort(descending=True)[: len(e) // 100]
    per_image = (err / e).median()
    print(f"{name:22s} block-12 latent energy per image: median {q[0]:8.1f}, 90% {q[1]:8.1f}, 99% {q[2]:9.1f}, max/median {e.max() / q[0]:7.1f} | "
          f"top 1% of images hold {e[top].sum() / e.sum():.2f} of energy | pooled latent FVU {compute_fvu(pred[:, None], phi[:, 1:]).item():.4f}, median per-image relative error {per_image:.4f}")
