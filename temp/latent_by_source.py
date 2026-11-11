"""Latent prediction error and latent size, split by image source (ImageNet val, ImageNet train, CC3M)."""
import os, sys, torch
from pathlib import Path
from dotenv import load_dotenv
from safetensors.torch import load_file
sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
from train import JordanKoopmanAutoencoder, encode_pairs, compute_fvu
load_dotenv("/home/nsa325/work/koopmann/.env")
torch.set_grad_enabled(False); torch.set_num_threads(16)
name, ckpt = sys.argv[1], Path(sys.argv[2])
stats = torch.load(ckpt / f"{name}_stats.pt")
kae = JordanKoopmanAutoencoder(256, shared_modes=stats["shared_modes"], in_features=2304, latent_features=2049,
                               hidden_config=stats["hidden_config"], bias=True, batchnorm=False, nonlinearity="leaky_relu")
kae.load_state_dict(load_file(ckpt / f"{name}.safetensors")); kae.eval()
root = Path(os.getenv("DATASETS_CACHE"))
def load(path, n):
    s = torch.load(path, mmap=True)["states"]
    idx = torch.linspace(0, len(s) - 1, n).long()
    return ((s[idx].float() - stats["mean"]) / stats["std"])
sources = {"imagenet val": load(root / "imagenet/clip_cls_patch_std_ViT-B-32_val_blocks_5_11.pt", 20000),
           "imagenet train": load(root / "imagenet/clip_cls_patch_std_ViT-B-32_train_3_views_blocks_5_11.pt", 20000),
           "cc3m": load(root / "cc3m/clip_cls_patch_std_ViT-B-32_train_blocks_5_11.pt", 20000)}
lam = torch.complex(kae.block_real, kae.block_imag)
out = {}
for k, x in sources.items():
    phi = torch.cat([encode_pairs(kae, b) for b in x.split(4000)])
    pred = kae.koopman_forward(phi[:, 0], 1)
    z12 = torch.view_as_complex(phi[:, 1, 1:].unflatten(-1, (-1, 2)).contiguous())
    out[k] = (phi, pred, z12)
    print(f"{k:15s} latent FVU {compute_fvu(pred[:, None], phi[:, 1:]).item():.4f} | mean sq error {(pred - phi[:, 1]).pow(2).sum(-1).mean():.2f} "
          f"| latent energy: block 6 {phi[:, 0].pow(2).sum(-1).mean():.2f}, block 12 {phi[:, 1].pow(2).sum(-1).mean():.2f} "
          f"| state norm 6/12 {x[:, 0].norm(dim=-1).mean():.1f}/{x[:, 1].norm(dim=-1).mean():.1f}")
# mixed batches like training (ImageNet 3 views ~3.84M + CC3M ~2.9M)
phi = torch.cat([out["imagenet train"][0][:11000], out["cc3m"][0][:9000]]); pred = torch.cat([out["imagenet train"][1][:11000], out["cc3m"][1][:9000]])
print(f"mixed 55/45       latent FVU {compute_fvu(pred[:, None], phi[:, 1:]).item():.4f}")
e_in, e_cc = out["imagenet val"][2].abs().pow(2).mean(0), out["cc3m"][2].abs().pow(2).mean(0)
top = e_cc.argsort(descending=True)[:8]
print("top CC3M block-12 modes: mode, energy per image cc3m / imagenet val, on-rate cc3m / imagenet val, |lambda|")
for m in top.tolist():
    print(f"  {m + 1:5d}  {e_cc[m]:9.2f} / {e_in[m]:9.2f}   {(out['cc3m'][2][:, m] != 0).float().mean():.3f} / {(out['imagenet val'][2][:, m] != 0).float().mean():.3f}   {lam[m].abs():.3f}")
