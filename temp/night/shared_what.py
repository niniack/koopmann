"""What do the always-on modes do? Tests whether they carry each block's average state, and what their per-image variation tracks."""
import os, sys, copy, torch, numpy as np, pandas as pd, open_clip
from pathlib import Path
from dotenv import load_dotenv
from safetensors.torch import load_file
sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
from train import JordanKoopmanAutoencoder, encode_pairs
load_dotenv("/home/nsa325/work/koopmann/.env")
torch.set_grad_enabled(False); torch.set_num_threads(8)
ck, name = Path(sys.argv[1]), sys.argv[2]
root = Path(os.getenv("DATASETS_CACHE"))
val = torch.load(root / "imagenet/clip_cls_patch_std_ViT-B-32_val_blocks_5_11.pt")
y = val["labels"]
stats = torch.load(ck / f"{name}_stats.pt"); mean, std = stats["mean"].expand(2, -1), stats["std"].expand(2, -1)
kae = JordanKoopmanAutoencoder(256, shared_modes=stats["shared_modes"], in_features=2304, latent_features=2049, hidden_config=stats["hidden_config"], bias=True, batchnorm=False, nonlinearity="leaky_relu")
kae.load_state_dict(load_file(ck / f"{name}.safetensors")); kae.eval()
x = (val["states"].float() - mean) / std
clip, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", force_quick_gelu=True, cache_dir=f"{os.getenv('HF_HOME')}/open_clip")
head = torch.load(Path(os.getenv("WEIGHTS_CACHE")) / "tokae/zero_shot_head_ViT-B-32.pt")
def classify(s):
    o = clip.visual.ln_post(s[..., :768] * std[1, :768] + mean[1, :768]) @ clip.visual.proj
    return ((o / o.norm(dim=-1, keepdim=True)) @ head).argmax(-1)
def decode(p):
    return torch.cat([kae.decode(b) for b in p.split(5000)])
def fvu(pred, true):
    return ((pred - true).pow(2).sum() / (true - true.mean(0)).pow(2).sum()).item()
def cos(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)

phi = torch.cat([encode_pairs(kae, b) for b in x.split(5000)])  # [images, 2, latent]
lam = torch.cat([kae.real_eigenvalue.to(torch.cfloat), torch.complex(kae.block_real, kae.block_imag)]).detach()
mode_of_dim = torch.cat([torch.zeros(1, dtype=torch.long), torch.arange(1, 1025).repeat_interleave(2)])
usage = torch.zeros(1025).index_add_(0, mode_of_dim, (phi[:, 0] != 0).float().mean(0)) / torch.bincount(mode_of_dim).float()
shared = usage > 0.95
growth = shared & (lam.abs() > 2)
small = shared & (lam.abs() <= 2)
dims = lambda modes: modes[mode_of_dim]
print(f"{name}: {int(shared.sum())} always-on modes = {int(growth.sum())} growth (|λ| > 2) + {int(small.sum())} small")

# 1. Do the always-on modes alone decode to each block's average state?
m6, m12 = x[:, 0].mean(0), x[:, 1].mean(0)
keep = lambda p, modes: p * dims(modes).float()
d6, d12 = decode(keep(phi[:, 0], shared)), decode(keep(phi[:, 1], shared))
zero = kae.decode(torch.zeros(1, 2049))[0]
print("\n1. Decoding only the always-on modes")
print(f"   block 6:  cosine to block 6's average state, median {cos(d6, m6).median():.3f}; to block 12's average {cos(d6, m12).median():.3f}")
print(f"   block 12: cosine to block 12's average state, median {cos(d12, m12).median():.3f}; to block 6's average {cos(d12, m6).median():.3f}")
print(f"   decoder at zero latent: cosine to block 6's average {cos(zero, m6):.3f}, to block 12's {cos(zero, m12):.3f}; norm {zero.norm():.1f} vs averages {m6.norm():.1f} / {m12.norm():.1f}")
print(f"   FVU of images, using: always-on modes only {fvu(d6, x[:, 0]):.3f} (block 6), {fvu(d12, x[:, 1]):.3f} (block 12); the block average {fvu(m6.expand_as(x[:, 0]), x[:, 0]):.3f} / {fvu(m12.expand_as(x[:, 1]), x[:, 1]):.3f}; all modes {fvu(decode(phi[:, 0]), x[:, 0]):.3f} / {fvu(decode(phi[:, 1]), x[:, 1]):.3f}")
# Same with each kind alone, at the end where it is big
print(f"   growth modes only at block 12: cosine to block 12's average {cos(decode(keep(phi[:, 1], growth)), m12).median():.3f}; small modes only at block 6: cosine to block 6's average {cos(decode(keep(phi[:, 0], small)), m6).median():.3f}")

# 2. Ablations on the prediction path: zero-shot of block 12 predicted from block 6
def predict(p6):
    return decode(kae.koopman_forward(p6[:, None], 1)[:, 0])
base = (classify(predict(phi[:, 0])) == y).double().mean().item()
const = phi[:, 0].clone()
avg = phi[:, 0][:, dims(shared)].mean(0)
const[:, dims(shared)] = avg
rows = {"as trained": base,
        "always-on modes set to their average over images": (classify(predict(const)) == y).double().mean().item(),
        "growth modes set to their average": None, "small always-on modes set to their average": None,
        "always-on modes removed": (classify(predict(keep(phi[:, 0], ~shared))) == y).double().mean().item(),
        "only the always-on modes": (classify(predict(keep(phi[:, 0], shared))) == y).double().mean().item()}
for label, modes in [("growth modes set to their average", growth), ("small always-on modes set to their average", small)]:
    p = phi[:, 0].clone(); p[:, dims(modes)] = phi[:, 0][:, dims(modes)].mean(0); rows[label] = (classify(predict(p)) == y).double().mean().item()
print("\n2. Zero-shot of block 12 predicted from block 6")
for k, v in rows.items():
    print(f"   {k}: {v:.3f}")

# 3. What does the per-image size of the growth modes track?
raw6, raw12 = val["states"][:, 0].float(), val["states"][:, 1].float()
g12 = phi[:, 1][:, dims(growth)].norm(dim=-1)
g6 = phi[:, 0][:, dims(growth)].norm(dim=-1)
s6 = phi[:, 0][:, dims(small)].norm(dim=-1)
r = lambda a, b: np.corrcoef(a.numpy(), b.numpy())[0, 1]
print("\n3. Correlation across images")
print(f"   growth modes' size at block 12 vs block-12 state norm (raw CLS + mean + std): {r(g12, raw12.norm(dim=-1)):.2f}; vs block-12 CLS norm: {r(g12, raw12[:, :768].norm(dim=-1)):.2f}; vs block-6 norm: {r(g12, raw6.norm(dim=-1)):.2f}")
print(f"   growth modes' size at block 6 vs at block 12: {r(g6, g12):.2f}")
print(f"   growth modes vs patch spread at block 12 (std part norm): {r(g12, raw12[:, 1536:].norm(dim=-1)):.2f}; vs mean-patch part: {r(g12, raw12[:, 768:1536].norm(dim=-1)):.2f}")
print(f"   small always-on modes' size at block 6 vs block-6 state norm: {r(s6, raw6.norm(dim=-1)):.2f}; vs block-6 patch spread: {r(s6, raw6[:, 1536:].norm(dim=-1)):.2f}")
torch.save({"g12": g12, "s6": s6, "growth": growth.nonzero()[:, 0], "small": small.nonzero()[:, 0]}, Path(__file__).parent / f"shared_{name}.pt")
