import os, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
from dotenv import load_dotenv
from safetensors.torch import load_file
sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
import train
load_dotenv("/home/nsa325/work/koopmann/.env")
torch.set_grad_enabled(False)
device = "cuda"
save_dir = Path(os.getenv("WEIGHTS_CACHE")) / "tokae"
root = Path(os.getenv("DATASETS_CACHE")) / "imagenette"
kae = train.JordanKoopmanAutoencoder(128, in_features=768, latent_features=1025, hidden_config=[1024], bias=True, batchnorm=False, nonlinearity="leaky_relu")
print(kae.load_state_dict(load_file(save_dir / "b6_jordan128.safetensors")))
kae = kae.to(device).eval()
stats = torch.load(save_dir / "b6_jordan128_stats.pt", map_location=device)
a, b = kae.block_real, kae.block_imag
size = (a.pow(2) + b.pow(2)).sqrt()
angle = torch.atan2(b, a).abs()
print("real eigenvalue", round(kae.real_eigenvalue.item(), 3))
print("block |eigenvalue|: median", round(size.median().item(), 3), "min", round(size.min().item(), 3), "max", round(size.max().item(), 3))
print("block angle: median", round(angle.median().item(), 3), "max", round(angle.max().item(), 3))
print("top 10 sizes", [round(v, 2) for v in size.topk(10).values.tolist()])
data = torch.load(root / "clip_cls_ViT-B-32_val.pt")
labels = data["labels"].to(device)
x = (data["states"].to(device)[:, stats["blocks"]] - stats["mean"]) / stats["std"]
for name, states in [("start", x[:, 0]), ("end", x[:, 1])]:
    phi = kae.encode(states)
    blocks = phi[:, 1:].unflatten(-1, (-1, 2))
    active = blocks.norm(dim=-1) > 0
    print(name, "active modes per image", round(active.sum(1).float().mean().item(), 1), "| modes ever active", (active.any(0)).sum().item(), "of", active.shape[1],
          "| modes active on >1% of images", (active.float().mean(0) > 0.01).sum().item(), "| on >50%", (active.float().mean(0) > 0.5).sum().item())
    use = active.float().mean(0).sort(descending=True).values
    print("   usage of most used modes", [round(v, 2) for v in use[:12].tolist()])
    energy = blocks.pow(2).sum(-1).sum(0); energy = energy / energy.sum()
    print("   energy share of top 1/5/20/64 modes", [round(energy.sort(descending=True).values[:k].sum().item(), 3) for k in (1, 5, 20, 64)])
start, end = kae.encode(x[:, 0]), kae.encode(x[:, 1])
sa = start[:, 1:].unflatten(-1, (-1, 2)).norm(dim=-1) > 0
ea = end[:, 1:].unflatten(-1, (-1, 2)).norm(dim=-1) > 0
print("same modes active at start and end, share of top-k:", round(((sa & ea).sum(1).float() / ea.sum(1).clamp(min=1).float()).mean().item(), 3))
