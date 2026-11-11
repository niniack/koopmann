import os, sys, torch
sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
from dotenv import load_dotenv
load_dotenv("/home/nsa325/work/koopmann/.env")
from safetensors.torch import load_file
from train import JordanKoopmanAutoencoder
import open_clip
torch.set_grad_enabled(False)
save_dir = os.getenv("WEIGHTS_CACHE") + "/tokae/"
name = "im_jordan256_2k_both"
stats = torch.load(save_dir + f"{name}_stats.pt")
kae = JordanKoopmanAutoencoder(256, in_features=768, latent_features=2049, hidden_config=[2048], bias=True, batchnorm=False, nonlinearity="leaky_relu")
kae.load_state_dict(load_file(save_dir + f"{name}.safetensors")); kae.eval()
val = torch.load(os.getenv("DATASETS_CACHE") + "/imagenet/clip_cls_ViT-B-32_val_blocks_5_11.pt")
x = (val["states"] - stats["mean"]) / stats["std"]; y = val["labels"]
phi = torch.cat([kae.encode(b) for b in x[:, 0].split(5000)])
coef = torch.cat([phi[:, :1], phi[:, 1:].unflatten(-1, (-1, 2)).norm(dim=-1)], dim=1)  # [images, modes]
eig = torch.cat([kae.real_eigenvalue.to(torch.cfloat), torch.complex(kae.block_real, kae.block_imag)])
freq = (coef > 0).float().mean(0)
names = open_clip.IMAGENET_CLASSNAMES
rows = []
for m in range(coef.shape[1]):
    if freq[m] < 0.003: continue
    top = coef[:, m].topk(50).indices
    counts = torch.bincount(y[top], minlength=1000)
    rows.append((m, freq[m].item(), eig[m].abs().item(), (counts > 0).sum().item(), counts.max().item() / 50, names[counts.argmax()]))
torch.save({"coef": coef, "eig": eig, "y": y}, os.path.join(sys.argv[1], "modes_val.pt"))
print("modes used at all:", (freq > 0).sum().item(), "of", len(freq), "| used on >= 0.3% of images:", len(rows))
rows.sort(key=lambda r: r[3], reverse=True)
print("most cross-class (distinct classes in top 50):")
for r in rows[:30]: print(f"mode {r[0]:4d} on {100*r[1]:5.1f}%  |eig| {r[2]:.2f}  classes {r[3]:2d}  top class {100*r[4]:.0f}% {r[5]}")
print("most class-pure:")
for r in sorted(rows, key=lambda r: -r[4])[:10]: print(f"mode {r[0]:4d} on {100*r[1]:5.1f}%  |eig| {r[2]:.2f}  classes {r[3]:2d}  top class {100*r[4]:.0f}% {r[5]}")
import collections
print("distinct-class histogram:", sorted(collections.Counter(min(r[3] // 10 * 10, 50) for r in rows).items()))
for m in [903, 878]:
    r = [r for r in rows if r[0] == m][0]; print("named", r)
