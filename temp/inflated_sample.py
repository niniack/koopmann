"""How many CC3M images get inflated latents, and what a random sample of them looks like."""
import io, os, sys, tarfile, torch, numpy as np, open_clip, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
from safetensors.torch import load_file
sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
from train import JordanKoopmanAutoencoder, encode_pairs
load_dotenv("/home/nsa325/work/koopmann/.env")
torch.set_grad_enabled(False); torch.manual_seed(0); torch.set_num_threads(16)
name, ckpt = sys.argv[1], Path(sys.argv[2])
stats = torch.load(ckpt / f"{name}_stats.pt")
mean, std = stats["mean"].expand(2, -1), stats["std"].expand(2, -1)
kae = JordanKoopmanAutoencoder(256, shared_modes=True, in_features=2304, latent_features=2049, hidden_config=stats["hidden_config"], bias=True, batchnorm=False, nonlinearity="leaky_relu")
kae.load_state_dict(load_file(ckpt / f"{name}.safetensors")); kae.eval()
clip, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", force_quick_gelu=True, cache_dir=f"{os.getenv('HF_HOME')}/open_clip"); clip.eval()
raw = []
for shard in sorted((Path(os.getenv("DATASETS_CACHE")) / "cc3m-wds").glob("cc3m-train-*.tar")):
    with tarfile.open(shard) as tar:
        for m in tar:
            if m.name.endswith(".jpg"):
                raw.append(tar.extractfile(m).read())
                if len(raw) == 10000: break
    if len(raw) == 10000: break
def states(pix):
    out = []
    def keep(_, __, o):
        p = o[:, 1:]; out.append(torch.cat([o[:, 0], p.mean(1), p.std(1)], -1))
    hs = [clip.visual.transformer.resblocks[b].register_forward_hook(keep) for b in stats["blocks"]]
    clip.encode_image(pix)
    for h in hs: h.remove()
    return torch.stack(out, 1)
op = lambda i: Image.open(io.BytesIO(raw[i])).convert("RGB")
x = torch.cat([states(torch.stack([preprocess(op(i)) for i in range(s, min(s + 250, len(raw)))])) for s in range(0, len(raw), 250)])
x = (x - mean) / std
phi = torch.cat([encode_pairs(kae, b) for b in x.split(5000)])
e = phi[:, 1].pow(2).sum(-1)
for t in [1e3, 1e4, 1e5]:
    sel = e > t
    print(f"block-12 latent energy > {t:.0e}: {int(sel.sum())} of {len(e)} CC3M images ({sel.double().mean():.1%}), holding {e[sel].sum() / e.sum():.3f} of all energy")
pool = (e > 1e4).nonzero()[:, 0]
pick = pool[torch.randperm(len(pool))[:24]]
fig, axes = plt.subplots(3, 8, figsize=(16, 6.5))
for ax, i in zip(axes.flat, pick.tolist()):
    im = op(i); im.thumbnail((256, 256)); ax.imshow(np.asarray(im)); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(f"energy {e[i]:.0e}", fontsize=8)
fig.suptitle("24 random CC3M images with block-12 latent energy > 1e4", x=0.01, ha="left")
fig.tight_layout(); fig.savefig(sys.argv[3], dpi=70)
