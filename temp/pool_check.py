"""On a subset, how much more do finer poolings of block 6's patches tell a linear map about block 12's CLS?"""
import os, sys, time, torch, torch.nn.functional as F, open_clip
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Subset
sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
load_dotenv("/home/nsa325/work/koopmann/.env")
from train import ImageNet, fit_linear, apply_linear
torch.set_grad_enabled(False); torch.set_num_threads(16); torch.manual_seed(0)
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", force_quick_gelu=True, cache_dir=f"{os.getenv('HF_HOME')}/open_clip")
model.eval()
head = torch.load(os.getenv("WEIGHTS_CACHE") + "/tokae/zero_shot_head_ViT-B-32.pt")
out = {}
def keep6(_, __, o):
    patches = o[:, 1:]
    grid = patches.transpose(1, 2).unflatten(-1, (7, 7))
    out["cls"], out["mean"], out["std"] = o[:, 0], patches.mean(1), patches.std(1)
    out["2x2"] = F.adaptive_avg_pool2d(grid, 2).flatten(2).transpose(1, 2).flatten(1)
    out["3x3"] = F.adaptive_avg_pool2d(grid, 3).flatten(2).transpose(1, 2).flatten(1)
def keep12(_, __, o):
    out["end"] = o[:, 0]
model.visual.transformer.resblocks[5].register_forward_hook(keep6)
model.visual.transformer.resblocks[11].register_forward_hook(keep12)

def run(split, n):
    data = ImageNet(split, preprocess)
    idx = torch.randperm(len(data))[:n].tolist()
    rows, labels, start = {}, [], time.time()
    for i, (images, y) in enumerate(DataLoader(Subset(data, idx), batch_size=250, num_workers=8)):
        model.encode_image(images)
        for k, v in out.items():
            rows.setdefault(k, []).append(v.half())
        labels.append(y)
        if i % 40 == 0: print(split, i * 250, f"{time.time() - start:.0f}s", flush=True)
    return {k: torch.cat(v).float() for k, v in rows.items()}, torch.cat(labels)

train, ty = run("train", 100_000)
val, vy = run("val", 10_000)
torch.save({"train": train, "ty": ty, "val": val, "vy": vy}, os.path.dirname(__file__) + "/pool_check.pt")

def classify(cls):
    o = model.visual.ln_post(cls) @ model.visual.proj
    return (o / o.norm(dim=-1, keepdim=True) @ head).argmax(-1)
clip_answers = classify(val["end"])
print("real CLIP zero-shot on this val subset", (clip_answers == vy).float().mean().item())
inputs = {"CLS": ["cls"], "CLS + mean": ["cls", "mean"], "CLS + mean + std": ["cls", "mean", "std"],
          "CLS + 2x2": ["cls", "2x2"], "CLS + mean + 2x2": ["cls", "mean", "2x2"], "CLS + 3x3": ["cls", "3x3"]}
for name, keys in inputs.items():
    x, vx = torch.cat([train[k] for k in keys], -1), torch.cat([val[k] for k in keys], -1)
    mu, sd = x.mean(0), x.std(0) + 1e-6
    x, vx = (x - mu) / sd, (vx - mu) / sd
    linear_map = fit_linear(x, train["end"], reg=100.0)
    answers = classify(apply_linear(linear_map, vx))
    print(f"{name:20s} {x.shape[1]:5d} dims: zero-shot {(answers == vy).float().mean():.3f}, agrees with CLIP {(answers == clip_answers).float().mean():.3f}", flush=True)
