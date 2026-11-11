"""Does adding block 6's mean patch token help a linear map predict block 12's CLS, read with CLIP's head?"""
import os, sys
import open_clip, torch, torch.nn.functional as F
from dotenv import load_dotenv
sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
load_dotenv("/home/nsa325/work/koopmann/.env")
from train import fit_linear, apply_linear

root = os.getenv("DATASETS_CACHE") + "/imagenet"
train = torch.load(f"{root}/clip_cls_patch_ViT-B-32_train_blocks_5_11.pt")
val = torch.load(f"{root}/clip_cls_patch_ViT-B-32_val_blocks_5_11.pt")
x, y, vx, vy = train["states"], train["labels"], val["states"], val["labels"]
mean, std = x.mean(dim=[0, 1]), x.std(dim=[0, 1])
x, vx = (x - mean) / std, (vx - mean) / std

model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", force_quick_gelu=True, cache_dir=f"{os.getenv('HF_HOME')}/open_clip")
head = torch.load(os.getenv("WEIGHTS_CACHE") + "/tokae/zero_shot_head_ViT-B-32.pt")

@torch.no_grad()
def classify(cls):
    cls = cls * std[:768] + mean[:768]
    return (F.normalize(model.visual.ln_post(cls) @ model.visual.proj, dim=-1) @ head).argmax(dim=-1)

clip_answers = classify(vx[:, 1, :768])
one_hot = F.one_hot(y, 1000).float()
inputs = {"CLS": slice(0, 768), "mean patch": slice(768, 1536), "CLS + mean patch": slice(0, 1536)}
print("real CLIP zero-shot", (clip_answers == vy).float().mean().item())
for name, s in inputs.items():
    probe = fit_linear(x[:, 0, s], one_hot)
    probe_acc = (apply_linear(probe, vx[:, 0, s]).argmax(-1) == vy).float().mean().item()
    linear_map = fit_linear(x[:, 0, s], x[:, 1, :768])
    pred = apply_linear(linear_map, vx[:, 0, s])
    target = vx[:, 1, :768]
    fvu = ((pred - target).pow(2).sum() / (target - target.mean(0)).pow(2).sum()).item()
    answers = classify(pred)
    print(f"{name:18s} probe on block 6 {probe_acc:.3f} | linear map to block-12 CLS: fvu {fvu:.3f}, zero-shot {(answers == vy).float().mean():.3f}, agrees with CLIP {(answers == clip_answers).float().mean():.3f}", flush=True)
