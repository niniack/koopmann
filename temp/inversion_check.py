"""Checks that the reader rebuilds the cached states, and times inversion steps on this device."""
import os, sys, time
from pathlib import Path
import torch
from datasets import load_dataset
from torchvision import transforms
import inversion
from inversion import train

threads = int(sys.argv[1]) if len(sys.argv) > 1 else 8
count = int(sys.argv[2]) if len(sys.argv) > 2 else 1
torch.set_num_threads(threads)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kae, stats = inversion.load_kae()
reader = inversion.Reader(stats["mean"], stats["std"]).to(device)
root = Path(os.getenv("DATASETS_CACHE"))
val = torch.load(root / "imagenet" / f"clip_cls_patch_std_{inversion.ENCODER}_val_blocks_5_11.pt")
images = load_dataset("ILSVRC/imagenet-1k", split="validation", cache_dir=os.getenv("DATASETS_CACHE"))
# The cache's preprocessing without its normalization, which the reader applies itself
to_picture = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC), transforms.CenterCrop(224), train.to_rgb, transforms.ToTensor()])
with torch.no_grad():
    check = reader.raw_states(torch.stack([to_picture(images[i]["image"]) for i in range(4)]).to(device)).cpu()
cached = val["states"][:4].float()
gap = ((check - cached).norm(dim=-1) / cached.norm(dim=-1)).max().item()
print(f"largest relative gap to the cached states: {gap:.2e}")
assert gap < 1e-2
targets = ((cached[:count, 1] - stats["mean"]) / stats["std"]).to(device)
print("generator weights:", sum(p.numel() for p in inversion.Generator().parameters()))
start = time.time()
inversion.invert(reader, targets, "all", 20, list(range(count)), log_every=10, label=f"{threads} threads, {count} targets")
print(f"{(time.time() - start) / 20:.3f} s per step")
