"""States of grayscale and sepia copies of 2,000 validation images, to see how losing or shifting colour shows up in the modes."""
import os, sys
from pathlib import Path
import torch
from datasets import load_dataset
from torchvision import transforms
import inversion
from inversion import train

torch.set_num_threads(16)
torch.set_grad_enabled(False)
COUNT = 2000
# Luminance weights, and the usual sepia matrix (rows give the new red, green and blue)
GRAY = torch.tensor([0.299, 0.587, 0.114])
SEPIA = torch.tensor([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]])


def to_gray(pictures):
    return torch.einsum("c,nchw->nhw", GRAY, pictures)[:, None].expand(-1, 3, -1, -1)


def to_sepia(pictures):
    return torch.einsum("dc,nchw->ndhw", SEPIA, pictures).clamp(0, 1)


if __name__ == "__main__":
    kae, stats = inversion.load_kae()
    reader = inversion.Reader(stats["mean"], stats["std"])
    root = Path(os.getenv("DATASETS_CACHE"))
    val = torch.load(root / "imagenet" / f"clip_cls_patch_std_{inversion.ENCODER}_val_blocks_5_11.pt")
    hidden = torch.load("hidden_mask.pt")
    hidden[22037] = True
    chosen = (~hidden).nonzero()[:, 0][torch.randperm(int((~hidden).sum()), generator=torch.Generator().manual_seed(0))[:COUNT]].sort().values
    images = load_dataset("ILSVRC/imagenet-1k", split="validation", cache_dir=os.getenv("DATASETS_CACHE"))
    to_picture = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC), transforms.CenterCrop(224), train.to_rgb, transforms.ToTensor()])
    states = {"original": [], "gray": [], "sepia": []}
    colourfulness = []
    for start in range(0, COUNT, 50):
        pictures = torch.stack([to_picture(images[i]["image"]) for i in chosen[start : start + 50].tolist()])
        colourfulness.append(inversion.get_colourfulness(pictures))
        states["original"].append(reader.raw_states(pictures))
        states["gray"].append(reader.raw_states(to_gray(pictures)))
        states["sepia"].append(reader.raw_states(to_sepia(pictures)))
        print(start, flush=True)
    states = {key: torch.cat(value) for key, value in states.items()}
    cached = val["states"][chosen].float()
    gap = ((states["original"] - cached).norm(dim=-1) / cached.norm(dim=-1)).max().item()
    print(f"largest relative gap to the cached states: {gap:.2e}")
    torch.save({"indices": chosen, "labels": val["labels"][chosen], "colourfulness": torch.cat(colourfulness), **states}, "alter_states.pt")
