"""
States after every block from 6 to 12 of DINOv3, for the 2,000 validation images of alter_states.py.

The KAE was trained on blocks 6 and 12 only. These states are for reading the blocks in between through its encoder,
to see whether a mode's value moves steadily from its block-6 value to its block-12 value. Nothing is trained on them.
"""

import os
from pathlib import Path

import torch
from datasets import load_dataset
from torchvision import transforms

import inversion
from inversion import train

torch.set_num_threads(6)
torch.set_grad_enabled(False)
BLOCKS = list(range(5, 12))  # 0-based: the outputs of blocks 6 to 12

kae, stats = inversion.load_kae()
reader = inversion.Reader(stats["mean"], stats["std"])
chosen = torch.load("alter_states.pt")["indices"]
images = load_dataset("ILSVRC/imagenet-1k", split="validation", cache_dir=os.getenv("DATASETS_CACHE"))
to_picture = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        train.to_rgb,
        transforms.ToTensor(),
    ]
)


def get_all_states(pictures):
    """Unstandardized states after each block of BLOCKS. shape: [pictures, blocks, 2304]"""

    states = []

    def keep(_, __, output):
        output = output[0] if isinstance(output, tuple) else output
        patches = output[:, reader.first_patch :]
        states.append(torch.cat([output[:, 0], patches.mean(dim=1), patches.std(dim=1)], dim=-1))

    hooks = [reader.dino.model.layer[block].register_forward_hook(keep) for block in BLOCKS]
    reader.dino(pixel_values=reader.normalize(pictures))
    for hook in hooks:
        hook.remove()

    return torch.stack(states, dim=1)


states = []
for start in range(0, len(chosen), 50):
    pictures = torch.stack([to_picture(images[i]["image"]) for i in chosen[start : start + 50].tolist()])
    states.append(get_all_states(pictures))
    print(start, flush=True)
states = torch.cat(states)
root = Path(os.getenv("DATASETS_CACHE"))
cached = torch.load(root / "imagenet" / f"clip_cls_patch_std_{inversion.ENCODER}_val_blocks_5_11.pt")
gap = (states[:, [0, -1]] - cached["states"][chosen].float()).norm(dim=-1) / cached["states"][chosen].float().norm(dim=-1)
print(f"largest relative gap to the cached states of blocks 6 and 12: {gap.max().item():.2e}")
torch.save({"indices": chosen, "labels": cached["labels"][chosen], "blocks": BLOCKS, "states": states}, "depth_states.pt")
