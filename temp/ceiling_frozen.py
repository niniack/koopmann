"""
The ceiling of the one plan, measured on the real CLIP model.

Our KAE updates the CLS token from block 6 to block 12 and sees the patch tokens only as they are after block 6.
Here the real blocks 7-12 run under the same restriction: after each of them the patch tokens are put back to their
block-6 values, so only the CLS token moves. No model that reads only block-6 tokens should beat this by much.

Usage: ceiling_frozen.py [<number of validation images, default 10000>]
"""

import sys

import torch
from torch.utils.data import DataLoader, Subset

import control_train as base

count = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
torch.set_grad_enabled(False)
device = base.get_device()
teacher = base.Teacher("ViT-B-32", device)
blocks = teacher.model.visual.transformer.resblocks
classes = torch.load(base.HERE / "zero_shot_head.pt").float().to(device)
frozen = {"on": False, "patches": None}


def keep_patches(_, __, output):
    frozen["patches"] = output[:, 1:]


def freeze_patches(_, __, output):
    if frozen["on"]:
        return torch.cat([output[:, :1], frozen["patches"]], dim=1)


blocks[5].register_forward_hook(keep_patches)
for block in blocks[6:]:
    block.register_forward_hook(freeze_patches)

answers = {False: [], True: []}
labels = []
loader = DataLoader(Subset(base.Views("val", teacher, 0), range(count)), batch_size=250, num_workers=8)
for images, label in loader:
    pixels = (images[:, 0].to(device).float() / 255 - teacher.pixel_mean) / teacher.pixel_std
    for frozen["on"] in (False, True):
        embedding = torch.nn.functional.normalize(teacher.model.encode_image(pixels), dim=-1)
        answers[frozen["on"]].append((embedding @ classes).argmax(dim=-1).cpu())
    labels.append(label)

labels, real, ceiling = torch.cat(labels), torch.cat(answers[False]), torch.cat(answers[True])
print(f"{count} validation images")
print(f"real CLIP: accuracy {(real == labels).float().mean():.4f}")
print(f"patch tokens frozen at block 6: accuracy {(ceiling == labels).float().mean():.4f}, agreement with real CLIP {(ceiling == real).float().mean():.4f}")
