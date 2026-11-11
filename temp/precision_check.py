"""
TEMP: does faster arithmetic change what real CLIP says? Real CLIP runs on validation images in fp32, with TF32 matrix
products, and in bf16. Each is compared with fp32: speed, zero-shot accuracy, how often the answer is fp32's, and the
FVU against fp32 of what the KAE trains on (the CLS token after blocks 6 and 12, the patch tokens after block 6).

Usage: precision_check.py [<number of validation images, default 10000>]
"""

import os
import sys
import time

import torch
from torch.utils.data import DataLoader, Subset

import control_train as base
from control_train import train

count = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
device = base.get_device()
teacher = base.Teacher("ViT-B-32", device)
width = teacher.model.visual.ln_post.normalized_shape[0]
stats = torch.load(f"{os.getenv('WEIGHTS_CACHE')}/tokae/im_iso_2x4k_stats.pt")
mean, std = stats["mean"][:width].to(device), stats["std"][:width].to(device)
head = train.ClipHead("ViT-B-32", mean, std, device, zero_shot=True)
loader = DataLoader(Subset(base.Views("val", teacher, 0), range(count)), batch_size=1_000, num_workers=16)
batches = [(images[:, 0].to(device), labels.to(device)) for images, labels in loader]
labels = torch.cat([label for _, label in batches])


def run(precision):
    """Real CLIP under one arithmetic: standardized CLS tokens [images, 2, width], patch tokens [tokens, 1, width], images/s."""

    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = precision != "fp32"
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
        teacher(batches[0][0])
        torch.cuda.synchronize()
        start = time.time()
        outputs = [teacher(images) for images, _ in batches]
        torch.cuda.synchronize()
    rate = count / (time.time() - start)
    states = (torch.cat([s for s, _ in outputs])[..., :width].float() - mean) / std

    return states, torch.cat([p for _, p in outputs]).float().flatten(0, 1)[:, None], rate


exact = None
for precision in ("fp32", "tf32", "bf16"):
    states, patches, rate = run(precision)
    exact = exact or (states, patches, head.classify(states[:, 1]))
    answers = head.classify(states[:, 1])
    fvu = train.compute_fvu(states, exact[0]).tolist() + train.compute_fvu(patches, exact[1]).tolist()
    print(
        f"{precision}: {rate:.0f} images/s, accuracy {(answers == labels).float().mean():.4f}, same answer as fp32 {(answers == exact[2]).float().mean():.4f}, "
        f"FVU against fp32: CLS after block 6 {fvu[0]:.1e}, CLS after block 12 {fvu[1]:.1e}, patches after block 6 {fvu[2]:.1e}",
        flush=True,
    )
