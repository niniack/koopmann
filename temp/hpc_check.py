"""
TEMP: checks a GPU node before any training. GPU, ImageNet offline, and how fast fresh crops arrive against how fast
real CLIP digests them (the data loader may be the bottleneck, not the GPU).

Usage: hpc_check.py [<workers, default 16>]
"""

import os
import sys
import time

import torch
from torch.utils.data import DataLoader

import control_train as base

workers = int(sys.argv[1]) if len(sys.argv) > 1 else 16
device = base.get_device()
print(f"torch {torch.__version__}, cuda {torch.version.cuda}, gpu {torch.cuda.get_device_name()}, cpus {os.cpu_count()}", flush=True)

teacher = base.Teacher("ViT-B-32", device)
data = base.Views("train", teacher, 2)
print(f"ImageNet train images: {len(data)}", flush=True)

# Loader alone: batches of 512 images, 2 crops each, as in cls_train.py
loader = DataLoader(data, 512, shuffle=True, drop_last=True, num_workers=workers, persistent_workers=True)
batches = iter(loader)
next(batches)
start, count = time.time(), 10
for _ in range(count):
    images, _ = next(batches)
loader_rate = count * images.shape[0] * 2 / (time.time() - start)
print(f"loader with {workers} workers: {loader_rate:.0f} crops/s", flush=True)

# Real CLIP alone on 1,024 crops already on the GPU
crops = images.flatten(0, 1).to(device)
teacher(crops)
torch.cuda.synchronize()
start = time.time()
for _ in range(count):
    teacher(crops)
torch.cuda.synchronize()
print(f"real CLIP (teacher): {count * len(crops) / (time.time() - start):.0f} crops/s", flush=True)
print(f"one training step of 1,024 crops needs at least {len(crops) / loader_rate:.2f} s of loading", flush=True)
