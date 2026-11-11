"""
CIFAR-100 through the real CLIP model, for the adaptation test: states after blocks 6 and 12 and patch tokens after block 6.
Images get CLIP's own preprocessing (resize to 224, center crop). Saved to domain_cache/cifar100_{train,test}.pt.
"""

import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100

from control_train import HERE, Teacher

torch.set_num_threads(16)
teacher = Teacher("ViT-B-32", "cpu")
preprocess = transforms.Compose([*teacher.resize, teacher.to_rgb, transforms.PILToTensor()])
(HERE / "domain_cache").mkdir(exist_ok=True)

for split in ["train", "test"]:
    images = CIFAR100(os.getenv("DATASETS_CACHE"), train=split == "train", transform=preprocess)
    states, patches, labels = [], [], []
    for batch, batch_labels in DataLoader(images, batch_size=256, num_workers=8):
        batch_states, batch_patches = teacher(batch)
        states.append(batch_states)
        patches.append(batch_patches.half())
        labels.append(batch_labels)
    torch.save({"states": torch.cat(states), "patches": torch.cat(patches), "labels": torch.cat(labels), "classes": images.classes}, HERE / "domain_cache" / f"cifar100_{split}.pt")
    print(split, len(images), flush=True)
