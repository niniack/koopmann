"""
The ImageNet-shift test sets through the real CLIP model: states after blocks 6 and 12, patch tokens after block 6.
Sets: ImageNet-R, ImageNet-Sketch, ImageNet-A, ImageNet-V2 (clip-benchmark's webdataset shards). Labels are stored as
ImageNet-1k class indices. Images get CLIP's own preprocessing. Saved to domain_cache/<set>.pt.
"""

import sys
import tarfile
from pathlib import Path

import open_clip
import torch
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))  # NOTE: Run with `python -I`, which leaves this folder off the path
from control_train import HERE, Teacher  # noqa: E402

ROOT = Path("/scratch/nsa325/datasets/shift")
SETS = ["imagenetv2", "imagenet-r", "imagenet_sketch"]  # NOTE: imagenet-a is already extracted


class Shards(IterableDataset):
    """Images and labels from webdataset shards, where each image (jpg, webp or png) comes after its label file `x.cls`."""

    def __init__(self, folder, transform):
        self.shards = sorted(folder.glob("test/*.tar"))
        self.transform = transform
        # NOTE: A set's labels count its own class list, so they are mapped to ImageNet-1k's by class name
        names = (folder / "classnames.txt").read_text().splitlines()
        self.to_imagenet = [open_clip.IMAGENET_CLASSNAMES.index(name) for name in names]

    def __iter__(self):
        # NOTE: Each loader worker reads its own shards, as in train.ShardImages
        worker = get_worker_info()
        for shard in self.shards[worker.id :: worker.num_workers]:
            with tarfile.open(shard) as tar:
                for member in tar:
                    if member.name.endswith(".cls"):
                        label = self.to_imagenet[int(tar.extractfile(member).read())]
                    else:
                        yield self.transform(Image.open(tar.extractfile(member))), label


def extract(teacher, images):
    """Runs the real model over a dataset. Returns states, half-precision patch tokens and labels."""

    states, patches, labels = [], [], []
    for batch, batch_labels in DataLoader(images, batch_size=256, num_workers=8):
        batch_states, batch_patches = teacher(batch)
        states.append(batch_states)
        patches.append(batch_patches.half())
        labels.append(batch_labels)
    return {"states": torch.cat(states), "patches": torch.cat(patches), "labels": torch.cat(labels)}


if __name__ == "__main__":
    torch.set_num_threads(12)
    teacher = Teacher("ViT-B-32", "cpu")
    preprocess = transforms.Compose([*teacher.resize, teacher.to_rgb, transforms.PILToTensor()])
    for name in SETS:
        data = extract(teacher, Shards(ROOT / name, preprocess))
        torch.save(data, HERE / "domain_cache" / f"{name}.pt")
        print(name, len(data["labels"]), flush=True)
