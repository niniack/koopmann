"""
The hoofed-species set: 49 wild species of antelope, deer, sheep, goat and cattle, from the local slice of ImageNet-21k
(found by scan_w21.py). Only species are kept: no parent groups ("antelope", "deer"), no age, sex or breed classes.
Runs an encoder over the images with train.py's own extraction and saves the states after blocks 6 and 12.

Usage: extract_hoofed.py <encoder: ViT-B-32 or dinov3-vitb16>
"""

import sys
import tarfile
from pathlib import Path

import torch
from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
load_dotenv("/home/nsa325/work/koopmann/.env")
import train  # noqa: E402

HERE = Path(__file__).parent
SHARDS = Path("/scratch/nsa325/datasets/imagenet-w21-wds")
SPECIES = {
    "antelope": "n02420509 n02420828 n02421136 n02421449 n02421792 n02422106 n02422391 n02422699 n02423218 n02423589 n02424085 n02424486 "
    "n02424909 n02425228 n02425887 n02426481 n02427470 n02427576 n02427724 n02428508 n02429456",
    "deer": "n02431628 n02431785 n02431976 n02432291 n02432511 n02432704 n02432983 n02433318 n02433546 n02433925 n02434190 n02434954",
    "sheep and goat": "n02411206 n02415253 n02415577 n02415829 n02416104 n02417914 n02418465 n02419336 n02419634",
    "cattle": "n02404906 n02405302 n02408429 n02409508 n02410011 n02410702 n02410900",
}
SYNSETS = " ".join(SPECIES.values()).split()


class Hoofed(IterableDataset):
    """The species' images with their label (the species' place in SYNSETS), shard by shard."""

    def __init__(self, transform):
        self.names = torch.load(HERE / "w21_hoofed.pt")["names"]
        self.transform = transform

    def __iter__(self):
        worker = get_worker_info()
        for shard in sorted(self.names)[worker.id :: worker.num_workers]:
            wanted = {name for name in self.names[shard] if name.split("_")[0] in SYNSETS}
            with tarfile.open(SHARDS / shard) as tar:
                for member in tar:
                    if member.name in wanted:
                        yield self.transform(Image.open(tar.extractfile(member)).convert("RGB")), SYNSETS.index(member.name.split("_")[0])


if __name__ == "__main__":
    encoder = sys.argv[1]
    torch.set_num_threads(16)
    # NOTE: train.extract_states builds its dataset by name, so the "w21" dataset is swapped for this one. The rest is its own code
    train.ShardImages = lambda folder, pattern, transform, skip: Hoofed(transform)
    states, labels = train.extract_states(encoder, "w21", "train", None, "cpu", views=1, blocks=[5, 11], tokens="cls_patch_std")
    synsets = torch.load(HERE / "w21_hoofed.pt")["synsets"]
    torch.save({"states": states, "labels": labels, "names": [synsets[s]["name"].split(",")[0] for s in SYNSETS]}, HERE / f"hoofed_{encoder}.pt")
    print(encoder, states.shape, labels.bincount().tolist())
