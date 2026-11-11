"""
Finds the hoofed-animal images (deer, antelope, cattle, sheep, horses, ...) in the local slice of ImageNet-21k.

WordNet numbers the hoofed-animal synsets in one run, from `ungulate` up to the last giraffe relative, so a synset
belongs if its number falls in that run. ImageNet-1k's own classes are kept too (they are marked).
Saves animals/w21_hoofed.pt: the image names per shard, and the synsets with their names and counts.
"""

import tarfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from timm.data import ImageNetInfo

SHARDS = sorted(Path("/scratch/nsa325/datasets/imagenet-w21-wds").glob("imagenet_w21-train-*.tar"))
FIRST, LAST = "n02370806", "n02439398"  # `ungulate` to `okapi`


def get_names(shard):
    """Names of the hoofed-animal images in one shard."""

    with tarfile.open(shard) as tar:
        return [m.name for m in tar if m.name.endswith(".jpg") and FIRST <= m.name.split("_")[0] <= LAST]


if __name__ == "__main__":
    with ProcessPoolExecutor(48) as pool:
        names = dict(zip([s.name for s in SHARDS], pool.map(get_names, SHARDS)))
    counts = Counter(name.split("_")[0] for shard in names.values() for name in shard)
    info, in_1k = ImageNetInfo("imagenet-21k"), set(ImageNetInfo("imagenet-1k").label_names())
    synsets = {s: {"name": info.label_name_to_description(s), "count": c, "in_1k": s in in_1k} for s, c in sorted(counts.items())}
    torch.save({"names": names, "synsets": synsets}, Path(__file__).parent / "w21_hoofed.pt")
    print(len(synsets), "synsets,", sum(counts.values()), "images")
    for synset, row in synsets.items():
        print(synset, row["count"], "1k" if row["in_1k"] else "  ", row["name"])
