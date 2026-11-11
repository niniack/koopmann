"""
Builds the ImageNet-21k extra cache for training: runs the encoder over the downloaded shards (one center crop each, 1k classes skipped),
then drops images whose final embedding is near-identical to an ImageNet-1k val image's, and saves the cache in place.
Usage: prep_w21.py [encoder], ViT-B-32 by default.
"""
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
import train  # noqa: E402

THRESHOLD = 0.95
ENCODER = sys.argv[1] if len(sys.argv) > 1 else "ViT-B-32"
start = time.time()
device = train.get_device()
root = Path(os.getenv("DATASETS_CACHE"))
path = root / "w21" / f"clip_cls_patch_std_{ENCODER}_train_blocks_5_11.pt"
assert not path.exists(), f"{path} already exists"

states, _ = train.get_states(ENCODER, "w21", "train", device, [5, 11], tokens="cls_patch_std")
print(f"extracted {len(states)} images in {(time.time() - start) / 60:.1f} min", flush=True)

# The encoder's own final embedding (CLIP's image embedding, or DINOv3's final CLS output), from raw states
head = (train.DinoHead if ENCODER in train.DINO_MODELS else train.ClipHead)(ENCODER, torch.zeros(768), torch.ones(768), device, zero_shot=False)


@torch.no_grad()
def embed(cls):
    return torch.cat([head.embed(batch.to(device)) for batch in cls.split(50_000)])


# Block 12's CLS token, raw, for both sets
val = torch.load(root / "imagenet" / f"clip_cls_patch_std_{ENCODER}_val_blocks_5_11.pt")
val_embeddings = embed(val["states"][:, 1, :768].float())
nearest = torch.cat([(embed(batch[:, 1, :768].float()) @ val_embeddings.T).max(dim=1).values.cpu() for batch in states.split(10_000)])
for t in [0.9, 0.95, 0.98, 0.99]:
    print(f"images with a val image at cosine > {t}: {int((nearest > t).sum())}", flush=True)

keep = nearest <= THRESHOLD
torch.save({"dropped": (~keep).nonzero()[:, 0], "nearest": nearest}, Path(__file__).parent / f"w21_dedupe_{ENCODER}.pt")
torch.save({"states": states[keep], "labels": torch.full((int(keep.sum()),), -1)}, path)
print(f"saved {int(keep.sum())} images ({int((~keep).sum())} dropped) to {path} after {(time.time() - start) / 60:.1f} min", flush=True)
