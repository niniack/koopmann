"""CPU pilot: paints one validation image's real block-12 state, to pick the settings before the GPU batch."""
import os, sys
from pathlib import Path
import torch
from datasets import load_dataset
from torchvision import transforms
from torchvision.utils import save_image
import inversion
from inversion import train

index, part, steps = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
torch.set_num_threads(8)
kae, stats = inversion.load_kae()
reader = inversion.Reader(stats["mean"], stats["std"])
root = Path(os.getenv("DATASETS_CACHE"))
val = torch.load(root / "imagenet" / f"clip_cls_patch_std_{inversion.ENCODER}_val_blocks_5_11.pt")
target = (val["states"][index : index + 1, 1].float() - stats["mean"]) / stats["std"]
pictures, _, history = inversion.invert(reader, target, part, steps, [0], log_every=250, label=f"{index} {part}")
save_image(pictures, f"inv/pilot_{index}_{part}_{steps}.png")
with torch.no_grad():
    error = (reader(pictures) - target).pow(2).mean(dim=1) / target.pow(2).mean(dim=1)
print(f"state error left, as a share of the target's size: {error.item():.3f}", flush=True)
images = load_dataset("ILSVRC/imagenet-1k", split="validation", cache_dir=os.getenv("DATASETS_CACHE"))
to_picture = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC), transforms.CenterCrop(224), train.to_rgb, transforms.ToTensor()])
save_image(to_picture(images[index]["image"]), f"inv/original_{index}.png")
