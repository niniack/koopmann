"""CLS token and mean patch token after blocks 6 and 12, one center crop per image."""
import os, sys, time
from pathlib import Path
import open_clip, torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader
sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
load_dotenv("/home/nsa325/work/koopmann/.env")
from train import ImageNet

split = sys.argv[1]
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai", force_quick_gelu=True, cache_dir=f"{os.getenv('HF_HOME')}/open_clip"
)
model = model.cuda().eval()
states = []
resblocks = model.visual.transformer.resblocks
hooks = [
    resblocks[i].register_forward_hook(
        lambda _, __, out: states.append(torch.cat([out[:, 0], out[:, 1:].mean(dim=1)], dim=-1).cpu())
    )
    for i in [5, 11]
]
loader = DataLoader(ImageNet(split, preprocess), batch_size=256, num_workers=24)
all_states, all_labels, start = [], [], time.time()
with torch.no_grad():
    for i, (images, labels) in enumerate(loader):
        states.clear()
        model.encode_image(images.cuda())
        all_states.append(torch.stack(states, dim=1))
        all_labels.append(labels)
        if i % 200 == 0:
            print(i * 256, f"{time.time() - start:.0f}s", flush=True)
path = Path(os.getenv("DATASETS_CACHE")) / "imagenet" / f"clip_cls_patch_ViT-B-32_{split}_blocks_5_11.pt"
torch.save({"states": torch.cat(all_states), "labels": torch.cat(all_labels)}, path)
print("saved", path, torch.cat(all_states).shape, f"{time.time() - start:.0f}s")
