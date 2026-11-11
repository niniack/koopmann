import os
from pathlib import Path
import torch
import inversion
kae, stats = inversion.load_kae()
val = torch.load(Path(os.getenv("DATASETS_CACHE")) / "imagenet" / "clip_cls_patch_std_dinov3-vitb16_val_blocks_5_11.pt")
x = (val["states"][:2, 1].float() - stats["mean"]) / stats["std"]
spec = []
for i in range(2):
    spec.append({"name": f"{i}_real", "index": i, "kind": "real", "target": x[i], "seed": i})
    spec.append({"name": f"{i}_same", "index": i, "kind": "same", "target": x[i], "seed": i, "start": f"{i}_real"})
Path("inv_test").mkdir(exist_ok=True)
torch.save(spec, "inv_test/spec.pt")
