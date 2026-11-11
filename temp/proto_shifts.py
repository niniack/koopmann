import json, os, sys
import numpy as np, pandas as pd, torch
from PIL import ImageOps
from sklearn.metrics import roc_auc_score
os.chdir("/home/nsa325/work/koopmann/second_analysis")
nb = json.load(open("jordan_topk_modes.ipynb"))
cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
for i in [0, 1, 2]:
    exec(cells[i], globals())
torch.set_num_threads(48)
pd.set_option("display.width", 250)
SP = sys.argv[1]

# 5 validation images per class
indices = torch.cat([(val_y == c).nonzero()[:5, 0] for c in range(1000)])
labels = val_y[indices]


def grayscale(image):
    return image.convert("L").convert("RGB")


def sepia(image):
    return ImageOps.colorize(image.convert("L"), black="#1e1206", white="#f2dfbd")


def motion_blur(image):
    """Average of horizontal shifts, about 4% of the width."""
    pixels = np.asarray(image).astype(np.float32)
    length = max(3, image.width // 25)
    blurred = np.mean([np.roll(pixels, shift, axis=1) for shift in range(-(length // 2), length // 2 + 1)], axis=0)
    return Image.fromarray(blurred.astype(np.uint8))


SHIFTS = {"original": lambda image: image, "grayscale": grayscale, "sepia": sepia, "motion blur": motion_blur}


def get_states(shift):
    states, out = [], []
    hooks = [clip.visual.transformer.resblocks[b].register_forward_hook(lambda _, __, o: states.append(o[:, 0])) for b in stats["blocks"]]
    for batch in indices.split(250):
        clip.encode_image(torch.stack([preprocess(shift(images[i]["image"].convert("RGB"))) for i in batch.tolist()]))
        out.append(torch.stack(states, dim=1)); states.clear()
    for h in hooks: h.remove()
    return (torch.cat(out) - stats["mean"]) / stats["std"]


data = {}
for name, shift in SHIFTS.items():
    data[name] = get_states(shift); print("extracted", name, flush=True)
torch.save({"data": data, "indices": indices}, f"{SP}/shift_states.pt")

original = data["original"]
base_activity = get_activity(original[:, 0])
rows = []
for name in ["grayscale", "sepia", "motion blur"]:
    shifted = data[name]
    act = get_activity(shifted[:, 0])
    # how well each mode's activity tells shifted from original, on the same images
    both = torch.cat([base_activity, act]); is_shifted = torch.cat([torch.zeros(len(act)), torch.ones(len(act))]).numpy()
    aucs = torch.tensor([roc_auc_score(is_shifted, both[:, m].numpy()) if both[:, m].any() else 0.5 for m in range(both.shape[1])])
    top = aucs.argsort(descending=True)[:3].tolist()
    mode = top[0]
    row = {"shift": name, "CLIP, original": (classify(original[:, 1]) == labels).double().mean().item(),
           "CLIP, shifted": (classify(shifted[:, 1]) == labels).double().mean().item(),
           "KAE, original": (classify(predict(kae, original[:, 0])) == labels).double().mean().item(),
           "KAE, shifted": (classify(predict(kae, shifted[:, 0])) == labels).double().mean().item(),
           "top modes (AUC)": ", ".join(f"{m} ({aucs[m]:.2f})" for m in top),
           "on, shifted": (act[:, mode] > 0).double().mean().item(), "on, original": (base_activity[:, mode] > 0).double().mean().item()}
    for alpha in [0.5, 0.0]:
        edited = dampen(mode, alpha)
        row[f"KAE shifted, alpha {alpha}"] = (classify(predict(edited, shifted[:, 0])) == labels).double().mean().item()
        row[f"KAE original, alpha {alpha}"] = (classify(predict(edited, original[:, 0])) == labels).double().mean().item()
    rows.append(row)
    print(row, flush=True)
print(pd.DataFrame(rows).set_index("shift").T.to_string())
