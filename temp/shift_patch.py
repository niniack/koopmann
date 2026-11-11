"""Synthetic shifts on 5 val images per class: which modes detect them, and does damping them help? Usage: name:top_k:width:tokens"""
import copy, os, sys, numpy as np, pandas as pd, torch, open_clip
from pathlib import Path
from PIL import Image, ImageOps
from datasets import load_dataset
from dotenv import load_dotenv
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
from train import JordanKoopmanAutoencoder, get_states
load_dotenv("/home/nsa325/work/koopmann/.env")
torch.set_grad_enabled(False); torch.set_num_threads(16)
pd.set_option("display.width", 250)
save_dir = Path(os.getenv("CKPT_DIR", Path(os.getenv("WEIGHTS_CACHE")) / "tokae"))
head = torch.load(Path(os.getenv("WEIGHTS_CACHE")) / "tokae" / "zero_shot_head_ViT-B-32.pt")
clip, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", force_quick_gelu=True, cache_dir=f"{os.getenv('HF_HOME')}/open_clip")
clip.eval()
name, top_k, width, tokens = sys.argv[1].split(":")
top_k, width = int(top_k), int(width)
stats = torch.load(save_dir / f"{name}_stats.pt")
_, val_y = get_states("ViT-B-32", "imagenet", "val", "cpu", [5, 11], tokens=tokens)
kae = JordanKoopmanAutoencoder(top_k=top_k, in_features=len(stats["mean"]), latent_features=width, hidden_config=[2048], bias=True, batchnorm=False, nonlinearity="leaky_relu")
kae.load_state_dict(load_file(save_dir / f"{name}.safetensors")); kae.eval()
images = load_dataset("ILSVRC/imagenet-1k", split="validation", cache_dir=os.getenv("DATASETS_CACHE"))

def classify(states):
    cls = states[..., :768] * stats["std"][:768] + stats["mean"][:768]
    out = clip.visual.ln_post(cls) @ clip.visual.proj
    return (out / out.norm(dim=-1, keepdim=True) @ head).argmax(-1)
predict = lambda model, s: model.decode(model.koopman_forward(model.encode(s), 1))
def get_activity(start):
    phi = kae.encode(start)
    return torch.cat([phi[:, :1].abs(), phi[:, 1:].unflatten(-1, (-1, 2)).norm(dim=-1)], dim=-1)
def dampen(mode, alpha):
    model = copy.deepcopy(kae)
    model.block_real[mode - 1].mul_(alpha); model.block_imag[mode - 1].mul_(alpha)
    return model

indices = torch.cat([(val_y == c).nonzero()[:5, 0] for c in range(1000)])
labels = val_y[indices]
def grayscale(image): return image.convert("L").convert("RGB")
def sepia(image): return ImageOps.colorize(image.convert("L"), black="#1e1206", white="#f2dfbd")
def motion_blur(image):
    pixels = np.asarray(image).astype(np.float32)
    length = max(3, image.width // 25)
    blurred = np.mean([np.roll(pixels, shift, axis=1) for shift in range(-(length // 2), length // 2 + 1)], axis=0)
    return Image.fromarray(blurred.astype(np.uint8))
SHIFTS = {"original": lambda image: image, "grayscale": grayscale, "sepia": sepia, "motion blur": motion_blur}

def keep(_, __, o):
    cls, patches = o[:, 0], o[:, 1:]
    parts = {"cls": [cls], "cls_patch": [cls, patches.mean(1)], "cls_patch_std": [cls, patches.mean(1), patches.std(1)]}[tokens]
    states.append(torch.cat(parts, -1))
states = []
hooks = [clip.visual.transformer.resblocks[b].register_forward_hook(keep) for b in [5, 11]]
data = {}
for shift_name, shift in SHIFTS.items():
    out = []
    for batch in indices.split(250):
        clip.encode_image(torch.stack([preprocess(shift(images[i]["image"].convert("RGB"))) for i in batch.tolist()]))
        out.append(torch.stack(states, dim=1)); states.clear()
    data[shift_name] = (torch.cat(out) - stats["mean"]) / stats["std"]
    print("extracted", shift_name, flush=True)
for h in hooks: h.remove()

original = data["original"]
base = get_activity(original[:, 0])
rows = []
for shift_name in ["grayscale", "sepia", "motion blur"]:
    shifted = data[shift_name]
    act = get_activity(shifted[:, 0])
    both = torch.cat([base, act]); is_shifted = torch.cat([torch.zeros(len(act)), torch.ones(len(act))]).numpy()
    aucs = torch.tensor([roc_auc_score(is_shifted, both[:, m].numpy()) if both[:, m].any() else 0.5 for m in range(both.shape[1])])
    top = aucs.argsort(descending=True)[:3].tolist()
    mode = top[0]
    row = {"shift": shift_name, "CLIP, original": (classify(original[:, 1]) == labels).double().mean().item(),
           "CLIP, shifted": (classify(shifted[:, 1]) == labels).double().mean().item(),
           "KAE, original": (classify(predict(kae, original[:, 0])) == labels).double().mean().item(),
           "KAE, shifted": (classify(predict(kae, shifted[:, 0])) == labels).double().mean().item(),
           "KAE agrees with CLIP, shifted": (classify(predict(kae, shifted[:, 0])) == classify(shifted[:, 1])).double().mean().item(),
           "top modes (AUC)": ", ".join(f"{m} ({aucs[m]:.2f})" for m in top),
           "on, shifted": (act[:, mode] > 0).double().mean().item(), "on, original": (base[:, mode] > 0).double().mean().item(),
           "mode on/off unchanged": ((act > 0) == (base > 0)).double().mean().item()}
    for alpha in [0.5, 0.0]:
        edited = dampen(mode, alpha)
        row[f"KAE shifted, alpha {alpha}"] = (classify(predict(edited, shifted[:, 0])) == labels).double().mean().item()
        row[f"KAE original, alpha {alpha}"] = (classify(predict(edited, original[:, 0])) == labels).double().mean().item()
    rows.append(row)
print(pd.DataFrame(rows).set_index("shift").T.to_string())
