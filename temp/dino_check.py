"""DINOv3 ViT-B/16 vs CLIP ViT-B/32: how much of block 12 can a linear map recover from block 6? Read with a linear probe on the true block 12 CLS."""
import os, sys, time, torch, torch.nn.functional as F
from datasets import load_dataset
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoModel
load_dotenv("/home/nsa325/work/koopmann/.env")
torch.set_grad_enabled(False); torch.set_num_threads(24); torch.manual_seed(0)
SP = os.path.dirname(os.path.abspath(__file__))

def fit_linear(keys, values, reg=100.0):
    keys = F.pad(keys, (0, 1), value=1).double()
    gram = keys.T @ keys
    return torch.linalg.solve(gram + reg * torch.eye(len(gram), dtype=gram.dtype), keys.T @ values.double()).float()
apply_linear = lambda op, x: F.pad(x, (0, 1), value=1) @ op

def report(name, feats, labels, val, vlabels, head=None):
    """feats: dict of block-6 inputs plus 'end' (block 12 CLS). head: the model's own classifier on a raw block 12 CLS."""
    mu, sd = {}, {}
    for k in feats:
        mu[k], sd[k] = feats[k].mean(0), feats[k].std(0) + 1e-6
    z = lambda d, k: (d[k] - mu[k]) / sd[k]
    probe = fit_linear(z(feats, "end"), F.one_hot(labels, 1000).float())
    read = lambda x: apply_linear(probe, x).argmax(-1)
    true = read(z(val, "end"))
    print(f"\n{name}: probe on true block 12 CLS {(true == vlabels).float().mean():.3f}", flush=True)
    if head is not None:
        head_true = head(val["end"]).argmax(-1)
        print(f"  official head on true block 12 CLS {(head_true == vlabels).float().mean():.3f}", flush=True)
    for inputs in INPUTS[name]:
        x, vx = torch.cat([z(feats, k) for k in inputs], -1), torch.cat([z(val, k) for k in inputs], -1)
        own = fit_linear(x, F.one_hot(labels, 1000).float())
        pred = apply_linear(fit_linear(x, z(feats, "end")), vx)
        target = z(val, "end")
        fvu = ((pred - target).pow(2).sum() / (target - target.mean(0)).pow(2).sum()).item()
        answers = read(pred)
        extra = ""
        if head is not None:
            head_answers = head(pred * sd["end"] + mu["end"]).argmax(-1)
            extra = f" | official head {(head_answers == vlabels).float().mean():.3f}, agrees {(head_answers == head_true).float().mean():.3f}"
        print(f"  {' + '.join(inputs):28s} probe on block 6 {(apply_linear(own, vx).argmax(-1) == vlabels).float().mean():.3f} | "
              f"linear map to block 12: fvu {fvu:.3f}, block-12 probe {(answers == vlabels).float().mean():.3f}, agrees with probe on true {(answers == true).float().mean():.3f}{extra}", flush=True)

INPUTS = {"CLIP ViT-B/32": [["cls"], ["cls", "mean"], ["cls", "mean", "std"]],
          "DINOv3 ViT-B/16": [["cls"], ["cls", "mean"], ["cls", "mean", "std"], ["cls", "registers", "mean", "std"]]}

# CLIP, from the pooling check's saved subset
# NOTE: CLIP's numbers were printed by an earlier run of this script

# DINOv3
model = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m").eval()
preprocess = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

class Images(Dataset):
    def __init__(self, split, n):
        self.data = load_dataset("ILSVRC/imagenet-1k", split=split, cache_dir=os.getenv("DATASETS_CACHE"))
        self.idx = torch.randperm(len(self.data))[:n].tolist()
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        s = self.data[self.idx[i]]
        return preprocess(s["image"].convert("RGB")), s["label"]

def run(split, n):
    rows, labels, start = {}, [], time.time()
    for i, (images, y) in enumerate(DataLoader(Images(split, n), batch_size=200, num_workers=8)):
        h = model(pixel_values=images, output_hidden_states=True).hidden_states
        six, end = h[6], h[12]
        patches = six[:, 5:]  # NOTE: CLS, then 4 registers, then patches
        out = {"cls": six[:, 0], "registers": six[:, 1:5].flatten(1), "mean": patches.mean(1), "std": patches.std(1), "end": end[:, 0]}
        for k, v in out.items():
            rows.setdefault(k, []).append(v.half())
        labels.append(y)
        if i % 50 == 0: print(split, i * 200, f"{time.time() - start:.0f}s", flush=True)
    return {k: torch.cat(v).float() for k, v in rows.items()}, torch.cat(labels)

train, ty = run("train", 100_000)
val, vy = run("validation", 10_000)
torch.save({"train": train, "ty": ty, "val": val, "vy": vy}, f"{SP}/dino_check.pt")
# The official linear head from github.com/m2b3/dinov3-in1k-probes: final layer norm, then 768 -> 1000, trained at 512 px
from glob import glob
from safetensors.torch import load_file
probe = load_file(glob(os.getenv("HF_HOME") + "/hub/models--yberreby--dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe/snapshots/*/model.safetensors")[0])
head = lambda cls: model.norm(cls) @ probe["weight"].T + probe["bias"]
report("DINOv3 ViT-B/16", train, ty, val, vy, head)
