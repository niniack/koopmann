"""CPU: candidate block-6 summaries of the patch tokens, plus block-12 CLS, for ImageNet validation. Saved for linear probes."""
import os, sys, torch, open_clip
import torch.nn.functional as F
from pathlib import Path
from dotenv import load_dotenv
from datasets import load_dataset
load_dotenv("/home/nsa325/work/koopmann/.env")
torch.set_grad_enabled(False); torch.set_num_threads(24)
N = int(sys.argv[1]); out_path = sys.argv[2]
clip, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", force_quick_gelu=True, cache_dir=f"{os.getenv('HF_HOME')}/open_clip"); clip.eval()
images = load_dataset("ILSVRC/imagenet-1k", split="validation", cache_dir=os.getenv("DATASETS_CACHE"))
blocks = clip.visual.transformer.resblocks
nxt = blocks[6]  # block 7, which reads block 6's output
heads = nxt.attn.num_heads
store = {}
def at6(_, __, out):
    patches = out[:, 1:]
    grid = patches.reshape(len(out), 7, 7, -1).permute(0, 3, 1, 2)
    # Block 7's attention from CLS to every token, averaged over heads, then renormalised over patches
    x = nxt.ln_1(out)
    q, k, _ = F.linear(x, nxt.attn.in_proj_weight, nxt.attn.in_proj_bias).chunk(3, dim=-1)
    q = q[:, 0].unflatten(-1, (heads, -1))
    k = k.unflatten(-1, (heads, -1))
    w = torch.einsum("bhd,bthd->bht", q, k).div(q.shape[-1] ** 0.5).softmax(-1)[:, :, 1:]
    w = (w / w.sum(-1, keepdim=True)).mean(1)
    store.update(cls=out[:, 0], mean=patches.mean(1), std=patches.std(1), max=patches.max(1).values,
                 quad2=F.adaptive_avg_pool2d(grid, 2).flatten(1), grid3=F.adaptive_avg_pool2d(grid, 3).flatten(1),
                 attn=torch.einsum("bt,btd->bd", w, patches))
def at12(_, __, out):
    store["target"] = out[:, 0]
h = [blocks[5].register_forward_hook(at6), blocks[11].register_forward_hook(at12)]
acc = {}
for s in range(0, N, 250):
    idx = list(range(s, min(s + 250, N)))
    clip.encode_image(torch.stack([preprocess(images[i]["image"].convert("RGB")) for i in idx]))
    for k, v in store.items():
        acc.setdefault(k, []).append(v.half())
    if s % 5000 == 0:
        print(s, flush=True)
feats = {k: torch.cat(v) for k, v in acc.items()}
feats["labels"] = torch.tensor(images["label"][:N]) if not hasattr(images["label"], "__array__") else torch.tensor(list(images["label"])[:N])
torch.save(feats, out_path)
print("saved", {k: tuple(v.shape) for k, v in feats.items()})
