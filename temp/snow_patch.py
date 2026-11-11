"""Snow test for a saved KAE: find the mode that best separates snowy photos, then damp its eigenvalue. Usage: name:top_k:width:patches"""
import copy, os, sys, torch, open_clip
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
from train import JordanKoopmanAutoencoder
load_dotenv("/home/nsa325/work/koopmann/.env")
torch.set_grad_enabled(False); torch.set_num_threads(16)
SNOW_FOLDERS = {"armytank": 847, "carwheel": 479, "firetruck": 555, "motorscooter": 670, "racingcar": 751, "schoolbus": 779, "trafficlight": 920}
save_dir = Path(os.getenv("CKPT_DIR", Path(os.getenv("WEIGHTS_CACHE")) / "tokae"))
head_dir = Path(os.getenv("WEIGHTS_CACHE")) / "tokae"
data = Path(os.getenv("DATASETS_CACHE"))
clip, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", force_quick_gelu=True, cache_dir=f"{os.getenv('HF_HOME')}/open_clip")
clip.eval()
head = torch.load(head_dir / "zero_shot_head_ViT-B-32.pt")

name, top_k, width, patches = sys.argv[1].split(":")
top_k, width = int(top_k), int(width)
tokens = {"0": "cls", "1": "cls_patch"}.get(patches, patches)
stats = torch.load(save_dir / f"{name}_stats.pt")
val = torch.load(data / "imagenet" / f"clip_{tokens}_ViT-B-32_val_blocks_5_11.pt")
val_x, val_y = (val["states"] - stats["mean"]) / stats["std"], val["labels"]
kae = JordanKoopmanAutoencoder(top_k=top_k, in_features=val_x.shape[-1], latent_features=width, hidden_config=[2048], bias=True, batchnorm=False, nonlinearity="leaky_relu")
kae.load_state_dict(load_file(save_dir / f"{name}.safetensors")); kae.eval()

def classify(states):
    cls = states[..., :768] * stats["std"][:768] + stats["mean"][:768]
    out = clip.visual.ln_post(cls) @ clip.visual.proj
    return (out / out.norm(dim=-1, keepdim=True) @ head).argmax(-1)

def predict(model, start):
    return torch.cat([model.decode(model.koopman_forward(model.encode(b), 1)) for b in start.split(5000)])

def get_activity(start):
    phi = torch.cat([kae.encode(b) for b in start.split(5000)])
    return torch.cat([phi[:, :1].abs(), phi[:, 1:].unflatten(-1, (-1, 2)).norm(dim=-1)], dim=-1)

def dampen(mode, alpha):
    model = copy.deepcopy(kae)
    model.block_real[mode - 1].mul_(alpha); model.block_imag[mode - 1].mul_(alpha)
    return model

paths = sorted((data / "snowy_imagenet").glob("*/*.jpg"))
states = []
keep = lambda _, __, out: states.append(torch.cat({"cls": [out[:, 0]], "cls_patch": [out[:, 0], out[:, 1:].mean(1)], "cls_patch_std": [out[:, 0], out[:, 1:].mean(1), out[:, 1:].std(1)]}[tokens], -1))
hooks = [clip.visual.transformer.resblocks[b].register_forward_hook(keep) for b in [5, 11]]
clip.encode_image(torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths]))
for h in hooks: h.remove()
snowy_x = (torch.stack(states, dim=1) - stats["mean"]) / stats["std"]
snowy_y = torch.tensor([SNOW_FOLDERS[p.parent.name] for p in paths])
normal = torch.isin(val_y, torch.tensor(list(SNOW_FOLDERS.values())))
x, y = torch.cat([val_x[normal], snowy_x]), torch.cat([val_y[normal], snowy_y])
is_snowy = torch.cat([torch.zeros(int(normal.sum()), dtype=torch.bool), torch.ones(len(snowy_x), dtype=torch.bool)])

activity = get_activity(x[:, 0])
aucs = torch.tensor([roc_auc_score(is_snowy.numpy(), activity[:, m].numpy()) if activity[:, m].any() else 0.5 for m in range(activity.shape[1])])
mode = aucs.argmax().item()
val_activity = get_activity(val_x[:, 0])
lam = complex(kae.block_real[mode - 1].item(), kae.block_imag[mode - 1].item())
print(f"{name}: snow mode {mode}, AUC {aucs[mode]:.3f}, |eigenvalue| {abs(lam):.3f}, on for {(val_activity[:, mode] > 0).double().mean():.3f} of val; next best AUCs {aucs.topk(4).values[1:].numpy().round(3)}")
# Pick on one half of the snow test, score on the other
fold = torch.arange(len(y)) % 2
for half in [0, 1]:
    pick, test = fold == half, fold != half
    a = torch.tensor([roc_auc_score(is_snowy[pick].numpy(), activity[pick, m].numpy()) if activity[pick, m].any() else 0.5 for m in range(activity.shape[1])])
    best = a.argmax().item()
    print(f"  half {half}: picks mode {best}, held-out AUC {roc_auc_score(is_snowy[test].numpy(), activity[test, best].numpy()):.3f}")

# Does real CLIP see snow in the mode's top 200 validation images?
tokenizer = open_clip.get_tokenizer("ViT-B-32")
text = clip.encode_text(tokenizer(["a photo in the snow", "a photo"]))
text = text / text.norm(dim=-1, keepdim=True)
direction = (text[0] - text[1]) / (text[0] - text[1]).norm()
cls = val["states"][:, 1, :768]
emb = clip.visual.ln_post(cls) @ clip.visual.proj
snow_score = (emb / emb.norm(dim=-1, keepdim=True)) @ direction
top = val_activity[:, mode].topk(200).indices
names = list(open_clip.IMAGENET_CLASSNAMES)
classes = torch.bincount(val_y[top], minlength=1000).topk(5)
print(f"  top 200 val images: CLIP snow score > 0 for {(snow_score[top] > 0).double().mean():.3f} (all val {(snow_score > 0).double().mean():.3f}); classes {[(names[c], n) for c, n in zip(classes.indices.tolist(), classes.values.tolist())]}")
clip_correct = classify(x[:, 1]) == y
print(f"  CLIP: snowy {clip_correct[is_snowy].double().mean():.3f}, normal {clip_correct[~is_snowy].double().mean():.3f}")
base_val = classify(predict(kae, val_x[:, 0])) == val_y
on = val_activity[:, mode] > 0
for alpha in [1, 0.75, 0.5, 0.25, 0]:
    model = dampen(mode, alpha)
    correct = classify(predict(model, x[:, 0])) == y
    val_correct = base_val.clone(); val_correct[on] = classify(predict(model, val_x[on, 0])) == val_y[on]
    print(f"  alpha {alpha}: snowy {correct[is_snowy].double().mean():.3f}, normal {correct[~is_snowy].double().mean():.3f}; val where on ({int(on.sum())}): fixed {int((~base_val & val_correct).sum())}, broken {int((base_val & ~val_correct).sum())}")
