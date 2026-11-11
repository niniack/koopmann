"""
The snow test for the "cue and conclusion" reading of a mode, on a CLIP KAE.

K's prediction for a mode is λ·z6 = Re(λ)·z6 + i·Im(λ)·z6. The first term stays on the direction the mode had at block 6
(the "kept part"), the second is at a right angle to it (the "built part"). With `real_start`, a mode starts on its first
coordinate only, so the kept part is the pattern that was already there at block 6 (the cue: snow) and the built part is a
pattern that only exists at block 12 (the conclusion: snowplow). If that reading holds, setting a snow mode's angle to 0
(dropping the built part) should take back the wrong conclusion while the output still says "snow", and removing the mode
should take back both. On a run trained without the rule (the control) the two parts have no such roles.

Test images: 129 snowy photos of 7 vehicle classes, and ImageNet's validation images of the same classes.
"Says snow": cosine between CLIP's final image embedding and the text embedding of the word "snow".

Usage: cue_conclusion.py <CLIP run name> <real_start: 0 or 1>
"""

import os
import sys
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
from dotenv import load_dotenv  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

load_dotenv("/home/nsa325/work/koopmann/.env")
import train  # noqa: E402

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
torch.set_grad_enabled(False)
SP = Path(__file__).parent
ENCODER = "ViT-B-32"
FOLDERS = {"armytank": 847, "carwheel": 479, "firetruck": 555, "motorscooter": 670, "racingcar": 751, "schoolbus": 779, "trafficlight": 920}
name, real_start = sys.argv[1], sys.argv[2] == "1"
root, save_dir = Path(os.getenv("DATASETS_CACHE")), os.getenv("WEIGHTS_CACHE") + "/tokae"
stats = torch.load(f"{save_dir}/{name}_stats.pt")
mean, std = stats["mean"], stats["std"]
val_raw, val_y = train.get_states(ENCODER, "imagenet", "val", "cpu", [5, 11], tokens="cls_patch_std", extract=False)
val_raw = val_raw.float()

############# SNOWY STATES #############

snowy_path = SP / "snowy_clip_states.pt"
if not snowy_path.exists():
    import open_clip
    from datasets import load_dataset
    from PIL import Image

    clip, _, preprocess = open_clip.create_model_and_transforms(ENCODER, pretrained="openai", force_quick_gelu=True, cache_dir=f"{os.getenv('HF_HOME')}/open_clip")
    clip.eval()

    def get_raw_states(pixels):
        """Unstandardized states after blocks 6 and 12, built the way the training cache was."""

        states = []

        def keep(_, __, output):
            output = output[0] if isinstance(output, tuple) else output
            patches = output[:, 1:]
            states.append(torch.cat([output[:, 0], patches.mean(dim=1), patches.std(dim=1)], dim=-1))

        hooks = [clip.visual.transformer.resblocks[block].register_forward_hook(keep) for block in (5, 11)]
        for batch in pixels.split(32):
            clip.encode_image(batch)
        for hook in hooks:
            hook.remove()
        return torch.cat([torch.stack(states[i : i + 2], dim=1) for i in range(0, len(states), 2)])

    images = load_dataset("ILSVRC/imagenet-1k", split="validation", cache_dir=os.getenv("DATASETS_CACHE"))
    fresh = get_raw_states(torch.stack([preprocess(train.to_rgb(images[i]["image"])) for i in range(5)]))
    gap = ((fresh - val_raw[:5]).norm(dim=-1) / val_raw[:5].norm(dim=-1)).max().item()
    print(f"largest relative gap between fresh and cached states on 5 val images: {gap:.2e}")
    assert gap < 1e-2, "snowy states would not be comparable to the cached ones"
    paths = sorted((root / "snowy_imagenet").glob("*/*.jpg"))
    snowy_raw = get_raw_states(torch.stack([preprocess(train.to_rgb(Image.open(path))) for path in paths]))
    torch.save({"states": snowy_raw, "labels": torch.tensor([FOLDERS[path.parent.name] for path in paths]), "paths": [str(path) for path in paths]}, snowy_path)
snowy = torch.load(snowy_path)

############# MODEL #############

kae = train.JordanKoopmanAutoencoder(256, shared_modes=True, real_start=real_start, in_features=2304, latent_features=2049, hidden_config=stats["hidden_config"], bias=True, batchnorm=False, nonlinearity="leaky_relu")
kae.load_state_dict(load_file(f"{save_dir}/{name}.safetensors"))
kae.eval()
# NOTE: Building the zero-shot classifier takes over 30 minutes on a busy CPU, so it is read from a cache
head = train.ClipHead(ENCODER, mean, std, "cpu", zero_shot=False)
head.classes = torch.load(SP / "zero_shot_head.pt").float().cpu()
vocabulary = torch.load(SP / "clip_vocab.pt", weights_only=False)
snow_text = torch.nn.functional.normalize(vocabulary["embeddings"][vocabulary["words"].index("snow")].float(), dim=-1)
to_complex = lambda latent: torch.cat([latent[..., :1].to(torch.cfloat), torch.complex(latent[..., 1::2], latent[..., 2::2])], dim=-1)


def to_latent(z):
    blocks = torch.stack([z[..., 1:].real, z[..., 1:].imag], dim=-1).flatten(-2)
    return torch.cat([z[..., :1].real, blocks], dim=-1)


lam = torch.cat([kae.real_eigenvalue.to(torch.cfloat), torch.complex(kae.block_real, kae.block_imag)])
standardize = lambda raw: (raw - mean) / std
normal = torch.isin(val_y, torch.tensor(list(FOLDERS.values())))
x = torch.cat([standardize(val_raw[normal]), standardize(snowy["states"].float())])
y = torch.cat([val_y[normal], snowy["labels"]])
is_snowy = torch.cat([torch.zeros(int(normal.sum()), dtype=torch.bool), torch.ones(len(snowy["labels"]), dtype=torch.bool)])
z6 = to_complex(train.encode_pairs(kae, x))[:, 0]
# What an edit leaves of a mode's predicted value λ·z6
EDITS = {
    "K as trained": lambda l, z: l * z,
    "mode removed": lambda l, z: 0 * z,
    "built part dropped (angle to 0)": lambda l, z: l.real * z,
    "kept part dropped": lambda l, z: 1j * l.imag * z,
    "no turn, same size": lambda l, z: l.abs() * z,
    "half size": lambda l, z: 0.5 * l * z,
}


def predict(z, modes, edit):
    latent = lam * z
    for m in modes:
        latent[:, m] = EDITS[edit](lam[m], z[:, m])
    return kae.decode(to_latent(latent))


says_snow = lambda states: head.embed(states) @ snow_text
print(f"== {name} (real_start {real_start}): {int(is_snowy.sum())} snowy and {int((~is_snowy).sum())} normal images of the same 7 classes")
true_right = head.classify(x[:, 1]) == y
base = predict(z6, [], "K as trained")
base_right = head.classify(base) == y
print(f"real CLIP: snowy {true_right[is_snowy].float().mean():.3f}, normal {true_right[~is_snowy].float().mean():.3f}; says snow: snowy {says_snow(x[:, 1])[is_snowy].mean():.4f}, normal {says_snow(x[:, 1])[~is_snowy].mean():.4f}")
print(f"KAE:       snowy {base_right[is_snowy].float().mean():.3f}, normal {base_right[~is_snowy].float().mean():.3f}; says snow: snowy {says_snow(base)[is_snowy].mean():.4f}, normal {says_snow(base)[~is_snowy].mean():.4f}")
answers = head.classify(base)[is_snowy]
names = __import__("open_clip").IMAGENET_CLASSNAMES
print("KAE's most common wrong answers on snowy images:", [(names[c], n) for c, n in zip(*[t.tolist() for t in answers[answers != y[is_snowy]].unique(return_counts=True)]) if n >= 4])

activity = z6.abs()
aucs = torch.tensor([roc_auc_score(is_snowy.numpy(), activity[:, m].numpy()) if activity[:, m].any() else 0.5 for m in range(activity.shape[1])])
order = aucs.argsort(descending=True)
print("\nmodes whose block-6 size best separates snowy from normal images: mode | AUC | on for snowy | on for normal | size of λ | angle of λ")
for m in order[:8].tolist():
    print(f"  {m:5d} | {aucs[m]:.3f} | {(activity[is_snowy, m] > 0).float().mean():.2f} | {(activity[~is_snowy, m] > 0).float().mean():.2f} | {lam[m].abs():.2f} | {torch.rad2deg(torch.angle(lam[m])):7.1f}°")

print("\nedit of the top snow modes: snowy accuracy | fixed | broken | normal accuracy | says snow on snowy images (KAE before the edit, then after)")
for count in [1, 2, 4, 8]:
    modes = order[:count].tolist()
    print(f"  top {count}: modes {modes}")
    for edit in list(EDITS)[1:]:
        edited = predict(z6, modes, edit)
        right = head.classify(edited) == y
        fixed, broken = (right & ~base_right)[is_snowy].sum().item(), (~right & base_right)[is_snowy].sum().item()
        print(f"    {edit:32s} {right[is_snowy].float().mean():.3f} | {fixed:3d} | {broken:3d} | {right[~is_snowy].float().mean():.3f} | {says_snow(base)[is_snowy].mean():.4f} -> {says_snow(edited)[is_snowy].mean():.4f}")

print("\nheld out: the 4 modes are picked on one half of the images and scored on the other")
fold = torch.arange(len(y)) % 2
for half in [0, 1]:
    pick, test = fold == half, fold != half
    half_aucs = torch.tensor([roc_auc_score(is_snowy[pick].numpy(), activity[pick, m].numpy()) if activity[pick, m].any() else 0.5 for m in range(activity.shape[1])])
    modes = half_aucs.argsort(descending=True)[:4].tolist()
    line = [f"half {half}, modes {modes}: before {base_right[test & is_snowy].float().mean():.3f}"]
    for edit in ["mode removed", "built part dropped (angle to 0)", "kept part dropped"]:
        right = head.classify(predict(z6, modes, edit)) == y
        line.append(f"{edit} {right[test & is_snowy].float().mean():.3f}")
    print("  " + "; ".join(line))

print("\ncost on ImageNet validation, top 4 modes: images that keep one | answers fixed | broken")
modes = order[:4].tolist()
val_x = standardize(val_raw)
fixed, broken, keeps = {edit: 0 for edit in EDITS}, {edit: 0 for edit in EDITS}, 0
for first in range(0, len(val_x), 5000):
    rows = slice(first, first + 5000)
    z = to_complex(train.encode_pairs(kae, val_x[rows]))[:, 0]
    on = (z[:, modes] != 0).any(dim=1)
    keeps += on.sum().item()
    before = head.classify(predict(z[on], [], "K as trained")) == val_y[rows][on]
    for edit in list(EDITS)[1:4]:
        after = head.classify(predict(z[on], modes, edit)) == val_y[rows][on]
        fixed[edit] += (after & ~before).sum().item()
        broken[edit] += (~after & before).sum().item()
for edit in list(EDITS)[1:4]:
    print(f"  {edit:32s} {keeps} | {fixed[edit]} | {broken[edit]}")
print("done")
