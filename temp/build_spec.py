"""
Builds the list of block-12 states to paint (see invert_batch.py). Everything is fixed here, before any picture is seen.

Images: for each mode with a known meaning, the validation image where it is largest at block 6 (hidden images left out),
plus the runner-up for modes 667 and 211, whose first images show little (bare woods, a nearly black frame).
"""
import math, os
from pathlib import Path
import torch
from datasets import load_dataset
from torchvision import transforms
from torchvision.utils import save_image
import inversion
from inversion import train, to_complex, to_latent, turn
from alter_states import to_gray, to_sepia

torch.set_num_threads(8)
torch.set_grad_enabled(False)
# image: (its mode, what the mode fires on)
IMAGES = {
    42917: (667, "winter woods, snow"),
    39437: (667, "winter woods, snow"),
    12995: (644, "snow, white water"),
    15059: (247, "fruit piles"),
    26388: (37, "open water"),
    32814: (71, "birds against sky"),
    46610: (535, "knitted"),
    33079: (211, "fire or glow in the dark"),
    27631: (211, "fire or glow in the dark"),
}
COLOUR_IMAGES = 5
ALWAYS_ON = {861, 338, 144, 560, 117, 988, 871, 746, 652, 533, 331, 77, 68, 653, 877, 465}

kae, stats = inversion.load_kae()
reader = inversion.Reader(stats["mean"], stats["std"])
lam = torch.cat([kae.real_eigenvalue.to(torch.cfloat), torch.complex(kae.block_real, kae.block_imag)])
factors = torch.load("alter_factors.pt")["factors"]
images = load_dataset("ILSVRC/imagenet-1k", split="validation", cache_dir=os.getenv("DATASETS_CACHE"))
to_picture = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC), transforms.CenterCrop(224), train.to_rgb, transforms.ToTensor()])
pictures = torch.stack([to_picture(images[i]["image"]) for i in IMAGES])
colourfulness = inversion.get_colourfulness(pictures)
most_coloured = set(torch.tensor(list(IMAGES))[colourfulness.argsort(descending=True)[:COLOUR_IMAGES]].tolist())
print("colourfulness:", dict(zip(IMAGES, colourfulness.round(decimals=3).tolist())), "colour edits on", most_coloured)


def standardize(raw):
    return (raw - stats["mean"]) / stats["std"]


def predict(z_start):
    return kae.decode(to_latent(lam * z_start))


states = {"original": standardize(reader.raw_states(pictures)), "gray": standardize(reader.raw_states(to_gray(pictures))), "sepia": standardize(reader.raw_states(to_sepia(pictures)))}
z = {key: to_complex(train.encode_pairs(kae, value)) for key, value in states.items()}
spec, notes = [], {}
Path("inv").mkdir(exist_ok=True)
for row, (index, (mode, meaning)) in enumerate(IMAGES.items()):
    save_image(pictures[row], f"inv/photo_{index}.png")
    z6 = z["original"][row, 0]
    assert z6[mode] != 0, "the image does not keep its mode"

    # Control: the kept mode, not always on, whose predicted size at block 12 is closest to the tested mode's
    sizes = (lam * z6).abs()
    candidates = [m for m in (z6 != 0).nonzero()[:, 0].tolist() if m != mode and m not in ALWAYS_ON]
    control = min(candidates, key=lambda m: abs(sizes[m] - sizes[mode]).item())
    notes[index] = {"mode": mode, "meaning": meaning, "control": control, "coloured": index in most_coloured}

    def add(kind, target, start=None):
        entry = {"name": f"{index}_{kind}", "index": index, "kind": kind, "target": target.clone(), "seed": index}
        if start is not None:
            entry["start"] = f"{index}_{start}"
        spec.append(entry)

    def edited(changes):
        values = z6.clone()
        for m, factor in changes.items():
            values[m] = values[m] * factor
        return predict(values)

    add("real", states["original"][row, 1])
    add("pred", predict(z6))
    add("recon", kae.decode(to_latent(z["original"][row, 1])))
    add("same", predict(z6), start="pred")
    add("removed", edited({mode: 0}), start="pred")
    add("x2", edited({mode: 2}), start="pred")
    add("+90", edited({mode: turn(90)}), start="pred")
    add("-90", edited({mode: turn(-90)}), start="pred")
    add("control", edited({control: 0}), start="pred")
    if index in most_coloured:
        for key in ["gray", "sepia"]:
            add(f"real {key}", states[key][row, 1], start="real")
            add(f"knob {key}", predict(z6 * factors[key]), start="pred")
        add("pred of gray copy", predict(z["gray"][row, 0]), start="pred")

torch.save(spec, "inv/spec.pt")
torch.save(notes, "inv/notes.pt")
print(len(spec), "targets:", sum("start" not in e for e in spec), "from scratch,", sum("start" in e for e in spec), "continued")
print(notes)
