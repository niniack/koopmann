"""
TEMP: the fixed test of a run of cls_train.py, on each arm's best weights and on every saved step.

1. On ImageNet validation, ImageNet-V2, ImageNet-Sketch, ImageNet-R, ImageNet-A and the 129 snowy photos: accuracy with
   CLIP's zero-shot head, and how often the answer is real CLIP's. For ImageNet-R and ImageNet-A the answers are limited
   to the set's own 200 classes, as is usual.
2. For the KAE arms, on ImageNet validation: the eigenvalue ablation of angle_tests.py, the number of modes in use and
   of modes that are on for 99% of images, the median size and angle of λ, and the control's share of z12.
The rows go to a CSV in this folder, which is written anew on every call.

Usage: baseline_eval.py --files="base_*"
"""

import os
from pathlib import Path

import fire
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

import angle_tests as tests
from angle_tests import base, device, train

ENCODER = "ViT-B-32"
# The test sets next to ImageNet validation, and whether answers are limited to the set's own classes
SETS = {"imagenetv2": False, "imagenet_sketch": False, "imagenet-r": True, "imagenet-a": True, "snow": False}


def extract_snow(path):
    """The 129 snowy photos through real CLIP, saved like extract_shift.py's sets."""

    teacher = base.Teacher(ENCODER, device)
    preprocess = transforms.Compose([*teacher.resize, teacher.to_rgb, transforms.PILToTensor()])
    photos = sorted((Path(os.getenv("DATASETS_CACHE")) / "snowy_imagenet").glob("*/*.jpg"))
    states, patches = teacher(torch.stack([preprocess(Image.open(photo)) for photo in photos]).to(device))

    # NOTE: The labels are those of the states extracted earlier, so the photos have to come in the same order
    earlier = torch.load(base.HERE / "snowy_clip_states.pt")
    gap = (states.cpu() - earlier["states"]).abs().max().item()
    assert gap < 0.05, f"the snowy states differ from the earlier ones by up to {gap}"
    torch.save({"states": states.cpu(), "patches": patches.half().cpu(), "labels": earlier["labels"]}, path)


def get_sets(mean, std):
    """Each test set as (standardized CLS states [n, 2, width], block-6 patch tokens, labels, the classes an answer can be)."""

    sets = {"imagenet": (*base.get_val(None, ENCODER, device), False)}
    for name, limited in SETS.items():
        path = base.HERE / "domain_cache" / f"{name}.pt"
        if name == "snow" and not path.exists():
            extract_snow(path)
        data = torch.load(path)
        sets[name] = (data["states"].to(device), data["patches"].to(device), data["labels"].to(device), limited)

    for name, (states, patches, labels, limited) in sets.items():
        classes = labels.unique() if limited else torch.arange(1_000, device=device)
        # NOTE: Labels count within `classes`, as the head's answers do once it is cut to these classes
        sets[name] = ((states[..., : len(mean)] - mean) / std, patches, torch.searchsorted(classes, labels), classes)

    return sets


def get_spectrum(kae, z6):
    """Modes in use (by 50+ images) and always on, the usual size and angle of λ, and the control's share of z12."""

    usage = (tests.to_complex(z6)[:, 1:] != 0).float().mean(dim=0)
    used = usage * len(z6) >= 50
    eigenvalues = torch.complex(kae.block_real, kae.block_imag)[used]

    return {
        "modes_used": used.sum().item(),
        "modes_always_on": (usage >= 0.99).sum().item(),
        "size_median": eigenvalues.abs().median().item(),
        "angle_median": eigenvalues.angle().abs().rad2deg().median().item(),
        "control_share": kae.control_share.item(),
    }


def main(files: str = "base_*", out: str = "baseline_eval.csv"):
    """Tests every weights file whose name matches `files`: `<run>_<arm>` (best) and `<run>_<arm>_step<N>`."""

    folder = Path(os.getenv("WEIGHTS_CACHE")) / "tokae"
    names = sorted(path.stem for path in folder.glob(f"{files}.safetensors"))
    stats = torch.load(folder / f"{names[0].partition('_step')[0]}_stats.pt")
    mean, std = stats["mean"].to(device), stats["std"].to(device)
    head = train.ClipHead(ENCODER, mean, std, device, zero_shot=True)
    all_classes = head.classes
    sets = get_sets(mean, std)

    rows = []
    for name in names:
        run, _, step = name.partition("_step")
        model, stats = tests.load_kae(run, name)
        for data, (states, patches, labels, classes) in sets.items():
            head.classes = all_classes[:, classes]
            clip = head.classify(states[:, 1])
            model.set_control(patches)
            z6 = torch.cat([train.encode_pairs(model, part)[:, 0] for part in states.split(5_000)])
            answers = {"as trained": tests.classify(model, head, model.koopman_forward(z6, 1))}
            print(f"{name} | {data}: accuracy {tests.rate(answers['as trained'], labels):.2f} (real CLIP {tests.rate(clip, labels):.2f}), same answer as real CLIP {tests.rate(answers['as trained'], clip):.2f}", flush=True)

            spectrum = {}
            if data == "imagenet" and stats["arm"] != "mlp":
                spectrum = get_spectrum(model, z6)
                print("  " + ", ".join(f"{key} {value:.2f}" for key, value in spectrum.items()))
                answers = tests.ablate(model, head, z6, labels, clip)

            for variant, given in answers.items():
                scores = {"accuracy": tests.rate(given, labels), "same_as_clip": tests.rate(given, clip), "same_as_trained": tests.rate(given, answers["as trained"])}
                rows.append({"model": run, "step": step or "best", "data": data, "variant": variant, **scores, "clip_accuracy": tests.rate(clip, labels), **spectrum})

        pd.DataFrame(rows).to_csv(base.HERE / out, index=False)


if __name__ == "__main__":
    fire.Fire(main)
