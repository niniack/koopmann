"""
TEMP: the fixed test of a run of cls_train.py, on each arm's best weights and on every saved step.

1. On ImageNet validation, ImageNet-V2, ImageNet-Sketch, ImageNet-R, ImageNet-A and the 129 snowy photos: accuracy with
   CLIP's zero-shot head, and how often the answer is real CLIP's. For ImageNet-R and ImageNet-A the answers are limited
   to the set's own 200 classes, as is usual.
2. For the KAE arms, on ImageNet validation: the eigenvalue ablation of angle_tests.py, the number of modes in use and
   of modes that are on for 99% of images, the median size and angle of λ, and the control's share of z12.
3. On every set: the answers when an image gets the CLS token of another image, and when it gets the control's average
   over the set in place of its own control (for a reader of lift.py: the patch tokens of another image). They show
   what the model takes from each of its two inputs.
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
import lift
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

    # NOTE: The labels are those of the states extracted earlier, so the photos have to come in the same order.
    # The same photo differs by up to 2% between the two machines, and two different photos by far more.
    earlier = torch.load(base.HERE / "snowy_clip_states.pt")
    gap = ((states.cpu() - earlier["states"]).norm(dim=-1) / earlier["states"].norm(dim=-1)).max().item()
    print(f"largest relative gap between the snowy states and the earlier ones: {gap:.1e}")
    assert gap < 0.1, "the snowy photos are not the earlier ones, or not in their order"
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
        "control_share": getattr(kae, "control_share", torch.zeros(())).item(),
    }


def encode_start(model, states, patches):
    """The block-6 code of every image. NOTE: A reader encodes the CLS token together with the patch tokens, so it gets the whole set at once."""

    model.set_control(patches)
    if isinstance(model, lift.LiftKae):
        return model.encode(states[:, 0])

    return torch.cat([train.encode_pairs(model, part)[:, 0] for part in states.split(5_000)])


def swap_inputs(model, head, states, patches, z6):
    """
    Answers with one input made uninformative: the CLS token of the image half the set away, and the set's average
    control (for a reader: the patch tokens of that other image).
    """

    half = len(z6) // 2
    if isinstance(model, lift.LiftKae):
        codes = {"CLS of another image": encode_start(model, states.roll(half, dims=0), patches), "patch tokens of another image": encode_start(model, states, patches.roll(half, dims=0))}
        return {row: tests.classify(model, head, model.koopman_forward(code, 1)) for row, code in codes.items()}

    signal = model.signal
    answers = {"CLS of another image": tests.classify(model, head, model.koopman_forward(z6.roll(half, dims=0), 1))}
    if signal is not None:
        model.signal = signal.mean(dim=0, keepdim=True).expand_as(signal)
        answers["control averaged"] = tests.classify(model, head, model.koopman_forward(z6, 1))
        model.signal = signal

    return answers


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
            z6 = encode_start(model, states, patches)
            answers = {"as trained": tests.classify(model, head, model.koopman_forward(z6, 1))}
            print(f"{name} | {data}: accuracy {tests.rate(answers['as trained'], labels):.2f} (real CLIP {tests.rate(clip, labels):.2f}), same answer as real CLIP {tests.rate(answers['as trained'], clip):.2f}", flush=True)

            spectrum = {}
            if data == "imagenet" and stats["arm"] != "mlp":
                spectrum = get_spectrum(model, z6)
                print("  " + ", ".join(f"{key} {value:.2f}" for key, value in spectrum.items()))
                answers = tests.ablate(model, head, z6, labels, clip)

            swapped = swap_inputs(model, head, states, patches, z6)
            print("  " + ", ".join(f"{row}: accuracy {tests.rate(given, labels):.2f}" for row, given in swapped.items()), flush=True)
            answers |= swapped

            for variant, given in answers.items():
                scores = {"accuracy": tests.rate(given, labels), "same_as_clip": tests.rate(given, clip), "same_as_trained": tests.rate(given, answers["as trained"])}
                rows.append({"model": run, "step": step or "best", "data": data, "variant": variant, **scores, "clip_accuracy": tests.rate(clip, labels), **spectrum})

        pd.DataFrame(rows).to_csv(base.HERE / out, index=False)


if __name__ == "__main__":
    fire.Fire(main)
