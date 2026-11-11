"""
What the image charts share: every model's saved answers (save_preds.py) as right / wrong per image, and small helpers.

`load()` gives a namespace with, for the chosen set of images:
    labels, group (broad group of each image's class, an index into groups.NAMES)
    clip: the real model's {top5, gap, right}
    kae:  {model name: {top5, gap, right, agree}}     right = best answer is the label, agree = same answer as the real model
"""

from types import SimpleNamespace

import open_clip

import groups
from common import *

CLASS_NAMES = open_clip.IMAGENET_CLASSNAMES
# The models most charts follow, with one colour each in every chart. Real CLIP is always INK.
MAIN = {"cls_add": BLUE, "on_none": ORANGE, "cls_mlp": AQUA}
CLS_KAES = ["cls_add", "cls_add_real", "share_real_w01", "share_real_w03", "share_real_w1"]
percent = lambda mask: 100 * mask.float().mean().item()


def load(prefix="preds", names=tuple(MODELS), real="clip"):
    """The saved answers of one set of images (see the top of this file)."""

    clip = torch.load(DATA / f"{prefix}_{real}.pt")
    labels = clip.pop("labels")
    clip["right"] = clip["top5"][:, 0] == labels
    kae = {name: torch.load(DATA / f"{prefix}_{name}.pt") for name in names}
    for answers in kae.values():
        answers["right"] = answers["top5"][:, 0] == labels
        answers["agree"] = answers["top5"][:, 0] == clip["top5"][:, 0]
    group = groups.class_groups()[labels] if labels.max() > 99 else None  # NOTE: CIFAR-100 has its own 100 classes
    return SimpleNamespace(labels=labels, group=group, clip=clip, kae=kae)


def by_class(mask, labels, classes=1000):
    """Percent of each class's images inside a mask. shape: [classes]"""

    count = torch.zeros(classes).index_add_(0, labels, torch.ones(len(labels)))
    return 100 * torch.zeros(classes).index_add_(0, labels, mask.float()) / count


def by_group(mask, group):
    """Percent of each broad group's images inside a mask. shape: [groups]"""

    return torch.tensor([percent(mask[group == index]) for index in range(len(groups.NAMES))])


def deciles(values):
    """Ten equal-sized bins of the images by a value: 0 for the lowest tenth, 9 for the highest. shape: [images]"""

    return (values.argsort().argsort() * 10 // len(values)).clamp(max=9)


def write_table(name, header, rows):
    """Saves a table as data/<name>.csv, and returns it as markdown for the digest."""

    text = lambda value: f"{value:.2f}" if isinstance(value, float) else str(value)
    rows = [[text(value) for value in row] for row in rows]
    (DATA / f"{name}.csv").write_text("\n".join(",".join(f'"{v}"' if "," in v else v for v in row) for row in [header, *rows]) + "\n")
    return "\n".join("| " + " | ".join(row) + " |" for row in [header, ["---"] * len(header), *rows])
