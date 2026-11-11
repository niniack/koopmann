"""
Where the KAEs succeed and fail, from the saved predictions (data/preds_*.pt, data/shift_*.pt made by save_preds.py).
Charts only, no photos. Prints every number the report quotes.

  images_outcomes  per model: both right / only real CLIP right / only the KAE right / both wrong
  images_groups    points lost against real CLIP, by group of classes, without and with the control
  images_sureness  agreement with real CLIP by how sure real CLIP is
  images_shift     accuracy on ImageNet-V2 and ImageNet-Sketch against ImageNet

Usage: images_report.py
"""

import torch

from common import *
from groups import NAMES, class_groups

SHOWN = ["on_none", "on_add", "cls_add", "cls_add_real", "share_real_w03", "cls_mlp"]
clip = torch.load(DATA / "preds_clip.pt")
labels, clip_answer = clip["labels"], clip["top5"][:, 0]
preds = {name: torch.load(DATA / f"preds_{name}.pt") for name in MODELS}


def plot_outcomes():
    fig, ax = plt.subplots(figsize=(9, 3.6))
    parts = [("both right", GOOD), ("only real CLIP right", BAD), ("only the KAE right", AQUA), ("both wrong", GRID)]
    for y, name in enumerate(SHOWN):
        answer = preds[name]["top5"][:, 0]
        kae_right, clip_right = answer == labels, clip_answer == labels
        shares = [100 * (kae_right & clip_right).float().mean(), 100 * (~kae_right & clip_right).float().mean(), 100 * (kae_right & ~clip_right).float().mean(), 100 * (~kae_right & ~clip_right).float().mean()]
        left = 0
        for (label, color), share in zip(parts, shares):
            ax.barh(y, share, left=left, color=color, height=0.6, label=label if y == 0 else None)
            if share > 4:
                ax.text(left + share / 2, y, f"{share:.0f}", ha="center", va="center", fontsize=9, color="white" if color != GRID else INK)
            left += share
        top5 = (preds[name]["top5"] == clip_answer[:, None]).any(dim=1)
        lost = ~kae_right & clip_right
        print(f"{name}: both right {shares[0]:.1f}, only CLIP {shares[1]:.1f}, only KAE {shares[2]:.1f}, both wrong {shares[3]:.1f}; where only CLIP is right, CLIP's answer in the KAE's top 5: {100 * top5[lost].float().mean():.1f}%")
    ax.set_yticks(range(len(SHOWN)), [MODELS[name] for name in SHOWN])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of the 50,000 validation images")
    ax.legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_title("The KAEs miss 8 to 17% of images real CLIP gets right, and get about 5.6% that CLIP misses", pad=28)
    save(fig, "images_outcomes")


def group_accuracy(answer):
    group = class_groups()[labels]
    return torch.stack([100 * (answer[group == index] == labels[group == index]).float().mean() for index in range(len(NAMES))])


def plot_groups():
    real = group_accuracy(clip_answer)
    lost = {name: real - group_accuracy(preds[name]["top5"][:, 0]) for name in ("on_none", "cls_add_real")}
    order = lost["cls_add_real"].argsort()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for offset, (name, color) in zip((0.2, -0.2), [("on_none", MUTED), ("cls_add_real", BLUE)]):
        ax.barh(torch.arange(len(NAMES)) + offset, lost[name][order], height=0.38, color=color, label=MODELS[name])
    ax.set_yticks(range(len(NAMES)), [f"{NAMES[i]} ({real[i]:.0f}%)" for i in order])
    ax.set_xlabel("points below real CLIP (real CLIP's accuracy in brackets)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_title("The control helps every group; animals now lose least, clothing and household objects most")
    for i in order.tolist():
        print(f"{NAMES[i]}: CLIP {real[i]:.1f}, lost no control {lost['on_none'][i]:.1f}, lost CLS-only real start {lost['cls_add_real'][i]:.1f}")
    save(fig, "images_groups")


def plot_sureness():
    bins = torch.quantile(clip["gap"], torch.linspace(0, 1, 11))
    which = torch.bucketize(clip["gap"], bins[1:-1])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for name, color in [("on_none", MUTED), ("cls_add_real", BLUE)]:
        same = preds[name]["top5"][:, 0] == clip_answer
        values = [100 * same[which == b].float().mean() for b in range(10)]
        ax.plot(range(1, 11), values, color=color, lw=2, marker="o", ms=4)
        ax.annotate(MODELS[name], (10, values[-1]), textcoords="offset points", xytext=(6, 0), va="center", fontsize=9, color=color)
        ax.set_xlim(0.5, 13.5)
        print(f"{name}: agreement with CLIP by CLIP's sureness decile: {[round(float(v), 1) for v in values]}")
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("how sure real CLIP is (tenths of the images, least sure on the left)")
    ax.set_ylabel("same answer as real CLIP (%)")
    ax.set_ylim(0, 100)
    ax.set_title("The KAE copies CLIP where CLIP is sure and drifts where CLIP is unsure")
    save(fig, "images_sureness")


def plot_shift():
    sets = {"ImageNet": None, "ImageNet-V2": "imagenetv2", "ImageNet-Sketch": "imagenet_sketch"}
    rows = {"real CLIP": "clip", **{MODELS[name]: name for name in ("on_none", "on_add", "cls_add_real", "cls_mlp")}}
    fig, ax = plt.subplots(figsize=(8, 3.4))
    colors = [INK, MUTED, ORANGE, BLUE, AQUA]
    for i, ((label, name), color) in enumerate(zip(rows.items(), colors)):
        values = []
        for key in sets.values():
            if key is None:
                answer, truth = (clip_answer if name == "clip" else preds[name]["top5"][:, 0]), labels
            else:
                data = torch.load(DATA / f"shift_{key}_{name}.pt")
                answer, truth = data["top5"][:, 0], torch.load(DATA / f"shift_{key}_clip.pt")["labels"]
            values.append(100 * (answer == truth).float().mean())
        ax.bar([x + (i - 2) * 0.16 for x in range(3)], values, width=0.15, color=color, label=label)
        print(f"{label}: {[round(float(v), 1) for v in values]} (ImageNet, V2, Sketch)")
    ax.set_xticks(range(3), list(sets))
    ax.set_ylabel("accuracy (%)")
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="x", visible=False)
    ax.set_title("The gap to real CLIP is 4 points on ImageNet, 6 on new photos, and 19 on sketches")
    save(fig, "images_shift")


if __name__ == "__main__":
    plot_outcomes()
    plot_groups()
    plot_sureness()
    plot_shift()
    lost_all = (clip_answer == labels) & torch.stack([preds[n]["top5"][:, 0] != labels for n in SHOWN]).all(dim=0)
    print(f"images real CLIP gets right and all {len(SHOWN)} shown KAEs get wrong: {lost_all.sum()} ({100 * lost_all.float().mean():.1f}%)")
