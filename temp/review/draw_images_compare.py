"""
Charts that compare the models with each other, image by image (questions e to h of the review).

    images_common       how many of the eleven models miss a photo real CLIP gets right, against chance
    images_overlap      for each pair of models, how much of one's lost photos the other loses too
    images_control      what the control buys, by broad group and by class (on_none to on_add)
    images_eigen_cost   where the MLP baseline and the KAE differ (cls_mlp against cls_add)
    images_shift        accuracy on the shift sets
    images_dino         the DINOv3 KAE against real DINOv3, by how sure real DINOv3 is

Every number drawn is saved as a table in data/images_*.csv and printed as markdown (kept in data/draw_images_compare.log).

Usage: draw_images_compare.py
"""

from matplotlib.colors import LinearSegmentedColormap

from draw_images import REAL, draw_confidence
from images_data import *

SETS = {"preds": "ImageNet\nvalidation", "shift_imagenetv2": "ImageNet-V2", "shift_imagenet_sketch": "ImageNet-\nSketch", "shift_imagenet-r": "ImageNet-R",
        "shift_imagenet-a": "ImageNet-A", "shift_cifar100_test": "CIFAR-100"}  # fmt: skip
DINO = "im_dino_w21_wd_e30"


def get_lost(d):
    """Photos real CLIP gets right and each model gets wrong. shape: [models, photos real CLIP gets right]"""

    return torch.stack([~a["right"][d.clip["right"]] for a in d.kae.values()])


def by_chance(rates, photos):
    """How many photos would be lost by 0, 1, ... models, if each model lost its share of photos independently of the others."""

    chance = torch.ones(1)
    for rate in rates:
        chance = torch.cat([chance * (1 - rate), torch.zeros(1)]) + torch.cat([torch.zeros(1), chance * rate])
    return chance * photos


def draw_common(d):
    """Photos real CLIP gets right, by the number of models that get them wrong."""

    lost = get_lost(d)
    observed = torch.bincount(lost.sum(dim=0), minlength=len(lost) + 1)
    chance = by_chance(lost.float().mean(dim=1), lost.shape[1])
    print(write_table("images_common", ["models that lose the photo", "photos", "photos if independent"], [[n, observed[n].item(), chance[n].item()] for n in range(len(observed))]))
    names = list(d.kae)
    for label, chosen in {"the 9 with eigenvalues": [n for n in names if "ml" not in n], "the 8 with a control": names[3:], "the 6 CLS-only": names[5:], "the 3 without control": names[:3]}.items():
        part = lost[[names.index(n) for n in chosen]]
        print(f"lost by all of {label}: {part.all(dim=0).sum()} photos (independent: {by_chance(part.float().mean(dim=1), part.shape[1])[-1]:.1f}); by none of them: {(~part.any(dim=0)).sum()}")
    everyone = torch.stack([a["right"] for a in d.kae.values()])
    print(f"of all {len(d.labels)} photos: right for real CLIP and all models {(everyone.all(dim=0) & d.clip['right']).sum()}, wrong for real CLIP and all models {(~everyone.any(dim=0) & ~d.clip['right']).sum()}\n")

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    x = torch.arange(len(observed))
    ax.bar(x, observed, color=BLUE, width=0.62, label="photos")
    ax.scatter(x, chance, s=60, color=INK, marker="_", linewidth=2, zorder=3, label="if each model lost its photos independently of the others")
    for n in [0, len(lost)]:
        ax.annotate(f"{observed[n]:,}", (n, observed[n]), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, color=INK)
    ax.set_title(f"{observed[-1]:,} photos that real CLIP gets right are lost by all {len(lost)} models; independent mistakes would give {chance[-1]:.0f}", pad=24)
    ax.set(xticks=x, xlabel=f"how many of the {len(lost)} models get the photo wrong", ylabel=f"photos (of the {lost.shape[1]:,} real CLIP gets right)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1), ncol=2, fontsize=8.5, borderaxespad=0.1)
    save(fig, "images_common")


def draw_overlap(d):
    """For each pair of models: of the photos the row model loses, the share the column model loses too."""

    lost = get_lost(d).float()
    both = lost @ lost.T
    share = 100 * both / lost.sum(dim=1, keepdim=True)
    rate = lost.mean(dim=1)
    names = list(d.kae)
    pairs = [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))]
    rows = [[names[i], names[j], int(both[i, j]), (rate[i] * rate[j] * lost.shape[1]).item(), (both[i, j] / (rate[i] * rate[j] * lost.shape[1])).item(), share[i, j].item(), share[j, i].item()] for i, j in pairs]
    print(write_table("images_overlap", ["model A", "model B", "photos both lose", "photos if independent", "times chance", "of A's lost photos, B loses (%)", "of B's lost photos, A loses (%)"], rows), "\n")

    fig, ax = plt.subplots(figsize=(8.6, 7))
    labels = [MODELS[name] for name in names]
    shown = share.clone().fill_diagonal_(float("nan"))
    ax.imshow(shown, cmap=LinearSegmentedColormap.from_list("blues", ["#eef4fc", BLUE]), vmin=40, vmax=90)
    for i, j in [(i, j) for i in range(len(names)) for j in range(len(names)) if i != j]:
        ax.text(j, i, f"{share[i, j]:.0f}", ha="center", va="center", fontsize=8, color="white" if share[i, j] > 68 else INK)
    ax.set_xticks(range(len(names)), labels, rotation=40, ha="right", color=INK)
    ax.set_yticks(range(len(names)), labels, color=INK)
    ax.tick_params(length=0)
    ax.grid(False)
    ax.spines[:].set_visible(False)
    low = min(rate) * 100
    ax.set_title(f"Models lose the same photos: of the photos one loses, another loses {shown.nan_to_num(100).min():.0f} to {shown.nan_to_num(0).max():.0f}%.\nIf mistakes were independent this would be {low:.0f} to {max(rate) * 100:.0f}%")
    ax.set_xlabel("... the share this model also gets wrong (percent)")
    ax.set_ylabel("Of the photos real CLIP gets right and this model gets wrong ...")
    save(fig, "images_overlap")


def draw_control(d, before="on_none", after="on_add", count=12):
    """Accuracy gained from the KAE without control to the same KAE with additive control, by group and by class."""

    a, b = d.kae[before]["right"], d.kae[after]["right"]
    lost_a, lost_b = d.clip["right"] & ~a, d.clip["right"] & ~b
    print(f"{before} to {after}: fixed {(b & ~a).sum()} photos, broken {(a & ~b).sum()}, right in both {(a & b).sum()}, wrong in both {(~a & ~b).sum()}")
    print(f"photos real CLIP gets right: lost without control {lost_a.sum()}, lost with control {lost_b.sum()}, lost in both {(lost_a & lost_b).sum()}; "
          f"of those lost with control, {percent(lost_a[lost_b]):.1f}% were lost without it too\n")  # fmt: skip
    group_gain = by_group(b, d.group) - by_group(a, d.group)
    class_gain = by_class(b, d.labels) - by_class(a, d.labels)
    order = class_gain.argsort(descending=True)
    rows = [[groups.NAMES[g], by_group(a, d.group)[g].item(), by_group(b, d.group)[g].item(), group_gain[g].item()] for g in group_gain.argsort(descending=True).tolist()]
    print(write_table("images_control_groups", ["group", MODELS[before], MODELS[after], "gain (points)"], rows))
    rows = [[c, CLASS_NAMES[c], by_class(a, d.labels)[c].item(), by_class(b, d.labels)[c].item(), class_gain[c].item()] for c in [*order[:25].tolist(), *order[-10:].tolist()]]
    print(write_table("images_control_classes", ["class", "name", MODELS[before], MODELS[after], "gain (points)"], rows))
    print(f"classes that gain {(class_gain > 0).sum()}, lose {(class_gain < 0).sum()}, same {(class_gain == 0).sum()}\n")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    chosen = group_gain.argsort(descending=True)
    axes[0].barh([groups.NAMES[g] for g in chosen.tolist()], group_gain[chosen], color=BLUE, height=0.6)
    axes[0].set_title("By broad group", fontsize=10)
    axes[1].barh([CLASS_NAMES[c] for c in order[:count].tolist()], class_gain[order[:count]], color=BLUE, height=0.6)
    axes[1].set_title(f"The {count} classes that gain most (50 photos each)", fontsize=10)
    for ax in axes:
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelcolor=INK, length=0)
        ax.set_xlabel("accuracy gained with the control (points)")
    fig.suptitle(f"The control helps every group, from {group_gain.min():.1f} to {group_gain.max():.1f} points; {(class_gain > 0).sum()} of 1,000 classes gain", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    save(fig, "images_control")


def class_spread(a, b, labels, rounds=500):
    """Spread over classes of the accuracy gap between two models, and the spread if the photos they differ on fell on classes by chance."""

    differ, generator = a != b, torch.Generator().manual_seed(0)
    observed = (by_class(a, labels) - by_class(b, labels)).std()
    chance = torch.stack([(by_class(differ & flip, labels) - by_class(differ & ~flip, labels)).std() for flip in torch.rand(rounds, len(a), generator=generator) < a[differ].float().mean()])
    return observed.item(), chance.mean().item(), chance.quantile(0.975).item()


def draw_eigen_cost(d, mlp="cls_mlp", kae="cls_add"):
    """Photos only the MLP baseline gets right and photos only the KAE gets right, by real CLIP's confidence and by group."""

    a, b = d.kae[mlp]["right"], d.kae[kae]["right"]
    only_mlp, only_kae = a & ~b, b & ~a
    bins = deciles(d.clip["gap"])
    print(f"{mlp} against {kae}: only the MLP right {only_mlp.sum()}, only the KAE right {only_kae.sum()}, both right {(a & b).sum()}, both wrong {(~a & ~b).sum()}, same answer {percent(d.kae[mlp]['top5'][:, 0] == d.kae[kae]['top5'][:, 0]):.2f}%")
    print(f"of the photos only the MLP gets right, real CLIP is right on {percent(d.clip['right'][only_mlp]):.1f}%; of those only the KAE gets right, {percent(d.clip['right'][only_kae]):.1f}%")
    rows = [[n + 1, (only_mlp & (bins == n)).sum().item(), (only_kae & (bins == n)).sum().item()] for n in range(10)]
    print(write_table("images_eigen_confidence", ["tenth (real CLIP least to most sure)", "only the MLP right", "only the KAE right"], rows))
    group_rows = [[groups.NAMES[g], by_group(only_mlp, d.group)[g].item(), by_group(only_kae, d.group)[g].item(), (by_group(a, d.group) - by_group(b, d.group))[g].item()] for g in range(len(groups.NAMES))]
    print(write_table("images_eigen_groups", ["group", "only the MLP right (%)", "only the KAE right (%)", "net gain of the MLP (points)"], group_rows))
    gap = by_class(a, d.labels) - by_class(b, d.labels)
    order = gap.argsort(descending=True)
    print(write_table("images_eigen_classes", ["class", "name", "MLP minus KAE (points)"], [[c, CLASS_NAMES[c], gap[c].item()] for c in [*order[:15].tolist(), *order[-15:].tolist()]]))
    for other in [mlp, "cls_add_real"]:
        observed, chance, high = class_spread(d.kae[other]["right"], b, d.labels)
        print(f"spread over classes of the accuracy gap, {other} minus {kae}: {observed:.2f} points; by chance {chance:.2f} (97.5th percentile {high:.2f})")
    print()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1.1]})
    x, width = torch.arange(1, 11), 0.38
    for offset, column, colour, label in [(-width / 2, 1, AQUA, "only the MLP baseline is right"), (width / 2, 2, BLUE, "only the KAE is right")]:
        axes[0].bar(x + offset, [row[column] for row in rows], width=width - 0.04, color=colour, label=label)
        y = torch.arange(len(group_rows)) + offset
        axes[1].barh(y, [row[column] for row in group_rows], height=width - 0.04, color=colour)
    axes[0].set(xticks=x, xlabel="tenths of the photos, from real CLIP least sure (1) to most sure (10)", ylabel="photos")
    axes[0].set_title("By how sure real CLIP is", fontsize=10)
    axes[0].grid(axis="x", visible=False)
    axes[0].legend(fontsize=8.5)
    axes[1].set_yticks(range(len(group_rows)), [row[0] for row in group_rows], color=INK)
    axes[1].invert_yaxis()
    axes[1].grid(axis="y", visible=False)
    axes[1].tick_params(axis="y", length=0)
    axes[1].set_xlabel("share of the group's photos (percent)")
    axes[1].set_title("By broad group", fontsize=10)
    fig.suptitle(f"The MLP baseline's lead is a small net: it alone is right on {only_mlp.sum():,} photos, the KAE alone on {only_kae.sum():,}", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    save(fig, "images_eigen_cost")


def draw_shift():
    """Accuracy of real CLIP and the models on each shift set."""

    sets = {prefix: load(prefix) for prefix in SETS}
    rows = []
    for prefix, d in sets.items():
        clip = percent(d.clip["right"])
        rows.append([SETS[prefix].replace("\n", " ").replace("- ", "-"), len(d.labels), clip, *[percent(a["right"]) for a in d.kae.values()]])
        print(prefix, "| agreement with real CLIP:", ", ".join(f"{name} {percent(a['agree']):.2f}" for name, a in d.kae.items()))
        print(prefix, "| share of real CLIP's accuracy kept:", ", ".join(f"{name} {100 * percent(a['right']) / clip:.1f}" for name, a in d.kae.items()))
    print(write_table("images_shift", ["set", "photos", REAL, *MODELS], rows), "\n")

    fig, ax = plt.subplots(figsize=(10, 4.4))
    x, width = torch.arange(len(sets)), 0.2
    series = [(REAL, INK, [row[2] for row in rows]), *[(MODELS[name], colour, [percent(d.kae[name]["right"]) for d in sets.values()]) for name, colour in MAIN.items()]]
    for index, (label, colour, values) in enumerate(series):
        ax.bar(x + (index - 1.5) * width, values, width=width - 0.03, color=colour, label=label)
    kept = [100 * series[1][2][n] / series[0][2][n] for n in range(len(sets))]
    ax.set_title(f"Away from ImageNet the KAEs lose more: the CLS-only KAE keeps {kept[0]:.0f}% of real CLIP's accuracy on ImageNet, {min(kept):.0f}% at worst", pad=24)
    ax.set_xticks(x, SETS.values(), color=INK)
    ax.set(ylabel="accuracy (percent)")
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1), ncol=4, fontsize=8.5, borderaxespad=0.1, columnspacing=1.2)
    save(fig, "images_shift")


def draw_dino():
    """The DINOv3 KAE against real DINOv3 with its ImageNet probe: outcomes in numbers, confidence as a chart."""

    d = load(names=(DINO,), real="dino")
    a = d.kae[DINO]
    print(f"real DINOv3 {percent(d.clip['right']):.2f}, DINOv3 KAE {percent(a['right']):.2f}, agreement {percent(a['agree']):.2f}")
    print(f"both right {percent(a['right'] & d.clip['right']):.2f}, only the KAE right {percent(a['right'] & ~d.clip['right']):.2f}, only real DINOv3 right {percent(~a['right'] & d.clip['right']):.2f}, both wrong {percent(~a['right'] & ~d.clip['right']):.2f}")
    lost = d.clip["right"] & ~a["right"]
    print(f"photos lost {lost.sum()}: label in the KAE's 5 best {percent((a['top5'] == d.labels[:, None]).any(dim=1)[lost]):.2f}%")
    print(write_table("images_dino_groups", ["group", "real DINOv3", "DINOv3 KAE"], [[groups.NAMES[g], by_group(d.clip["right"], d.group)[g].item(), by_group(a["right"], d.group)[g].item()] for g in range(len(groups.NAMES))]))
    draw_confidence(d, {DINO: VIOLET}, {DINO: "DINOv3 KAE, no control"}, "real DINOv3", "images_dino")


if __name__ == "__main__":
    data = load()
    draw_common(data)
    draw_overlap(data)
    draw_control(data)
    draw_eigen_cost(data)
    draw_shift()
    draw_dino()
