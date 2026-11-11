"""
Charts of where each model is right and wrong on ImageNet validation, image by image (questions a to d of the review).

    images_outcomes     both right / only the KAE right / only real CLIP right / both wrong, per model
    images_classes      accuracy per class, one KAE against real CLIP
    images_class_ends   the classes that lose most and least, averaged over the CLS-only KAEs
    images_groups       accuracy per broad group of classes
    images_confidence   accuracy and agreement by how sure real CLIP is
    images_mistakes     how far off a model is on the photos real CLIP gets right and the model gets wrong

Every number drawn is saved as a table in data/images_*.csv and printed as markdown (kept in data/draw_images.log).

Usage: draw_images.py
"""

from images_data import *

LIGHT_GOOD, LIGHT_BAD = "#a6dba6", "#f0b8b8"
REAL = "real CLIP"


def draw_outcomes(d):
    """One bar per model, cut into the four outcomes of a photo."""

    cut = lambda a: [percent(a["right"] & d.clip["right"]), percent(a["right"] & ~d.clip["right"]), percent(~a["right"] & d.clip["right"]), percent(~a["right"] & ~d.clip["right"])]
    rows = [[name, MODELS[name], percent(a["right"]), percent(a["agree"]), *cut(a)] for name, a in d.kae.items()]
    print(write_table("images_outcomes", ["run", "model", "accuracy", "agreement", "both right", "only model right", "only real CLIP right", "both wrong"], rows), "\n")

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    parts = {"both right": (GOOD, "white"), "only the model right": (LIGHT_GOOD, INK), "only real CLIP right": (BAD, "white"), "both wrong": (LIGHT_BAD, INK)}
    left = torch.zeros(len(rows))
    for column, (part, (colour, ink)) in enumerate(parts.items(), start=4):
        widths = torch.tensor([row[column] for row in rows])
        bars = ax.barh([row[1] for row in rows], widths, left=left, color=colour, label=part, height=0.72, edgecolor="white", linewidth=1.5)
        ax.bar_label(bars, labels=[f"{width:.1f}" for width in widths], label_type="center", fontsize=8, color=ink)
        left += widths
    lost, won = [row[6] for row in rows], [row[5] for row in rows]
    ax.set_title(f"Of 100 photos, a model loses {min(lost):.0f} to {max(lost):.0f} that real CLIP gets right and wins {min(won):.0f} to {max(won):.0f} that it gets wrong", pad=28)
    ax.set(xlim=(0, 100), xlabel="share of the 50,000 ImageNet validation photos (percent)")
    ax.invert_yaxis()
    ax.grid(False)
    ax.tick_params(axis="y", labelcolor=INK, length=0)
    ax.legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1), fontsize=9, handlelength=1.2, columnspacing=1.5)
    save(fig, "images_outcomes")


def class_table(d):
    """Accuracy per class for real CLIP and every model, saved as a table. Returns the tensors."""

    clip = by_class(d.clip["right"], d.labels)
    kae = {name: by_class(a["right"], d.labels) for name, a in d.kae.items()}
    group = groups.class_groups()
    rows = [[c, CLASS_NAMES[c], groups.NAMES[group[c]], clip[c].item(), *[kae[name][c].item() for name in MODELS]] for c in range(1000)]
    write_table("images_classes", ["class", "name", "group", "real CLIP", *MODELS], rows)
    return clip, kae


def draw_classes(clip, kae, name="cls_add"):
    """Accuracy of one KAE against real CLIP, a dot per class."""

    loss = clip - kae[name]
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.plot([0, 100], [0, 100], color=MUTED, linewidth=1)
    ax.plot([20, 100], [0, 80], color=GRID, linewidth=1)
    jitter = torch.rand(2, 1000, generator=torch.Generator().manual_seed(0)) * 1.4 - 0.7  # NOTE: A class has 50 photos, so dots would stack
    ax.scatter(clip + jitter[0], kae[name] + jitter[1], s=14, color=BLUE, alpha=0.4, linewidth=0)
    ax.text(97, 99, "same accuracy", color=MUTED, fontsize=8, ha="right", va="bottom", rotation=45, rotation_mode="anchor")
    ax.text(97, 79, "20 points lower", color=MUTED, fontsize=8, ha="right", va="bottom", rotation=45, rotation_mode="anchor")
    for c in loss.topk(4).indices.tolist():
        ax.annotate(CLASS_NAMES[c], (clip[c], kae[name][c]), xytext=(6, -9), textcoords="offset points", fontsize=8, color=INK)
    ax.set_title(f"{(loss > 0).sum()} of 1,000 classes lose accuracy, {(loss < 0).sum()} gain;\n{(loss >= 20).sum()} lose 20 points or more ({MODELS[name]})")
    ax.set(xlim=(0, 102), ylim=(0, 102), xlabel="real CLIP: accuracy on the class (percent of its 50 photos)", ylabel="KAE: accuracy on the class (percent)", aspect="equal")
    save(fig, "images_classes")
    print(f"per class, {name}: lose {(loss > 0).sum()}, gain {(loss < 0).sum()}, same {(loss == 0).sum()}, lose >= 10: {(loss >= 10).sum()}, lose >= 20: {(loss >= 20).sum()}, mean loss {loss.mean():.2f}, largest {loss.max():.0f}\n")


def draw_class_ends(clip, kae, count=15):
    """The classes whose accuracy drops most and least from real CLIP to the CLS-only KAEs (mean of the five)."""

    loss = clip - torch.stack([kae[name] for name in CLS_KAES]).mean(dim=0)
    everywhere = torch.stack([clip - kae[name] for name in MODELS]).min(dim=0).values
    order = loss.argsort(descending=True)
    rows = [[c, CLASS_NAMES[c], clip[c].item(), loss[c].item(), everywhere[c].item()] for c in [*order[:30].tolist(), *order[-30:].tolist()]]
    print(write_table("images_class_ends", ["class", "name", "real CLIP accuracy", "mean loss, 5 CLS-only KAEs (points)", "smallest loss over all 11 models"], rows), "\n")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax, chosen, title in zip(axes, [order[:count], order.flip(0)[:count]], ["lose most", "lose least (below zero: the KAEs gain)"]):
        ax.barh([CLASS_NAMES[c] for c in chosen.tolist()], loss[chosen], color=BLUE, height=0.6)
        ax.invert_yaxis()
        ax.axvline(0, color=MUTED, linewidth=1)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelcolor=INK, length=0)
        ax.set_title(f"The {count} classes that {title}", fontsize=10)
        ax.set_xlabel("accuracy lost against real CLIP (points)")
    fig.suptitle("The biggest losses are on look-alike classes; a few classes gain", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    save(fig, "images_class_ends")


def draw_groups(d, shown=("cls_add", "on_none")):
    """Accuracy per broad group: real CLIP and two KAEs as dots on one line per group."""

    count = torch.bincount(d.group, minlength=len(groups.NAMES))
    clip = by_group(d.clip["right"], d.group)
    kae = {name: by_group(a["right"], d.group) for name, a in d.kae.items()}
    rows = [[groups.NAMES[g], count[g].item() // 50, count[g].item(), clip[g].item(), *[kae[name][g].item() for name in MODELS]] for g in range(len(count))]
    print(write_table("images_groups", ["group", "classes", "photos", "real CLIP", *MODELS], rows), "\n")

    order = (clip - kae[shown[0]]).argsort(descending=True)
    names = [f"{groups.NAMES[g]} ({count[g] // 50} classes)" for g in order.tolist()]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.hlines(names, torch.stack([kae[name][order] for name in shown]).min(dim=0).values, clip[order], color=GRID, linewidth=2, zorder=1)
    for values, colour, label in [(clip, INK, REAL), *[(kae[name], MAIN[name], MODELS[name]) for name in shown]]:
        ax.scatter(values[order], names, s=55, color=colour, edgecolor="white", linewidth=1.5, zorder=2, label=label)
        ax.annotate(label, (values[order][0], 0), xytext=(0, 11), textcoords="offset points", fontsize=8, color=INK, ha="center")
    lost = clip - kae[shown[0]]
    ax.set_title(f"{groups.NAMES[order[0]]} lose most ({lost[order[0]]:.0f} points), {groups.NAMES[order[-1]].lower()} least ({lost[order[-1]]:.0f}); without control every group is lower", pad=14)
    ax.set(xlabel="accuracy on ImageNet validation (percent)", ylim=(len(names) - 0.4, -1.1))
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelcolor=INK, length=0)
    save(fig, "images_groups")


def draw_confidence(d):
    """Accuracy and agreement in ten equal bins of the gap between real CLIP's best and second-best score."""

    bins = deciles(d.clip["gap"])
    share = lambda mask: [percent(mask[bins == b]) for b in range(10)]
    edges = [f"{d.clip['gap'][bins == b].min():.2f} to {d.clip['gap'][bins == b].max():.2f}" for b in range(10)]
    columns = {REAL: share(d.clip["right"])}
    for name, a in d.kae.items():
        columns[f"{name} accuracy"], columns[f"{name} agreement"] = share(a["right"]), share(a["agree"])
    print(write_table("images_confidence", ["tenth", "gap of real CLIP (log probability)", *columns], [[b + 1, edges[b], *[v[b] for v in columns.values()]] for b in range(10)]), "\n")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    x = torch.arange(1, 11)
    axes[0].plot(x, columns[REAL], color=INK, linewidth=2, marker="o", markersize=5, label=REAL)
    for name, colour in MAIN.items():
        for ax, kind in zip(axes, ["accuracy", "agreement"]):
            ax.plot(x, columns[f"{name} {kind}"], color=colour, linewidth=2, marker="o", markersize=5, label=MODELS[name])
    for ax, title in zip(axes, ["Accuracy", "Agreement with real CLIP's answer"]):
        ax.set_title(title, fontsize=10)
        ax.set(xticks=x, xlabel="tenths of the photos, from real CLIP least sure (1) to most sure (10)", ylim=(0, 101))
    axes[0].set_ylabel("percent of the tenth's 5,000 photos")
    axes[0].legend(fontsize=8, loc="upper left")
    first, last = columns["cls_add agreement"][0], columns["cls_add agreement"][-1]
    fig.suptitle(f"The KAEs follow real CLIP where it is sure ({last:.0f}% the same answer) and drift where it is unsure ({first:.0f}%)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    save(fig, "images_confidence")


def draw_mistakes(d):
    """On photos real CLIP gets right and a model gets wrong: how near is the model's answer?"""

    class_group = groups.class_groups()
    in_top5 = lambda top5, classes: (top5 == classes[:, None]).any(dim=1)
    rows = []
    for name, a in d.kae.items():
        lost, answer = d.clip["right"] & ~a["right"], a["top5"][:, 0]
        near, known, same = in_top5(a["top5"], d.labels), in_top5(d.clip["top5"], answer), class_group[answer] == d.group
        both_wrong = ~d.clip["right"] & ~a["right"]
        rows.append([name, MODELS[name], lost.sum().item(), percent(near[lost]), percent(known[lost]), percent(same[lost]), percent((~near & ~same)[lost]), percent(a["agree"][both_wrong])])
    wrong, answer = ~d.clip["right"], d.clip["top5"][:, 0]
    clip_row = [percent(in_top5(d.clip["top5"], d.labels)[wrong]), percent((class_group[answer] == d.group)[wrong])]
    header = ["run", "model", "photos lost", "label in model's 5 best", "model's answer in real CLIP's 5 best", "model's answer in the label's group", "far miss: neither 5 best nor group", "both wrong: same wrong answer"]
    print(write_table("images_mistakes", header, rows))
    print(f"real CLIP's own {wrong.sum()} mistakes: label in its 5 best {clip_row[0]:.2f}, its answer in the label's group {clip_row[1]:.2f}\n")

    labels = [row[1] for row in rows] + ["Real CLIP's own mistakes"]
    kinds = {"the right answer is among the model's 5 best": (3, BLUE), "the model's answer is among real CLIP's 5 best": (4, ORANGE), "the model's answer is in the same broad group": (5, AQUA)}
    fig, ax = plt.subplots(figsize=(9, 5))
    for kind, (column, colour) in kinds.items():
        values = [row[column] for row in rows] + ([clip_row[0]] if column == 3 else [float("nan")] if column == 4 else [clip_row[1]])
        ax.scatter(values, labels, s=55, color=colour, edgecolor="white", linewidth=1.5, label=kind, zorder=2)
    near = [row[3] for row in rows]
    ax.set_title(f"Most lost photos are near misses: in {min(near):.0f} to {max(near):.0f} of 100 the right answer is still among the model's 5 best", pad=42)
    ax.set(xlim=(0, 100), xlabel="share of the photos real CLIP gets right and the model gets wrong (percent)")
    ax.invert_yaxis()
    ax.axhline(len(rows) - 0.5, color=GRID, linewidth=1)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelcolor=INK, length=0)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1), fontsize=8.5, ncol=1, handletextpad=0.3, borderaxespad=0.2)
    save(fig, "images_mistakes")


if __name__ == "__main__":
    data = load()
    draw_outcomes(data)
    clip_classes, kae_classes = class_table(data)
    draw_classes(clip_classes, kae_classes)
    draw_class_ends(clip_classes, kae_classes)
    draw_groups(data)
    draw_confidence(data)
    draw_mistakes(data)
