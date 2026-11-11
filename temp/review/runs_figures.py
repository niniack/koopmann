"""
The five charts of the runs section, from data/runs.csv, curves.csv, references.csv and comparisons.csv.

  runs_ladder    accuracy of the reference points and the main models, worst to best
  runs_timeline  every ImageNet CLIP run as a dot: when it ended and its final accuracy
  runs_curves    accuracy by training step for the nine fresh-crop runs
  runs_changes   points gained or lost by every change of one setting
  runs_gap       accuracy on the training crops against validation accuracy over training

Accuracy is always `val/zero_shot_accuracy` (CLIP's zero-shot head on the decoded block-12 state), in percent.

Usage: runs_figures.py
"""

from datetime import datetime

import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from common import *
from runs_compare import load

RUNS = {row["name"]: row for row in load("runs.csv")}
CURVES = load("curves.csv")
REFS = {row["key"]: 100 * float(row["accuracy"]) for row in load("references.csv")}
CLIP = REFS["real_clip"]
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET]


def final(run):
    return 100 * float(RUNS[run]["final_accuracy"])


def get_curve(run, key="accuracy"):
    """(steps, values in percent) of one run, for the checks that logged `key`."""
    rows = [row for row in CURVES if row["run"] == run and row[key]]
    return [int(row["steps"]) for row in rows], [100 * float(row[key]) for row in rows]


def clean(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def clip_line(ax, label=True):
    """Real CLIP as a dashed line across a panel whose y-axis is accuracy."""
    ax.axhline(CLIP, color=INK, linewidth=1, linestyle="--")
    if label:
        ax.annotate(f"real CLIP {CLIP:.1f}", (0.01, CLIP), xycoords=("axes fraction", "data"), xytext=(0, 3), textcoords="offset points", color=INK, fontsize=8.5)


def draw_ladder():
    notes = {"on_none": "at step 15,000, where it was stopped", "on_add": "at step 15,000", "on_bilinear": "at step 15,000", "im_iso_2x4k": "stored crops", "im_mlpk_2x4k": "stored crops"}
    bars = [
        ("CLIP's head read straight off block 6", REFS["head_block6"], MUTED, ""),
        ("Real blocks 7-12, patch tokens frozen at block 6", REFS["frozen_patches"], MUTED, "measured on 5,000 images"),
        ("Linear map from block 6 to block 12", REFS["linear_map"], MUTED, ""),
        ("Plain MLP on the block-6 state, no KAE", REFS["mlp_probe"], ORANGE, "stored crops"),
        ("Real CLIP (all 12 blocks)", CLIP, INK, ""),
    ] + [(label, final(name), ORANGE if "MLP" in label else BLUE, notes.get(name, "")) for name, label in MODELS.items()]
    bars = sorted(bars, key=lambda bar: -bar[1])  # barh draws the first one at the bottom

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    ax.barh([bar[0] for bar in bars], [bar[1] for bar in bars], color=[bar[2] for bar in bars], height=0.68)
    for label, value, _, note in bars:
        inside = value > 20
        ax.text(value + (-0.7 if inside else 0.7), label, f"{value:.1f}", va="center", ha="right" if inside else "left", fontsize=9, color="white" if inside else INK, fontweight="bold")
        ax.text(65, label, note, va="center", fontsize=8, color=MUTED)
    ax.axvline(CLIP, color=INK, linewidth=1, linestyle="--", zorder=0)
    keys = [Patch(color=MUTED, label="reference point, not a training run"), Patch(color=BLUE, label="KAE"), Patch(color=ORANGE, label="MLP baseline, not a KAE"), Patch(color=INK, label="the real model")]
    ax.legend(handles=keys, loc="upper left", bbox_to_anchor=(0.33, 1.0), frameon=False, fontsize=8.5)
    ax.set_xlim(0, 84)
    ax.set_xticks(range(0, 61, 10))
    ax.set_xlabel("ImageNet validation accuracy with CLIP's zero-shot head (%)")
    ax.set_title(f"The control lifts the KAE from 52 to 59; the best KAE is {CLIP - final('share_real_w03'):.1f} points under real CLIP", loc="left")
    clean(ax, "x")
    ax.tick_params(axis="y", length=0, labelcolor=INK)
    save(fig, "runs_ladder")


def when(run):
    return datetime.strptime("2026-" + RUNS[run]["finished_est"], "%Y-%m-%d %H:%M")


def draw_timeline():
    groups = {"2": ("state = CLS token", BLUE), "3": ("patch mean and std added to the state", ORANGE), "4": ("hidden layers 2 x 4,096 or wider", AQUA),
              "6": ("hidden layers 2 x 4,096 or wider", AQUA), "7": ("fresh crops, with and without control", YELLOW), "8": ("state = CLS token, with control", MAGENTA), "9": ("the same with a penalty on the control", VIOLET)}  # fmt: skip
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for name, run in RUNS.items():
        if run["encoder"].startswith("CLIP") and run["family"][0] in groups and run["final_accuracy"]:
            colour = groups[run["family"][0]][1]
            ax.plot(when(name), final(name), "o", markersize=6.5, markeredgewidth=1.4, markeredgecolor=colour, markerfacecolor=colour if run["state_on_wandb"] == "finished" else "white")
    clip_line(ax)

    labels = [  # (run the leader points at, text, where the text sits: day of November and accuracy, alignment)
        ("im_jordan_2k_v3_both", "best with the\nCLS token alone\nas the state 40.9", (7.56, 45.6), "left"),
        ("im_jordan256_2k_std", "patch mean and std\nin the state 50.4", (8.05, 55.2), "center"),
        ("im_iso_2x4k", f"{MODELS['im_iso_2x4k']} {final('im_iso_2x4k'):.1f}", (9.17, 57.3), "center"),
        ("im_mlpk_2x4k", f"MLP in place of eigenvalues\n(older state) {final('im_mlpk_2x4k'):.1f}", (9.81, 46.2), "center"),
        ("on_none", f"{MODELS['on_none']} {final('on_none'):.1f}", (10.71, 48.6), "center"),
        ("on_add", f"additive / bilinear control\n{final('on_add'):.1f} / {final('on_bilinear'):.1f}", (10.36, 60.6), "center"),
        ("cls_mlp", f"CLS-only state with control:\nKAE {final('cls_add'):.1f}, real start {final('cls_add_real'):.1f},\nMLP in place of eigenvalues {final('cls_mlp'):.1f}", (10.95, 66.6), "center"),
        ("share_real_w1", f"control penalty\n0.1 / 0.3 / 1.0:\n{final('share_real_w01'):.1f} / {final('share_real_w03'):.1f} / {final('share_real_w1'):.1f}", (11.5, 53.2), "center"),
    ]
    for name, text, (day, accuracy), align in labels:
        spot = datetime(2026, 11, int(day), int(24 * (day % 1)), int(60 * (24 * (day % 1) % 1)))
        ax.annotate(text, (when(name), final(name)), xytext=(spot, accuracy), ha=align, va="center", fontsize=8.5, color=INK, arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8, "shrinkB": 5})

    early = sum(run["dataset"] == "imagenet" and run["accuracy_metric"].startswith("val/predicted") for run in RUNS.values())
    ax.text(0.01, 0.02, f"Not shown: the {early} ImageNet runs of 7 Nov before 16:33,\nwhich did not log this accuracy", transform=ax.transAxes, fontsize=8, color=MUTED)
    keys = [Line2D([], [], marker="o", linestyle="", color=colour, label=label) for label, colour in dict(groups.values()).items()]
    keys.append(Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor=MUTED, markeredgewidth=1.4, label="hollow: ended before its planned length"))
    ax.legend(handles=keys, loc="lower right", frameon=False, fontsize=8.5)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d Nov"))
    ax.set_xlim(datetime(2026, 11, 7, 12), datetime(2026, 11, 11, 23))
    ax.set_ylim(28, 70)
    ax.set_xlabel("when the run ended (EST; the tick is midnight)")
    ax.set_ylabel("final ImageNet accuracy (%)")
    ax.set_title("Accuracy moved in three steps: a wider state (8 Nov), wider layers (9 Nov), the control (10 Nov)", loc="left")
    clean(ax)
    save(fig, "runs_timeline")


def draw_curves():
    panels = [("Three-part state (stopped at step 15,000)", ["on_none", "on_add", "on_bilinear"]),
              ("State = CLS token, additive control", ["cls_add", "cls_add_real", "cls_mlp"]),
              ("The same with real start and a penalty", ["share_real_w01", "share_real_w03", "share_real_w1"])]  # fmt: skip
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), sharey=True, sharex=True)
    for ax, (title, names) in zip(axes, panels):
        for name, colour in zip(names, SERIES):
            steps, accuracy = get_curve(name)
            ax.plot([step / 1_000 for step in steps], accuracy, "-o", color=colour, markersize=3.5, linewidth=1.8)
        # Direct labels as lines of coloured text, best model first
        ranked = sorted(zip(names, SERIES), key=lambda pair: -final(pair[0]))
        for row, (name, colour) in enumerate(ranked):
            ax.text(0.97, 0.3 - 0.075 * row, f"{MODELS[name]}  {final(name):.1f}", transform=ax.transAxes, ha="right", color=colour, fontsize=8.5, fontweight="bold")
        clip_line(ax)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("training step (thousands)")
        clean(ax)
    axes[0].set_ylabel("ImageNet accuracy (%)")
    axes[0].set_ylim(44, 65)
    axes[0].set_xlim(0, 31)
    fig.suptitle("The control is ahead from the first check and 6.5 points ahead at step 15,000; real start and the penalty move it under half a point", x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout()
    save(fig, "runs_curves")


def draw_changes():
    rows = sorted((row for row in load("comparisons.csv") if "zero_shot" in row["metric"]), key=lambda row: float(row["difference_points"]))
    fig, ax = plt.subplots(figsize=(10, 0.265 * len(rows) + 1.6))
    ax.axvspan(-0.2, 0.2, color=GRID, zorder=0)
    for y, row in enumerate(rows):
        gain, (a, b) = float(row["difference_points"]), (round(int(row[key]) / 1_000) for key in ("steps_a", "steps_b"))
        ax.barh(y, gain, color=BLUE if gain > 0 else ORANGE, alpha=1 if row["kind"] == "one change" else 0.4, height=0.72)
        ax.text(gain + (0.25 if gain >= 0 else -0.25), y, f"{gain:+.1f}", va="center", ha="left" if gain >= 0 else "right", fontsize=8)
        ax.text(1.01, y, f"{a}k" if a == b else f"{a}k | {b}k", transform=ax.get_yaxis_transform(), va="center", fontsize=7.5, color=MUTED)
    ax.text(1.01, len(rows) - 0.1, "steps of\nthe two runs", transform=ax.get_yaxis_transform(), va="bottom", fontsize=7.5, color=MUTED)
    ax.set_yticks(range(len(rows)), [row["change"] for row in rows], fontsize=8)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(-15.5, 12.5)
    ax.axvline(0, color=INK, linewidth=0.8)
    keys = [Patch(color=BLUE, label="accuracy went up"), Patch(color=ORANGE, label="accuracy went down"), Patch(color=BLUE, alpha=0.4, label="faded: two settings changed at once,\nor the schedules were not matched"),
            Patch(color=GRID, label="grey band: 0.2 points, the most a new\nseed or a rerun moved accuracy (3 tests)")]  # fmt: skip
    ax.legend(handles=keys, loc="lower right", frameon=False, fontsize=8)
    ax.set_xlabel("change in ImageNet accuracy (points)")
    ax.set_title("What moved accuracy: the control, the KL loss, patch statistics in the state, wider layers\nand more images. Most other changes moved it by less than 1 point", loc="left")
    clean(ax, "x")
    ax.tick_params(axis="y", length=0, labelcolor=INK)
    save(fig, "runs_changes")


def draw_gap():
    panels = [("im_iso_2x4k", "3 stored crops per image"), ("im_iso_2x4k_wd", "3 stored crops, weight decay 0.1"),
              ("on_none", "Fresh crops every step"), ("cls_add", "Fresh crops, state = CLS token, control")]  # fmt: skip
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.3), sharey=True, sharex=True)
    for ax, (name, title) in zip(axes, panels):
        steps, val = get_curve(name)
        _, train = get_curve(name, "train_accuracy")
        steps, stored = [step / 1_000 for step in steps], "stored" in title
        ax.plot(steps, val, "-o", color=BLUE, markersize=3.5, linewidth=1.8)
        ax.plot(steps, train, "-o" if stored else "o", color=ORANGE, markersize=3.5 if stored else 5, linewidth=1.8)
        gap = train[-1] - val[-1] if stored else sum(t - v for t, v in zip(train, val)) / len(val)
        ax.text(0.97, 0.06, f"gap at the end: {gap:.1f} points" if stored else f"average gap over the {len(val)} checks:\n{gap:.1f} points", transform=ax.transAxes, ha="right", fontsize=8.5, color=INK)
        clip_line(ax, label=ax is axes[0])
        ax.set_title(f"{title}\n{MODELS.get(name, 'KAE, no control, weight decay')}", loc="left", fontsize=9)
        ax.set_xlabel("training step (thousands)")
        clean(ax)
    axes[0].text(13, 59.6, "training crops", color=ORANGE, fontsize=8.5, fontweight="bold", ha="right", va="bottom")
    axes[0].text(30.3, 51.3, "validation", color=BLUE, fontsize=8.5, fontweight="bold", ha="right", va="top")
    axes[2].text(16, 56.3, "one training batch\nof 1,024 fresh crops", color=ORANGE, fontsize=8.5, fontweight="bold", va="center")
    axes[0].set_ylabel("ImageNet accuracy (%)")
    axes[0].set_ylim(44, 66)
    axes[0].set_xlim(0, 31)
    fig.suptitle("Stored crops were memorized: 9.5 points easier than validation.\nWeight decay and fresh crops shrink that gap without lifting validation; the control lifts it", x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout()
    save(fig, "runs_gap")


if __name__ == "__main__":
    draw_ladder()
    draw_timeline()
    draw_curves()
    draw_changes()
    draw_gap()
