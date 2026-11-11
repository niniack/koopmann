"""
The two speed figures of the report, from data/speed.json (measured by speed_time.py and speed_memory.py).

  speed_whole     time per batch of 256 images, real CLIP against the CLS-only KAE, split into blocks 1-6 and the rest
  speed_tradeoff  accuracy against time per image (GPU, batch 256)

Usage: speed_report_figs.py
"""

import json
from statistics import median

from common import *

speed = json.loads((DATA / "speed.json").read_text())
ACCURACY = {"full": 63.34, "im_iso_2x4k": 52.11, "cls_add_real": 58.94, "cls_mlp": 60.46}
NAMES = {"full": "real CLIP", "im_iso_2x4k": MODELS["im_iso_2x4k"], "cls_add_real": MODELS["cls_add_real"], "cls_mlp": MODELS["cls_mlp"]}


def times(ms, batch):
    """Median milliseconds per part for one batch size."""

    return {part: median(values) for part, values in ms[str(batch)].items()}


def plot_whole():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    for ax, (device, ms) in zip(axes, [("GPU (RTX 6000 Ada)", speed["time_cuda"]["ms"]), ("CPU (16 threads)", speed["time_cpu"][0]["ms"])]):
        t = times(ms, 256)
        rows = [("real CLIP", t["back"]), ("CLS-only KAE", t["kae:cls_add_real"])]
        for y, (label, rest) in enumerate(rows):
            ax.barh(y, t["front"], color=MUTED, height=0.55)
            ax.barh(y, rest + t["head"], left=t["front"], color=INK if y == 0 else BLUE, height=0.55)
            ax.text(t["front"] + rest + t["head"], y, f"  {t['front'] + rest + t['head']:,.0f} ms", va="center", color=INK, fontsize=9)
        ax.text(t["front"] / 2, -0.45, "blocks 1-6", ha="center", color=MUTED, fontsize=9)
        ax.text(t["front"] + t["back"] / 2, -0.45, "blocks 7-12 + head", ha="center", color=INK, fontsize=9)
        ax.text(t["front"], 1.4, f"  KAE in place of blocks 7-12: {t['kae:cls_add_real']:.1f} ms\n  ({t['back'] / t['kae:cls_add_real']:.0f} times faster than the blocks it replaces)", color=BLUE, fontsize=9, va="center")
        ax.set_yticks([0, 1], [label for label, _ in rows])
        ax.set_ylim(-0.7, 1.8)
        ax.invert_yaxis()
        ax.set_xlim(0, (t["front"] + t["back"]) * 1.25)
        ax.set_xlabel("milliseconds per batch of 256 images")
        ax.set_title(device)
        ax.grid(axis="y", visible=False)
    fig.suptitle("The KAE makes the second half 14 to 21 times faster, but blocks 1-6 still run, so the whole model is about 1.9 times faster", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    save(fig, "speed_whole")


def plot_tradeoff():
    t = times(speed["time_cuda"]["ms"], 256)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for key, accuracy in ACCURACY.items():
        per_image = 1000 * (t["full"] if key == "full" else t[f"whole:{key}"]) / 256
        color = INK if key == "full" else (ORANGE if key == "cls_mlp" else BLUE)
        ax.scatter(per_image, accuracy, s=60, color=color, zorder=3)
        ax.annotate(NAMES[key], (per_image, accuracy), textcoords="offset points", xytext=(-8, 0), ha="right", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, 500)
    ax.set_ylim(48, 66)
    ax.set_xlabel("microseconds per image, whole model (GPU, batch 256)")
    ax.set_ylabel("ImageNet accuracy (%)")
    ax.set_title("Half the time of real CLIP for 3 to 4 points less accuracy")
    save(fig, "speed_tradeoff")


if __name__ == "__main__":
    plot_whole()
    plot_tradeoff()
