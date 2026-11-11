"""
The five charts of the speed section, from data/speed.json (run speed_summary.py first).

    speed_time      the replaced half: blocks 7-12 against each stand-in, for 1, 16 and 256 images, CPU and GPU
    speed_whole     256 images end to end as a stacked bar: blocks 1-6, then blocks 7-12 or the stand-in
    speed_tradeoff  accuracy against end-to-end time per image
    speed_memory    GPU memory against batch size, real CLIP and the KAE pipelines
    speed_parts     where the CLS-only KAE's parameters and multiply-adds sit
"""

from common import *  # noqa: F403
from speed_models import KAES, load

COLOURS = {"im_iso_2x4k": BLUE, "cls_add_real": ORANGE, "cls_mlp": AQUA, "share_real_w03": YELLOW}
DEVICES = {"cpu": "CPU, 16 threads", "gpu": "GPU"}
BATCHES = [1, 16, 256]
KAE_LABELS = [MODELS[name] for name in KAES]
KAE_COLOURS = [COLOURS[name] for name in KAES]


def ms_text(ms):
    return f"{ms:,.0f} ms" if ms >= 100 else f"{ms:.1f} ms" if ms >= 10 else f"{ms:.2f} ms"


def times_text(ratio):
    if 0.95 < ratio < 1.05:
        return "same speed"
    return f"{ratio:.{0 if ratio >= 10 else 1}f} times faster" if ratio > 1 else f"{1 / ratio:.1f} times slower"


def chart_time(summary):
    fig, axes = plt.subplots(3, 2, figsize=(9.5, 6.4), sharey=True)
    for column, (device, title) in enumerate(DEVICES.items()):
        for row, batch in enumerate(BATCHES):
            ax, time, speedup = axes[row, column], summary["time"][device][str(batch)], summary["speedup"][device][str(batch)]
            ms = [time["back"]["median"]] + [time[f"kae:{name}"]["median"] for name in KAES]
            notes = [""] + [", " + times_text(speedup[name]["half"]) for name in KAES]
            ax.barh(["CLIP blocks 7 to 12"] + KAE_LABELS, ms, color=[INK] + KAE_COLOURS, height=0.7)
            for position, (value, note) in enumerate(zip(ms, notes)):
                ax.text(value, position, f" {ms_text(value)}{note}", va="center", fontsize=9, color=INK)
            ax.set_xlim(0, 1.9 * max(ms))
            ax.set_xticks([])
            ax.grid(False)
            ax.spines["bottom"].set_visible(False)
            ax.set_title(f"{title}: {batch} image{'s' if batch > 1 else ''} per batch", fontsize=10)
    axes[0, 0].invert_yaxis()
    fig.suptitle("On big batches the KAE step is 15 to 20 times faster than the blocks it replaces.\nFor one image at a time the gain shrinks on the GPU and is gone on the CPU.", x=0.02, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, "speed_time")


def chart_whole(summary):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
    for ax, (device, title) in zip(axes, DEVICES.items()):
        time, speedup = summary["time"][device]["256"], summary["speedup"][device]["256"]
        first = time["front"]["median"]
        second = [time["back"]["median"]] + [time[f"kae:{name}"]["median"] for name in KAES]
        ax.barh(["Real CLIP"] + KAE_LABELS, first, color=MUTED, height=0.7)
        ax.barh(["Real CLIP"] + KAE_LABELS, second, left=first, color=[INK] + KAE_COLOURS, height=0.7)
        ax.text(first / 2, 0, "blocks 1 to 6", color="white", ha="center", va="center", fontsize=9)
        ax.text(first + second[0] / 2, 0, "blocks 7 to 12", color="white", ha="center", va="center", fontsize=9)
        for position, name in enumerate(KAES, start=1):
            ax.text(first + second[position], position, f" stand-in: {ms_text(second[position])}\n {speedup[name]['whole']:.2f} times faster end to end", va="center", fontsize=8.5, color=INK)
        ax.set_xlabel("Milliseconds for 256 images")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    fig.suptitle("End to end the model is about 1.8 times faster: blocks 1 to 6 now take almost all of the time", x=0.02, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, "speed_whole")


def chart_tradeoff(summary, accuracy):
    time = summary["time"]["gpu"]["256"]
    cost = {name: time[f"whole:{name}"]["median"] / 256 for name in KAES}
    cost["share_real_w03"] = cost["cls_add_real"]  # The same layers, so the same cost
    floor = (time["front"]["median"] + time["head"]["median"]) / 256

    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.axvline(floor, color=MUTED, linestyle="--", linewidth=1)
    ax.text(floor, 50.3, "blocks 1 to 6 alone: \nno stand-in can be faster ", color=MUTED, ha="right", va="bottom", fontsize=9)
    ax.scatter(time["full"]["median"] / 256, accuracy["clip"], color=INK, s=70, zorder=3)
    ax.text(time["full"]["median"] / 256, accuracy["clip"] - 0.7, f"Real CLIP, {accuracy['clip']:.1f}%", color=INK, ha="center", va="top", fontsize=9)
    for name in sorted(cost, key=accuracy.get):
        ax.scatter(cost[name], accuracy[name], color=COLOURS[name], s=70, zorder=3)
        ax.text(cost[name] + 0.008, accuracy[name], f"{MODELS[name]}, {accuracy[name]:.1f}%", color=INK, va="center", fontsize=9)
    ax.set(xlim=(0, 0.5), ylim=(50, 65), xlabel="Milliseconds per image, end to end (GPU, 256 images per batch)", ylabel="ImageNet accuracy with CLIP's zero-shot head (%)")
    ax.set_title("Every stand-in costs the same time; accuracy is what separates them", loc="left")
    save(fig, "speed_tradeoff")


def chart_memory(memory):
    batches = [int(batch) for batch in memory["clip"]["total"]]
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for name, label, colour in (("clip", "Real CLIP", INK), ("cls_add_real", MODELS["cls_add_real"], ORANGE)):
        total = list(memory[name]["total"].values())
        left.plot(batches, total, color=colour, linewidth=2)
        left.text(1, total[0] * (1.18 if name != "clip" else 0.72), label, color=colour, fontsize=9)
    left.set(xscale="log", yscale="log", xlabel="Images per batch", ylabel="GPU memory: weights + working memory (MB)", title="Total memory")

    right.axhline(0, color=MUTED, linewidth=1)
    for name in KAES:
        extra = [memory[name]["total"][str(batch)] - memory["clip"]["total"][str(batch)] for batch in batches]
        right.plot(batches, extra, color=COLOURS[name], linewidth=2)
        right.text(1, extra[0] + 8, MODELS[name], color=COLOURS[name], fontsize=9)
    right.text(2048, 6, "real CLIP ", color=MUTED, ha="right", fontsize=9)
    right.set(xscale="log", xlabel="Images per batch", ylabel="Extra memory over real CLIP (MB)", title="The stand-in's extra memory")
    for ax in (left, right):
        ax.set_xticks([1, 4, 16, 64, 256, 1024], ["1", "4", "16", "64", "256", "1,024"])
        ax.minorticks_off()
    fig.suptitle("The KAE pipeline needs about 150 MB more than real CLIP until batches reach about 1,000 images", x=0.02, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, "speed_memory")


def chart_parts(parts):
    names = ["encoder", "decoder", "control encoder: layers on the patch tokens", "control encoder: its MLP"]
    labels = ["Encoder", "Decoder", "Control encoder:\nlayers on the 49 patch tokens", "Control encoder:\nits MLP"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2), sharey=True)
    for ax, column, title in zip(axes, (0, 1), ("Parameters (millions)", "Multiply-adds per image (millions)")):
        values = [parts[name][column] / 1e6 for name in names]
        ax.barh(labels, values, color=[BLUE, BLUE, ORANGE, ORANGE], height=0.7)
        for position, value in enumerate(values):
            ax.text(value, position, f" {value:.1f} ({value / sum(values):.0%})", va="center", fontsize=9, color=INK)
        ax.set(xlim=(0, 1.3 * max(values)), xlabel=title)
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    axes[1].text(0.98, 0.86, "state path", color=BLUE, ha="right", fontsize=10, fontweight="bold", transform=axes[1].transAxes)
    axes[1].text(0.98, 0.1, "control path", color=ORANGE, ha="right", fontsize=10, fontweight="bold", transform=axes[1].transAxes)
    fig.suptitle("The control encoder holds 28% of the KAE's weights and does 56% of its arithmetic", x=0.02, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, "speed_parts")


if __name__ == "__main__":
    results = load()
    chart_time(results["summary"])
    chart_whole(results["summary"])
    chart_tradeoff(results["summary"], results["check"]["accuracy"])
    chart_memory(results["summary"]["memory_gpu"]["whole"])
    chart_parts(results["count"]["kaes"]["cls_add_real"])
