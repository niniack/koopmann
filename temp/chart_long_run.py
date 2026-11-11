"""Val accuracy by epoch: the 12-epoch DINOv3 pilot against the 30-epoch run, with the real model as the reference."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb
from dotenv import load_dotenv

load_dotenv("/home/nsa325/work/koopmann/.env")
BLUE, ORANGE, MUTED, INK = "#2a78d6", "#eb6834", "#898781", "#52514e"
api = wandb.Api()


def get_curve(run_id, key):
    rows = api.run(f"nishantaswani/tokae/{run_id}").history(keys=[key], pandas=False)
    return [100 * row[key] for row in rows]


short, long = get_curve("urzqs9i6", "val/zero_shot_accuracy"), get_curve("ze4m5akv", "val/zero_shot_accuracy")
long_train = get_curve("ze4m5akv", "train/zero_shot_accuracy")
print(len(short), len(long), short[-1], long[-1], long_train[-1])
REAL = 83.94

fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=160)
ax.axhline(REAL, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
ax.text(30.4, REAL, f"real DINOv3\n{REAL:.1f}", va="center", ha="left", fontsize=9, color=INK)
ax.plot(range(1, len(long_train) + 1), long_train, color=BLUE, linewidth=1.2, alpha=0.45)
ax.text(30.4, long_train[-1], f"30 epochs,\ntraining images\n{long_train[-1]:.1f}", va="center", ha="left", fontsize=9, color=INK)
ax.plot(range(1, len(short) + 1), short, color=ORANGE, linewidth=2)
ax.scatter([len(short)], [short[-1]], s=42, color=ORANGE, edgecolor="white", linewidth=1.5, zorder=3)
ax.text(len(short) + 0.5, short[-1] - 0.7, f"12 epochs, validation {short[-1]:.1f}", ha="left", va="center", fontsize=9, color=INK)
ax.plot(range(1, len(long) + 1), long, color=BLUE, linewidth=2)
ax.scatter([len(long)], [long[-1]], s=42, color=BLUE, edgecolor="white", linewidth=1.5, zorder=3)
ax.text(30.4, long[-1], f"30 epochs,\nvalidation\n{long[-1]:.1f}", va="center", ha="left", fontsize=9, color=INK)
ax.annotate("", xy=(27, REAL - 0.6), xytext=(27, long[26] + 0.8), arrowprops=dict(arrowstyle="<->", color=MUTED, linewidth=1))
ax.text(26.6, (REAL + long[26]) / 2 + 3.5, f"gap to the real model\n{REAL - long[-1]:.0f} points", ha="right", va="center", fontsize=9, color=INK)
ax.set_xlim(0.5, 30.2)
ax.set_ylim(48, 88)
ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
ax.set_xlabel("epoch", fontsize=9, color=INK)
ax.set_ylabel("ImageNet top-1 through the real head (%)", fontsize=9, color=INK)
ax.tick_params(colors=INK, labelsize=9, length=0)
ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
ax.set_axisbelow(True)
for side in ["top", "right", "left"]:
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color("#c9c8c3")
ax.set_title("2.5 times the epochs buys the KAE about one point (DINOv3, blocks 6 to 12)", fontsize=11, color="#1f1f1d", loc="left", pad=12)
fig.subplots_adjust(left=0.08, right=0.84, top=0.9, bottom=0.12)
fig.savefig("chart_long_run.png")
print("saved")
