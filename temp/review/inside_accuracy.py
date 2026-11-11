"""
Question a: where the accuracy goes. Per model, the gap to real CLIP is split in two:
  lost by the autoencoder   real block 12, against the real block 12 encoded and decoded (TopK included)
  lost by the step          that, against the model's prediction from block 6
The same split is made for the state FVU of the CLS token (the part of the state the head reads).

Writes figs/inside_accuracy.png and data/inside_accuracy.json.
"""

import json

from common import *
from inside_load import load_results

names = list(MODELS)
results = [load_results(name) for name in names]
READ_OUTS = ["real block 12", "block 12, encoded and decoded", "block 6, encoded and decoded", "prediction"]
accuracy = {key: torch.tensor([100 * r["read_outs"][key]["accuracy"] for r in results]) for key in READ_OUTS}
fvu = {key: torch.tensor([r["fvu_cls"][key] for r in results]) for key in READ_OUTS}

table = {}
for i, name in enumerate(names):
    table[name] = {"label": MODELS[name], **{f"accuracy, {key}": round(accuracy[key][i].item(), 2) for key in READ_OUTS}, **{f"CLS FVU, {key}": round(fvu[key][i].item(), 4) for key in READ_OUTS}}
    table[name]["agreement with real CLIP, prediction"] = round(100 * results[i]["read_outs"]["prediction"]["agreement"], 2)
    table[name]["whole-state FVU, prediction"] = round(results[i]["fvu"]["prediction"], 4)
    print(name, table[name])
(DATA / "inside_accuracy.json").write_text(json.dumps(table, indent=1, ensure_ascii=False))

coded, end = "block 12, encoded and decoded", "prediction"
panels = [
    ("Points of accuracy lost against real CLIP (63.3)", accuracy["real block 12"] - accuracy[coded], accuracy[coded] - accuracy[end]),
    ("State error of the CLS token (FVU, 0 is perfect)", fvu[coded], fvu[end] - fvu[coded]),
]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
rows = torch.arange(len(names))
for ax, (xlabel, by_autoencoder, by_step) in zip(axes, panels):
    by_autoencoder = by_autoencoder.clamp_min(0)
    ax.barh(rows, by_autoencoder, height=0.62, color=BLUE, edgecolor="white", lw=1.5, label="lost by the autoencoder alone")
    ax.barh(rows, by_step, left=by_autoencoder, height=0.62, color=ORANGE, edgecolor="white", lw=1.5, label="lost by the step from block 6")
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
axes[0].set_yticks(rows, [MODELS[name] for name in names])
axes[0].invert_yaxis()
axes[0].tick_params(axis="y", colors=INK, length=0)
axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9, handlelength=1.2, columnspacing=1.5, labelcolor=INK)
fig.suptitle("Most accuracy is lost in the step from block 6. The autoencoder costs few points, but about half of the state error", x=0.01, ha="left", color=INK, fontsize=12)
fig.tight_layout()
save(fig, "inside_accuracy")
