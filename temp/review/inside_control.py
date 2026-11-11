"""
Question d: what the control does and what the eigenvalues do, per model.

The step is z12 = λ·z6 + control (for the MLP baselines: z12 = z6 + MLP(z6) + control). Per model, from inside_pass.py:
  the control's share of z12's squared size
  the accuracy lost when the control is removed
  the accuracy lost when the model's own step is removed, so z6 passes through unchanged (every λ = 1, or no MLP)

Writes figs/inside_control.png and data/inside_control.json (every ablation row and every share).
"""

import json

from common import *
from inside_load import load_results

names = list(MODELS)
results = {name: load_results(name) for name in names}
get_row = lambda name, row: 100 * results[name]["rows"][row]["accuracy"] if row in results[name]["rows"] else float("nan")
own_step = lambda name: "MLP dropped" if "mlp" in name else "λ = 1"

table = {}
for name in names:
    rows, shares = results[name]["rows"], results[name]["shares"]
    table[name] = {"label": MODELS[name], "accuracy": {row: round(100 * score["accuracy"], 2) for row, score in rows.items()}, "share of z12 (%)": {key: round(100 * value, 2) for key, value in shares.items()}}
    table[name]["agreement with real CLIP"] = {row: round(100 * score["agreement"], 2) for row, score in rows.items()}
(DATA / "inside_control.json").write_text(json.dumps(table, indent=1, ensure_ascii=False))

trained = torch.tensor([get_row(name, "as trained") for name in names])
no_control = trained - torch.tensor([get_row(name, "control dropped") for name in names])
no_step = trained - torch.tensor([get_row(name, own_step(name)) for name in names])
share = torch.tensor([100 * results[name]["shares"]["control"] if "control dropped" in results[name]["rows"] else float("nan") for name in names])
for name, *numbers in zip(names, share, trained, no_control, no_step):
    print(f"{MODELS[name]:45s} control {numbers[0]:6.2f}% of z12 | as trained {numbers[1]:.2f} | control removed -{numbers[2]:.2f} | own step removed -{numbers[3]:.2f}")

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6), sharey=True, width_ratios=[1, 1.7])
rows = torch.arange(len(names))
axes[0].barh(rows, share, height=0.5, color=BLUE)
for row in rows[share.isnan()]:
    axes[0].text(1, row, "no control", va="center", fontsize=9, color=INK)
for row in rows[share < 5]:
    axes[0].text(share[row] + 2, row, f"{share[row]:.1f}%", va="center", fontsize=9, color=INK)
axes[0].set_xlabel("the control's share of z12 (% of its squared size)")
axes[0].set_xlim(0, 100)

axes[1].barh(rows, trained, height=0.78, color=GRID, label="the model's accuracy as trained (the most it can lose)")
axes[1].barh(rows - 0.18, no_control, height=0.32, color=BLUE, label="lost when the control is removed")
axes[1].barh(rows + 0.18, no_step, height=0.32, color=ORANGE, label="lost when z6 passes through unchanged (every λ = 1, or no MLP)")
axes[1].set_xlabel("points of accuracy")
axes[1].legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=1, fontsize=9, handlelength=1.2, labelcolor=INK)
axes[0].set_yticks(rows, [MODELS[name] for name in names])
axes[0].invert_yaxis()
axes[0].tick_params(axis="y", colors=INK, length=0)
for ax in axes:
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
fig.suptitle("A model with a control loses everything without it, even where the control is under 3% of z12.\nWithout the penalty, its eigenvalues can be set to 1 at a cost of 0.3 to 5.8 points", x=0.01, ha="left", color=INK, fontsize=12)
fig.tight_layout()
save(fig, "inside_control")
