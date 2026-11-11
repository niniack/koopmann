import json, os, sys, torch
os.chdir("/home/nsa325/work/koopmann/second_analysis")
nb = json.load(open("jordan_topk_modes.ipynb"))
cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
for i in [0, 1, 2]:
    exec(cells[i], globals())
exec(next(c for c in cells if "snowy_imagenet" in c).split("true_correct =")[0], globals())
logits = lambda s: embed(s) @ head
removed = dampen(SNOW_MODE, 0.0)
def ranks(start, truth):
    before, after = logits(predict(kae, start)), logits(predict(removed, start))
    old, new = before.argmax(-1), after.argmax(-1)
    changed = old != new
    # rank of the new answer among the old scores: 1 is the old answer, 2 the runner-up
    rank = (before > before.gather(1, new[:, None])).sum(-1) + 1
    return changed, rank, old, new
changed, rank, old, new = ranks(x[:, 0], y)
fixed = changed & (new == y)
print("snow test: answers changed", int(changed.sum()), "| rank of new answer in old scores:", torch.bincount(rank[changed]).tolist(), "| fixed:", int(fixed.sum()), "ranks of fixed:", rank[fixed].tolist())
on = activity[:, SNOW_MODE] > 0
changed, rank, old, new = ranks(val_x[on, 0], val_y[on])
r = rank[changed]
print("all validation where snow mode on: answers changed", int(changed.sum()), "of", int(on.sum()), "| new answer was runner-up:", f"{(r == 2).double().mean():.2f}", "| top 3:", f"{(r <= 3).double().mean():.2f}", "| top 5:", f"{(r <= 5).double().mean():.2f}")
