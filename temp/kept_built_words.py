"""
Is the part of a mode that K builds by turning (the "built part") about objects, and the part it keeps about looks?

Reads the pushes saved by name_modes.py: for each mode, the mean change of CLIP's final image embedding made by the whole
mode, by its kept part Re(λ)·z6 and by its built part i·Im(λ)·z6. Each push is scored against ImageNet's 1,000 class text
vectors. A push that is about an object lands on a few classes; a push that is about a look (snow, stringy, dark) is spread
over many. "Class focus" of a push = share of its squared class scores (centred over classes) held by its 5 largest classes.
With 1,000 classes, a push with no focus scores about 0.02.

Usage: kept_built_words.py <run name>
"""

import sys

import torch

name = sys.argv[1]
saved = torch.load(f"names_{name}.pt", weights_only=False)
table = saved["table"]
classes = torch.load("zero_shot_head.pt").float().cpu()
classes = torch.nn.functional.normalize(classes - classes.mean(dim=1, keepdim=True), dim=0)
column = lambda key: torch.tensor([float(row[key]) for row in table])
angle, size, usage = column("angle of λ").abs(), column("size of λ"), column("usage")
features = usage < 0.99


def get_focus(pushes):
    scores = torch.nn.functional.normalize(torch.stack(list(pushes)) if not torch.is_tensor(pushes) else pushes, dim=-1) @ classes
    scores = scores - scores.mean(dim=1, keepdim=True)
    squares = scores.pow(2)
    return squares.topk(5, dim=1).values.sum(dim=1) / squares.sum(dim=1), scores.max(dim=1).values


focus, peak = {}, {}
for key in ["mode", "kept", "built", "top images"]:
    focus[key], peak[key] = get_focus(saved[key])
rank = lambda a: a.argsort().argsort().float()
correlate = lambda a, b: torch.corrcoef(torch.stack([a.float(), b.float()]))[0, 1].item()


def partial(a, b, controls):
    design = torch.stack([torch.ones(len(a))] + [rank(c) for c in controls], dim=1)
    left = lambda v: rank(v) - design @ torch.linalg.lstsq(design, rank(v)[:, None]).solution[:, 0]
    return correlate(left(a), left(b))


quantiles = torch.tensor([0.1, 0.5, 0.9])
print(f"== {name}: {int(features.sum())} feature modes")
print("class focus of a push, 10 / 50 / 90% of modes")
for key in focus:
    print(f"  {key:11s}", " / ".join(f"{v:.3f}" for v in focus[key][features].quantile(quantiles).tolist()))
turning = features & (angle > 5)
gap = focus["built"] - focus["kept"]
print(f"modes that turn by over 5°: {int(turning.sum())}; built focus minus kept focus, 10 / 50 / 90%:", " / ".join(f"{v:+.3f}" for v in gap[turning].quantile(quantiles).tolist()), f"; built higher in {(gap[turning] > 0).float().mean():.0%}")
match = lambda key: column(f"{key} matches top images")
print(f"  match with the mode's top images (cosine), median: kept {match('kept')[turning].median():.3f}, built {match('built')[turning].median():.3f}; built against kept {column('built against kept')[turning].median():.3f}")
print("by |angle of λ|: modes | kept focus | built focus | built higher | kept matches top images | built matches top images | built size over kept size")
for low, high in [(0, 5), (5, 10), (10, 20), (20, 45), (45, 180)]:
    rows = features & (angle >= low) & (angle < high)
    if rows.sum() == 0:
        continue
    ratio = column("built size")[rows] / column("kept size")[rows].clamp(min=1e-6)
    print(f"  {low:3d}-{high:3d}°: {int(rows.sum()):4d} | {focus['kept'][rows].median():.3f} | {focus['built'][rows].median():.3f} | {(gap[rows] > 0).float().mean():.0%} | {match('kept')[rows].median():.3f} | {match('built')[rows].median():.3f} | {ratio.median():.2f}")
print("rank correlations over feature modes (then with size of λ and usage held fixed):")
for label, values in {"class focus of the whole push": focus["mode"], "built focus minus kept focus": gap, "built matches top images": match("built"), "kept matches top images": match("kept")}.items():
    print(f"  |angle| with {label:32s} {correlate(rank(angle[features]), rank(values[features])):6.3f} ({partial(angle[features], values[features], [size[features], usage[features]]):6.3f}); size of λ with it {correlate(rank(size[features]), rank(values[features])):6.3f} ({partial(size[features], values[features], [angle[features], usage[features]]):6.3f})")
print("\nthe 12 modes whose built part is most focused, among modes that turn by over 10°: mode | angle | kept words | built words | classes of top images")
order = (focus["built"] * (features & (angle > 10))).argsort(descending=True)[:12]
for i in order.tolist():
    row = table[i]
    print(f"  {row['mode']:5d} | {row['angle of λ']:6.1f}° | {row['words kept']} | {row['words built']} | {row['classes of top images']}")
print("done")
