# Extra sections for make_dino_modes_notebook.py: the always-on modes, and DINOv3's own mistakes.
# Executed by the generator, so `md` and `code` are its cell helpers. Raw strings: the cell source is written as it should appear.

md(r"""
##### The modes every image keeps

16 modes are on for every image: 2 growth, 10 offset and 4 inert. The table says how fast each grows and how far it turns, ranked among the modes that are ever kept (rank 1 is the largest).

- **Size spread**: the spread of the mode's block-6 size over images, divided by its average. Near 0 means every image has the same amount.
- **Phase spread**: a mode's block-6 value is two numbers, so a point on a plane. This is how much the direction of that point varies over images. Near 0° means every image points the same way. Above 100° means images point all around the plane.
""")

code(r"""
def get_phase_spread(mode):
    on = kept[:, mode]
    together = torch.exp(1j * torch.angle(z6[on, mode])).mean().abs().item()
    return math.degrees(math.sqrt(max(-2 * math.log(together), 0)))


def get_size_spread(mode):
    sizes = z6[kept[:, mode], mode].abs()
    return (sizes.std() / sizes.mean()).item()


ever_kept = ~never
rows = []
for name in ["growth", "offset", "inert"]:
    for mode in np.flatnonzero(group == name).tolist():
        rows.append(
            {
                "mode": mode,
                "group": name,
                "|λ|": size[mode].item(),
                "|λ| rank": int((size[ever_kept] > size[mode]).sum()) + 1,
                "|angle|°": angle[mode].abs().item(),
                "angle rank": int((angle[ever_kept].abs() > angle[mode].abs()).sum()) + 1,
                "size spread": get_size_spread(mode),
                "phase spread°": get_phase_spread(mode),
            }
        )
always_on = pd.DataFrame(rows).set_index("mode")
feature_modes = np.flatnonzero(group == "feature").tolist()
often = [mode for mode in feature_modes if usage[mode] >= 0.005]
print(f"{int(ever_kept.sum())} modes are ever kept. Feature modes: median |λ| {size[feature_modes].median():.2f}, median |angle| {angle[feature_modes].abs().median():.1f}°,")
print(f"median size spread {np.median([get_size_spread(mode) for mode in often]):.2f}, median phase spread {np.median([get_phase_spread(mode) for mode in often]):.1f}° (modes kept on at least 0.5% of images)")
always_on.round(2)
""")

md(r"""
##### Why the offset modes turn

Both blocks go through one shared standardization, so the average block-6 state and the average block-12 state sit on opposite sides of zero, at the same distance. Around those averages, block-12 states are spread about ten times wider than block-6 states. So from block 6 to block 12 the average has to flip its sign and keep its length, while the part that differs between images has to grow about tenfold.
""")

code(r"""
state6, state12 = val_x[:, 0], val_x[:, 1]
mean_state6, mean_state12 = state6.mean(dim=0), state12.mean(dim=0)
spread6 = (state6 - mean_state6).pow(2).sum(dim=1).mean().sqrt().item()
spread12 = (state12 - mean_state12).pow(2).sum(dim=1).mean().sqrt().item()
print(f"cosine between the two average states: {torch.cosine_similarity(mean_state6, mean_state12, dim=0).item():.3f}")
print(f"spread at block 12 over spread at block 6: {spread12 / spread6:.2f}")
pd.DataFrame(
    {
        "length of the average state": [mean_state6.norm().item(), mean_state12.norm().item()],
        "spread of states around it": [spread6, spread12],
        "share of a state's energy that is the average": [(mean_state6.pow(2).sum() / state6.pow(2).sum(dim=1).mean()).item(), (mean_state12.pow(2).sum() / state12.pow(2).sum(dim=1).mean()).item()],
    },
    index=["block 6", "block 12"],
).round(2)
""")

md(r"""
Decoding with only some of the modes, the rest set to zero. **Error left** is the squared error over the squared spread of states around their block's average, so 1 means "no better than the average state". The cosine compares the average output with the block's average state.
""")

code(r"""
growth_modes, offset_modes, inert_modes = (np.flatnonzero(group == name).tolist() for name in ["growth", "offset", "inert"])
subsets = {
    "no modes": [],
    "10 offset": offset_modes,
    "2 growth": growth_modes,
    "feature modes only": feature_modes,
    "feature + growth": feature_modes + growth_modes,
    "feature + offset": feature_modes + offset_modes,
    "all modes": list(range(len(eigenvalues))),
}
rows = {}
for label, chosen in subsets.items():
    mask = torch.zeros(len(eigenvalues), dtype=torch.bool)
    mask[chosen] = True
    start, end = decode(z6 * mask), decode(eigenvalues * z6 * mask)
    rows[label] = {
        "block 6: error left": ((start - state6).pow(2).sum() / (state6 - mean_state6).pow(2).sum()).item(),
        "block 6: cosine with the average state": torch.cosine_similarity(start.mean(dim=0), mean_state6, dim=0).item(),
        "block 12: error left": ((end - state12).pow(2).sum() / (state12 - mean_state12).pow(2).sum()).item(),
        "block 12: cosine with the average state": torch.cosine_similarity(end.mean(dim=0), mean_state12, dim=0).item(),
        "probe accuracy": (head.classify(end) == val_y).double().mean().item(),
    }
pd.DataFrame(rows).T.round(3)
""")

md(r"""
Are they constants? Each row changes one group of always-on modes in K's prediction and reports probe accuracy.

- **Same value for every image**: each mode takes its average value over images.
- **Angle set to 0**: the mode keeps its size at block 12 and loses its turn.
- For the growth modes, **average size, own direction** keeps where each image points on the mode's plane and gives every image the same size. **Own size, average direction** does the reverse.
""")

code(r"""
prediction = eigenvalues * z6


def get_accuracy(z_end):
    return (head.classify(decode(z_end)) == val_y).double().mean().item()


rows = {"K as trained": {"probe accuracy": get_accuracy(prediction)}}
for label, chosen in [("10 offset modes", offset_modes), ("2 growth modes", growth_modes), ("4 inert modes", inert_modes)]:
    same, removed, straight = prediction.clone(), prediction.clone(), prediction.clone()
    same[:, chosen] = (eigenvalues[chosen] * z6[:, chosen].mean(dim=0)).expand(len(z6), -1)
    removed[:, chosen] = 0
    straight[:, chosen] = size[chosen] * z6[:, chosen]
    rows[f"{label}: same value for every image"] = {"probe accuracy": get_accuracy(same)}
    rows[f"{label}: removed"] = {"probe accuracy": get_accuracy(removed)}
    rows[f"{label}: angle set to 0"] = {"probe accuracy": get_accuracy(straight)}
for mode in growth_modes:
    own_direction, own_size = prediction.clone(), prediction.clone()
    own_direction[:, mode] = eigenvalues[mode] * z6[:, mode].abs().mean() * torch.exp(1j * torch.angle(z6[:, mode]))
    own_size[:, mode] = eigenvalues[mode] * z6[:, mode].abs() * torch.exp(1j * torch.angle(z6[:, mode].mean()))
    rows[f"mode {mode}: average size, own direction"] = {"probe accuracy": get_accuracy(own_direction)}
    rows[f"mode {mode}: own size, average direction"] = {"probe accuracy": get_accuracy(own_size)}
pd.DataFrame(rows).T.round(4)
""")

md(r"""
##### What the growth modes hold

Each growth mode is a point on a plane that every image has. First, how well its two numbers are predicted by the main directions along which block-6 states differ from each other (their principal directions). Then, for each 45° slice of the plane, the classes whose images sit there most (the share of the class's total size that falls in the slice), and the images with the largest size in that slice.
""")

code(r"""
centered6 = state6 - mean_state6
_, _, directions6 = torch.pca_lowrank(centered6, q=20, center=False, niter=4)
projected6 = centered6 @ directions6


def get_r2(target, design):
    design = torch.cat([design, torch.ones(len(design), 1)], dim=1)
    fit = design @ torch.linalg.lstsq(design, target).solution
    return (1 - (fit - target).pow(2).sum() / (target - target.mean(dim=0)).pow(2).sum()).item()


def get_class_share(mode):
    # The share of the spread of the mode's value over images that is between classes
    by_class = torch.zeros(1000, dtype=torch.cfloat).index_add_(0, val_y, z6[:, mode]) / 50
    within = (z6[:, mode] - by_class[val_y]).abs().pow(2).mean()
    return (1 - within / (z6[:, mode] - z6[:, mode].mean()).abs().pow(2).mean()).item()


rows = {}
for mode in growth_modes:
    value = torch.view_as_real(z6[:, mode].contiguous())
    rows[f"mode {mode}"] = {
        "R² from the top block-6 direction": get_r2(value, projected6[:, :1]),
        "from the top 5": get_r2(value, projected6[:, :5]),
        "from the top 20": get_r2(value, projected6[:, :20]),
        "share of its spread that is between classes": get_class_share(mode),
    }
print(f"feature modes: median share of spread between classes {np.median([get_class_share(mode) for mode in often]):.3f}")
pd.DataFrame(rows).T.round(3)
""")

code(r"""
SLICE = 45
never_shown = set(HIDDEN_CLASSES.tolist())


def show_directions(mode):
    phase, sizes = torch.rad2deg(torch.angle(z6[:, mode])), z6[:, mode].abs()
    total = torch.zeros(1000).index_add_(0, val_y, sizes)
    lows = list(range(-180, 180, SLICE))
    fig, axes = plt.subplots(len(lows), 8, figsize=(16, 2.2 * len(lows)), squeeze=False)
    rows = []
    for row, low in zip(axes, lows):
        inside = (phase >= low) & (phase < low + SLICE)
        lift = torch.zeros(1000).index_add_(0, val_y[inside], sizes[inside]) / total
        top_classes = [c for c in lift.argsort(descending=True).tolist() if c not in never_shown][:6]
        rows.append({"direction": f"{low:+d}° to {low + SLICE:+d}°", "share of images": inside.double().mean().item(), "classes that sit there most": ", ".join(class_names[c] for c in top_classes)})
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
        shown = (inside & ~hidden).nonzero()[:, 0]
        for ax, index in zip(row, shown[sizes[shown].argsort(descending=True)][:8].tolist()):
            image = images[index]["image"].convert("RGB")
            image.thumbnail((256, 256))
            ax.imshow(np.asarray(image))
            ax.set_xlabel(class_names[val_y[index]][:22], fontsize=8, color=INK)
        row[0].set_ylabel(f"mode {mode}\n{low:+d}° to {low + SLICE:+d}°\n{inside.double().mean().item():.0%} of images", fontsize=9, color=INK)
    fig.tight_layout()
    plt.show()
    return pd.DataFrame(rows).set_index("direction").round(2)


with pd.option_context("display.max_colwidth", 120):
    display(show_directions(growth_modes[0]))
""")

code(r"""
with pd.option_context("display.max_colwidth", 120):
    display(show_directions(growth_modes[1]))
""")

md(r"""
##### DINOv3's own mistakes

Snow does not fool DINOv3, so where does it go wrong on ImageNet's validation images? First its wrong answers and the pairs of classes it mixes up most. The class number is shown as some classes share a name.
""")

code(r"""
log_probs = head.get_log_probs(val_x[:, 1])
dino_wrong = true_answers != val_y
top5 = log_probs.topk(5).indices
in_top5 = (top5 == val_y[:, None]).any(dim=1)
print(f"DINOv3: accuracy {1 - dino_wrong.double().mean().item():.4f}, top-5 accuracy {in_top5.double().mean().item():.4f}, {int(dino_wrong.sum())} wrong answers")
print(f"of the wrong answers, the label is DINOv3's second choice for {(top5[:, 1] == val_y)[dino_wrong].double().mean().item():.3f} and among its top 5 for {in_top5[dino_wrong].double().mean().item():.3f}")

confusion = torch.zeros(1000, 1000, dtype=torch.long)
confusion.index_put_((val_y[dino_wrong], true_answers[dino_wrong]), torch.ones(int(dino_wrong.sum()), dtype=torch.long), accumulate=True)
both_ways = confusion + confusion.T
print(f"wrong answers inside a pair of classes mixed up at least 10 times: {(confusion[both_ways >= 10].sum() / dino_wrong.sum()).item():.3f}, over {int((both_ways >= 10).sum()) // 2} pairs")
print(f"KAE on the images DINOv3 gets right: {(answers == val_y)[~dino_wrong].double().mean().item():.4f}; on the images it gets wrong: {(answers == val_y)[dino_wrong].double().mean().item():.4f}")
print(f"on DINOv3's wrong answers the KAE gives the label {int((answers == val_y)[dino_wrong].sum())} times, DINOv3's wrong answer {int((answers == true_answers)[dino_wrong].sum())} times and a third answer {int(((answers != val_y) & (answers != true_answers))[dino_wrong].sum())} times")
rows = []
for index in confusion.flatten().argsort(descending=True).tolist():
    label, answer = divmod(index, 1000)
    if label in never_shown or answer in never_shown:
        continue
    rows.append({"label": f"{class_names[label]} ({label})", "DINOv3's answer": f"{class_names[answer]} ({answer})", "count, of 50": int(confusion[label, answer]), "the other way round": int(confusion[answer, label])})
    if len(rows) == 15:
        break
pd.DataFrame(rows)
""")

md(r"""
##### Modes that flag DINOv3's mistakes

Does any mode mark the images DINOv3 gets wrong? Some classes are simply harder, so each mode is scored against its images' classes: we count DINOv3's wrong answers among the images that keep the mode, and compare with the count expected if the same number of images had been drawn from each class at random. The **score** is that excess in standard deviations. Halves A and B split each class's images in two, to see whether a score repeats.
""")

code(r"""
flag_modes = torch.tensor(often)
one_hot = torch.zeros(len(val_y), 1000).scatter_(1, val_y[:, None], 1.0)
position = torch.zeros(len(val_y), dtype=torch.long)
for c in range(1000):
    members = (val_y == c).nonzero()[:, 0]
    position[members] = torch.arange(len(members))
in_a = position % 2 == 0


def get_excess(chosen, per_class):
    on, bad = kept[chosen][:, flag_modes].float(), dino_wrong[chosen].float()
    count = one_hot[chosen].T @ on
    rate = (one_hot[chosen].T @ bad) / per_class
    seen, expected = (on * bad[:, None]).sum(dim=0), (count * rate[:, None]).sum(dim=0)
    variance = (count * (rate * (1 - rate))[:, None] * (per_class - count) / (per_class - 1)).sum(dim=0)
    return seen, expected, (seen - expected) / variance.sqrt()


seen, expected, score = get_excess(torch.arange(len(val_y)), 50)
score_a, score_b = get_excess(in_a.nonzero()[:, 0], 25)[2], get_excess((~in_a).nonzero()[:, 0], 25)[2]
flags = pd.DataFrame(
    {
        "kept": usage[flag_modes].numpy(),
        "|λ|": size[flag_modes].double().numpy(),
        "angle": angle[flag_modes].double().numpy(),
        "DINOv3 wrong": seen.long().numpy(),
        "expected": expected.double().numpy(),
        "score": score.double().numpy(),
        "score, half A": score_a.double().numpy(),
        "score, half B": score_b.double().numpy(),
    },
    index=pd.Index(often, name="mode"),
)
print(f"{len(often)} modes scored. By chance, a score above 3.7 is expected for less than one of them.")
print(f"modes above 3.7: {int((score > 3.7).sum())}; below -3.7: {int((score < -3.7).sum())}; half A against half B, correlation over modes: {np.corrcoef(score_a.numpy(), score_b.numpy())[0, 1]:.2f}")
flags.sort_values("score", ascending=False).head(10).round(2)
""")

md(r"""
##### Does editing a flagged mode fix the mistakes?

For the 8 modes with the highest scores, on the images that keep the mode: answers fixed / broken by an edit. **Through K** edits the KAE's own prediction. **True block 12** edits the mode in DINOv3's real block-12 state, encoded on the modes picked at block 6, which is the best any K could do. No edit of mode 549 changes any answer.
""")

code(r"""
FLAG_EDITS = {"removed": 0, "+90°": turn(90), "180°": turn(180)}
rows = []
for mode in flags.sort_values("score", ascending=False).index[:8].tolist():
    on = kept[:, mode].nonzero()[:, 0]
    for source, latent in [("through K", prediction[on]), ("true block 12", z12[on])]:
        before = head.classify(decode(latent)) == val_y[on]
        row = {"mode": mode, "edit of": source, "images": len(on), "DINOv3 wrong": int(dino_wrong[on].sum()), "right before": int(before.sum())}
        for label, factor in FLAG_EDITS.items():
            edited = latent.clone()
            edited[:, mode] = edited[:, mode] * factor
            after = head.classify(decode(edited)) == val_y[on]
            row[label] = f"{int((~before & after).sum())} / {int((before & ~after).sum())}"
        rows.append(row)
pd.DataFrame(rows).set_index(["mode", "edit of"])
""")

md(r"""
Held out: take the 30 modes with the highest score on half A, keep those whose removal fixes more than it breaks on half A, remove them all together and score on half B.
""")

code(r"""
candidates = flags.sort_values("score, half A", ascending=False).index[:30].tolist()
rows = {}
for source, latent in [("through K", prediction), ("true block 12", z12)]:
    before = head.classify(decode(latent)) == val_y
    helpful = []
    for mode in candidates:
        on = kept[:, mode].nonzero()[:, 0]
        edited = latent[on].clone()
        edited[:, mode] = 0
        after = head.classify(decode(edited)) == val_y[on]
        gain = (after & ~before[on])[in_a[on]].sum() - (~after & before[on])[in_a[on]].sum()
        if gain > 0:
            helpful.append(mode)
    edited = latent.clone()
    edited[:, helpful] = 0
    after = head.classify(decode(edited)) == val_y
    rows[source] = {
        "modes removed": len(helpful),
        "half A before": before[in_a].double().mean().item(),
        "half A after": after[in_a].double().mean().item(),
        "half B before": before[~in_a].double().mean().item(),
        "half B after": after[~in_a].double().mean().item(),
        "half B fixed": int((after & ~before)[~in_a].sum()),
        "half B broken": int((~after & before)[~in_a].sum()),
    }
pd.DataFrame.from_dict(rows, orient="index").round(4)
""")
