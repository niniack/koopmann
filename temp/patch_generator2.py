from pathlib import Path

path = Path(__file__).parent / "make_dino_modes_notebook.py"
text = path.read_text()


def replace(old, new):
    global text
    assert text.count(old) == 1, (text.count(old), old[:70])
    text = text.replace(old, new)


# 1. Which images are never shown
replace(
    'head = train.DinoHead(ENCODER, mean, std, torch.device("cpu"), zero_shot=True)\n',
    'head = train.DinoHead(ENCODER, mean, std, torch.device("cpu"), zero_shot=True)\n'
    "\n"
    "# Images we never show: insects, arachnids and their relatives, honeycombs (bees) and spider webs,\n"
    "# by label or by any of DINOv3's top 5 answers, as images of other classes can contain them too\n"
    "HIDDEN_CLASSES = torch.tensor([*range(69, 80), 126, *range(300, 327), 599, 815])\n"
    "hidden = torch.isin(val_y, HIDDEN_CLASSES) | torch.isin(head.get_log_probs(val_x[:, 1]).topk(5).indices, HIDDEN_CLASSES).any(dim=1)\n",
)
replace(
    "We write each mode's two numbers as one complex number: `z₆` at block 6 and `z₁₂` at block 12. K then says `z₁₂ ≈ λ·z₆`, mode by mode.\n",
    "We write each mode's two numbers as one complex number: `z₆` at block 6 and `z₁₂` at block 12. K then says `z₁₂ ≈ λ·z₆`, mode by mode.\n"
    "\n"
    "NOTE: No gallery in this notebook shows insects or arachnids. Images of those classes are skipped, along with honeycombs, spider webs and any image for which one of those classes is among DINOv3's top 5 answers. Modes that mostly fire on such images are left out of the galleries. They still count in every number.\n",
)

# 2. Galleries skip hidden images
replace(
    "        for ax, index in zip(row, z6[:, mode].abs().topk(8).indices.tolist()):\n",
    "        order = z6[:, mode].abs().argsort(descending=True)\n"
    "        for ax, index in zip(row, order[~hidden[order]][:8].tolist()):\n",
)
replace(
    '            "classes in top 16": len(set(val_y[top].tolist())),\n',
    '            "classes in top 16": len(set(val_y[top].tolist())),\n            "hidden in top 16": int(hidden[top].sum()),\n',
)

# 3. Modes that mostly fire on hidden images are left out of the galleries
replace(
    'cross_class = table[(table["classes in top 16"] >= 9) & feature].sort_values("coherence", ascending=False)\n',
    'showable = table["hidden in top 16"] < 4\n'
    'print(f"{int((~showable).sum())} modes mostly fire on hidden images and are left out of the galleries")\n'
    'cross_class = table[(table["classes in top 16"] >= 9) & feature & showable].sort_values("coherence", ascending=False)\n',
)
replace("candidates = table.index[feature].to_numpy()\n", "candidates = table.index[feature & showable].to_numpy()\n")
replace('needed = table[feature].sort_values("net").index[:6].tolist()\n', 'needed = table[feature & showable].sort_values("net").index[:6].tolist()\n')
path.write_text(text)

findings = Path(__file__).parent / "dino_findings.md"
text = findings.read_text()
old = "round fruit and balls, shaggy dogs, birds against the sky, antelope herds, knitted fabric, spiders."
assert text.count(old) == 1
findings.write_text(text.replace(old, "round fruit and balls, shaggy dogs, birds against the sky, antelope herds, knitted fabric."))
print("patched")
