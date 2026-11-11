from pathlib import Path

path = Path(__file__).parent / "make_dino_modes_notebook.py"
text = path.read_text()


def replace(old, new):
    global text
    assert text.count(old) == 1, (text.count(old), old[:60])
    text = text.replace(old, new)


replace("TopK 256 (129 modes per image)", "TopK 256 (128 modes per image)")
replace(
    "snow_z = to_complex(train.encode_pairs(kae, x))[:, 0]",
    "snow_pairs = to_complex(train.encode_pairs(kae, x))\nsnow_z, snow_z12 = snow_pairs[:, 0], snow_pairs[:, 1]",
)

GAP = '''md("""
##### Where the gap to DINOv3 opens

The KAE has three parts that can each lose accuracy: the encoder and decoder, the sharing of modes between the blocks, and K. To see which one does, we swap parts of the KAE's prediction for what DINOv3 really does at block 12:
- **block 12 through the KAE, its own modes**: encode block 12, decode. No K, no sharing. This is the cost of the encoder and decoder.
- **block 12 on the modes picked at block 6**: the same, but keeping the modes TopK picked at block 6. This is what a perfect step would give.
- **K's size, true turn** and **K's turn, true size**: K's prediction for every mode, with its angle or its size swapped for the true one at block 12.
- **true values for one group of modes**: K for every mode except one group, which takes its true block-12 value.

NOTE: Every row but the last uses block 12 itself, so these are ceilings. A step from block 6 cannot know everything about block 12, so the gap to "K as trained" is the most a better step could win back, not what it would.
""")

code("""
def mix(z_start, z_end, part):
    \\"\\"\\"K's prediction of every kept mode, with its size or its turn swapped for the true one at block 12.\\"\\"\\"

    predicted = eigenvalues * z_start
    if part == "size":
        mixed = z_end.abs() * torch.exp(1j * torch.angle(predicted))
    else:
        mixed = predicted.abs() * torch.exp(1j * torch.angle(z_end))
    return torch.where(z_start != 0, mixed, torch.zeros_like(mixed))


def get_scores(z_end):
    found = head.classify(decode(z_end))
    return {"probe accuracy": (found == val_y).double().mean().item(), "agrees with DINOv3": (found == true_answers).double().mean().item()}


def with_true(mask):
    \\"\\"\\"K's prediction, with the modes in `mask` taking their true block-12 value.\\"\\"\\"

    mixed = eigenvalues * z6
    mixed[:, mask] = z12[:, mask]
    return mixed


is_feature = torch.tensor(group == "feature")
strong, weak = is_feature & (size > 5), is_feature & (size <= 5)
rows = {
    "DINOv3's block 12": {"probe accuracy": (true_answers == val_y).double().mean().item(), "agrees with DINOv3": 1.0},
    "block 12 through the KAE, its own modes": get_scores(to_complex(torch.cat([kae.encode(batch) for batch in val_x[:, 1].split(5_000)]))),
    "block 12 on the modes picked at block 6": get_scores(z12),
    "K's size, true turn": get_scores(mix(z6, z12, "turn")),
    "K's turn, true size": get_scores(mix(z6, z12, "size")),
    f"true values for the {int(strong.sum())} feature modes with |λ| > 5": get_scores(with_true(strong)),
    f"true values for the {int(weak.sum())} feature modes with |λ| ≤ 5": get_scores(with_true(weak)),
    f"true values for the {int(always.sum())} always-on modes": get_scores(with_true(always)),
    "K as trained": scores,
}
pd.DataFrame(rows).T.round(3)
""")

'''
replace('md("""\n##### Snow: where the KAE fails and DINOv3 does not', GAP + 'md("""\n##### Snow: where the KAE fails and DINOv3 does not')

WHY = '''md("""
##### Is K wrong about the snow modes?

Removing the snow modes fixes snowy images. That would be a repair if K overdid snow on these images. We check it the same way as above, on the snow test: the top 4 snow modes take their true block-12 value while K handles the rest, and the other way round. Then how the snow modes really move from block 6 to block 12 on the snowy vehicles, against the snowplows and snowmobiles of validation.
""")

code("""
def get_snow_scores(z_end):
    right = head.classify(decode(z_end)) == y
    return {"accuracy, snowy": right[is_snowy].double().mean().item(), "accuracy, normal": right[~is_snowy].double().mean().item()}


chosen = SNOW_MODES[:4]
predicted = eigenvalues * snow_z
only_snow, all_but_snow = predicted.clone(), snow_z12.clone()
only_snow[:, chosen] = snow_z12[:, chosen]
all_but_snow[:, chosen] = predicted[:, chosen]
display(
    pd.DataFrame(
        {
            "K as trained": get_snow_scores(predicted),
            "true values for the top 4 snow modes, K for the rest": get_snow_scores(only_snow),
            "K for the top 4 snow modes, true values for the rest": get_snow_scores(all_but_snow),
            "true values for every mode (no K)": get_snow_scores(snow_z12),
            "K's size, true turn, every mode": get_snow_scores(mix(snow_z, snow_z12, "turn")),
            "K's turn, true size, every mode": get_snow_scores(mix(snow_z, snow_z12, "size")),
        }
    ).T.round(3)
)

assert (class_names[802], class_names[803]) == ("snowmobile", "snowplow")
snow_classes = (val_y == 802) | (val_y == 803)
image_groups = {
    "snowy vehicles": (snow_z[is_snowy], snow_z12[is_snowy]),
    "snowplows and snowmobiles in validation": (z6[snow_classes], z12[snow_classes]),
    "all validation": (z6, z12),
}
rows = []
for mode in chosen:
    for label, (start, end) in image_groups.items():
        on = start[:, mode] != 0
        a, b = start[on, mode], end[on, mode]
        pooled = (b * a.conj()).sum()
        rows.append(
            {
                "mode": mode,
                "images": label,
                "keep the mode": int(on.sum()),
                "|λ|": size[mode].item(),
                "angle": angle[mode].item(),
                "size at block 6": a.abs().median().item(),
                "measured gain": (b.abs() / a.abs()).median().item(),
                "measured turn": math.degrees(torch.angle(pooled).item()),
            }
        )
pd.DataFrame(rows).set_index(["mode", "images"]).round(2)
""")

'''
replace('md("""\n##### Every mode: coherence, and what removing it does', WHY + 'md("""\n##### Every mode: coherence, and what removing it does')
path.write_text(text)
print("patched")
