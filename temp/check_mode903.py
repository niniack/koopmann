import json, os
os.chdir("/home/nsa325/work/koopmann/second_analysis")
nb = json.load(open("24_topk_modes.ipynb"))
cells = ["".join(c["source"]) for c in nb["cells"]]
exec(cells[1], globals())
exec(cells[3], globals())
exec(cells[5].split("bundles = {name")[0], globals())
exec(cells[11], globals())
exec(cells[16], globals())

head = torch.load("/home/nsa325/work/koopmann/temp/zero_shot_head.pt")
zero_shot = lambda states: (to_embeddings(states) @ head).argmax(dim=1)

for name, top_k, mode in [("im_jordan256_2k_both", 256, 903), ("im_jordan256_2k_both", 256, 878)]:
    kae = load_model(name, top_k, True, 2048)
    eigenvalues, encode, decode = get_jordan_modes(kae)
    train_modes = encode(train_x[:, 0])
    mean = train_modes[:, mode].mean()
    active = train_modes[:, mode][train_modes[:, mode].abs() > 0]
    value = eigenvalues[mode]
    print(f"\n{name} mode {mode}: eigenvalue {value.real:.3f} + {value.imag:.3f}i, size {value.abs():.3f} (median of all {eigenvalues.abs().median():.3f})")
    print(f"Coefficient: training mean {mean.abs():.4f}, typical when on {active.abs().median():.4f}, on for {100 * len(active) / len(train_modes):.1f}% of training images")

    modes = encode(x[:, 0])
    base = decode(modes * eigenvalues)
    to_mean = modes.clone()
    to_mean[:, mode] = mean
    zeroed = eigenvalues.clone()
    zeroed[mode] = 0
    edits = {"coefficient set to mean": decode(to_mean * eigenvalues), "eigenvalue set to 0": decode(modes * zeroed)}

    rows = {}
    correct = zero_shot(base) == y
    for label, edited in edits.items():
        fixed = zero_shot(edited) == y
        rows[label] = {
            **{k: v for k, v in measure(base, edited, y, is_snow, ~is_snow, snow_direction).items() if k in ["gap closed", "size ratio"]},
            "snowy before": correct[is_snow].double().mean().item(),
            "snowy after": fixed[is_snow].double().mean().item(),
            "normal before": correct[~is_snow].double().mean().item(),
            "normal after": fixed[~is_snow].double().mean().item(),
        }
    print(pd.DataFrame(rows).T.round(3).to_string())
    a, b = edits.values()
    difference = (a.double() - b.double()).norm(dim=1) / (a.double() - base.double()).norm(dim=1).clamp_min(1e-9)
    print(f"Difference between the two edits, relative to edit size on snowy images: {difference[is_snow].median():.3f}")
