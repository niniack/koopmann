import json, os
os.chdir("/home/nsa325/work/koopmann/second_analysis")
nb = json.load(open("21_imagenet_features.ipynb"))
cells = ["".join(c["source"]) for c in nb["cells"]]
exec(cells[1], globals())
exec(cells[3], globals())
exec(cells[5].split("bundles = {name")[0], globals())
exec(cells[11], globals())
exec(cells[16], globals())

MODE = 414
kae = load_model("im_jordan128", 128, True)
eigenvalues, encode, decode = get_jordan_modes(kae)
train_modes = encode(train_x[:, 0])
mean = train_modes[:, MODE].mean()
active = train_modes[:, MODE][train_modes[:, MODE].abs() > 0]

value = eigenvalues[MODE]
print(f"Latent coordinates of mode {MODE}: {2 * MODE - 1} and {2 * MODE} of {eigenvalues.numel() * 2 - 1}")
print(f"Eigenvalue: {value.real:.4f} + {value.imag:.4f}i, size {value.abs():.4f}")
print(f"All eigenvalue sizes: median {eigenvalues.abs().median():.4f}, min {eigenvalues.abs().min():.4f}, max {eigenvalues.abs().max():.4f}")
print(f"Coefficient size: training mean {mean.abs():.4f}, typical when on {active.abs().median():.4f}, on for {100 * len(active) / len(train_modes):.1f}% of training images")

modes = encode(x[:, 0])
base = decode(modes * eigenvalues)

to_mean = modes.clone()
to_mean[:, MODE] = mean
edited_coefficient = decode(to_mean * eigenvalues)

zeroed = eigenvalues.clone()
zeroed[MODE] = 0
edited_eigenvalue = decode(modes * zeroed)

rows = {
    "coefficient set to mean": measure(base, edited_coefficient, y, is_snow, ~is_snow, snow_direction),
    "eigenvalue set to 0": measure(base, edited_eigenvalue, y, is_snow, ~is_snow, snow_direction),
}
print(pd.DataFrame(rows).T.round(3).to_string())
difference = (edited_coefficient.double() - edited_eigenvalue.double()).norm(dim=1) / (edited_coefficient.double() - base.double()).norm(dim=1).clamp_min(1e-9)
print(f"Difference between the two edits, relative to edit size on snowy images: {difference[is_snow].median():.3f}")
