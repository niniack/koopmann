import json

nb = json.load(open("second_analysis/13_editing.ipynb"))

for cell in nb["cells"]:
    source = "".join(cell["source"]).replace("b6_30v", "b6_novc")
    source = source.replace(
        "operator = kae.koopman_weights.T.double()",
        "# NOTE: On CUDA, `torch.linalg.eig` overwrites transposed inputs, so this must be contiguous\n"
        "operator = kae.koopman_weights.T.double().contiguous()",
    )
    cell["source"] = source.splitlines(keepends=True)
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

nb["cells"][0]["source"] = [
    "### Does editing K do something targeted, without variance and covariance terms?\n",
    "\n",
    "Same test as `13_editing`, on `b6_novc`. That run is `b6_30v` with `lambda_variance=0` and `lambda_covariance=0`. Nothing is trained here.\n",
    "\n",
    "**Specificity** is the target class's accuracy drop minus the mean drop of the other classes.\n",
]


def cell(kind, source):
    new = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        new.update({"outputs": [], "execution_count": None})
    return new


spectrum = '''def get_spectrum(values):
    """How many directions carry the operator, and the share held by the largest few."""

    shares = values.pow(2) / values.pow(2).sum()
    return {
        "effective rank": (-(shares * (shares + 1e-20).log()).sum()).exp().item(),
        "top 1": shares[:1].sum().item(),
        "top 5": shares[:5].sum().item(),
        "top 20": shares[:20].sum().item(),
        "top 100": shares[:100].sum().item(),
    }


old_kae, _ = KoopmanAutoencoder.load_model(file_path=save_dir / "b6_30v.safetensors")
old_values = torch.linalg.svdvals(old_kae.koopman_weights.double())

pd.DataFrame(
    {
        "K b6_30v": get_spectrum(old_values.to(device)),
        "K b6_novc": get_spectrum(k_values),
        "linear": get_spectrum(l_values),
    }
).T.round(3)
'''

nb["cells"] += [cell("markdown", "##### How concentrated is each operator?\n"), cell("code", spectrum)]
json.dump(nb, open("second_analysis/14_novc_editing.ipynb", "w"), indent=1)

ranks = '''def get_rank(x):
    """Effective rank of the covariance of `x`, the same measure as `latent/rank_*`."""

    values = torch.linalg.svdvals(x - x.mean(dim=0)).pow(2)
    shares = values / values.sum()
    return (-(shares * (shares + 1e-20).log()).sum()).exp().item()


old_kae = old_kae.to(device).eval()
old_phi = old_kae.encode(val_x[:, 0]).double()
old_operator = old_kae.koopman_weights.T.double()

pd.DataFrame(
    {
        "b6_30v": {
            "phi start": get_rank(old_phi),
            "phi end": get_rank(old_kae.encode(val_x[:, 1]).double()),
            "K applied to phi": get_rank(old_phi @ old_operator),
        },
        "b6_novc": {
            "phi start": get_rank(val_phi),
            "phi end": get_rank(kae.encode(val_x[:, 1]).double()),
            "K applied to phi": get_rank(val_phi @ operator),
        },
        "raw states": {
            "phi start": get_rank(val_x[:, 0].double()),
            "phi end": get_rank(val_x[:, 1].double()),
            "K applied to phi": get_rank(val_x[:, 0].double() @ linear_weight),
        },
    }
).round(1)
'''

modes = '''rows = {}
for c, name in enumerate(CLASSES):
    top = get_scores(eigen_coefficients, c).argmax()
    drops = kae_accs - get_class_accs(edit_kae_eigen(top[None]))

    # A complex mode has a conjugate partner with the same score
    pair = get_scores(eigen_coefficients, c).argsort(descending=True)[:2]
    is_complex = eigenvalues[top].imag.abs().item() > 1e-9
    pair_drops = kae_accs - get_class_accs(edit_kae_eigen(pair if is_complex else top[None]))

    linear_top = get_scores(l_coefficients, c).argmax()
    l_drops = linear_accs - get_class_accs(edit_linear_svd(linear_top[None]))

    rows[name] = {
        "mode": top.item(),
        "|eigenvalue|": eigenvalues[top].abs().item(),
        "complex": is_complex,
        "target": drops[c].item(),
        "other": ((drops.sum() - drops[c]) / 9).item(),
        "pair target": pair_drops[c].item(),
        "pair other": ((pair_drops.sum() - pair_drops[c]) / 9).item(),
        "linear target": l_drops[c].item(),
        "linear other": ((l_drops.sum() - l_drops[c]) / 9).item(),
    }

table = pd.DataFrame(rows).T
print(table[["target", "other", "pair target", "pair other", "linear target", "linear other"]].astype(float).mean().round(3))
table
'''

nb["cells"] += [
    cell("markdown", "##### Rank of what the data actually uses\n"),
    cell("code", ranks),
    cell("markdown", "##### The top eigenmode for each class\n"),
    cell("code", modes),
]
json.dump(nb, open("second_analysis/14_novc_editing.ipynb", "w"), indent=1)
