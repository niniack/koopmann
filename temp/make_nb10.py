import json
import sys

scratch = sys.argv[1]
nb = json.load(open("second_analysis/14_novc_editing.ipynb"))


def cell(kind, source):
    new = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        new.update({"outputs": [], "execution_count": None})
    return new


nb["cells"] = nb["cells"][:8]
for old in nb["cells"]:
    source = "".join(old["source"])
    source = source.replace('"b6_novc.safetensors"', 'f"{NAME}.safetensors"')
    source = source.replace('"b6_novc_stats.pt"', 'f"{NAME}_stats.pt"')
    source = source.replace("NUM_REMOVED = [1, 5, 20, 50, 100]", 'NUM_REMOVED = [1, 2, 5, 20, 50, 100]\nNAME = "b6_novc"')
    old["source"] = source.splitlines(keepends=True)
    if old["cell_type"] == "code":
        old["outputs"] = []
        old["execution_count"] = None

nb["cells"][0]["source"] = [
    "### Do directions taken from the data edit better than K's own?\n",
    "\n",
    "Most of K's spectrum sits in directions the data barely uses, so K's plain SVD may be a poor set of handles. Here we take the principal directions of each operator's outputs on training images, which is the SVD of K applied to the data. The linear map gets the same treatment.\n",
    "\n",
    "To edit, we remove the top `m` class-separating directions from the output, keeping the mean output. Nothing is trained here.\n",
    "\n",
    "**Specificity** is the target class's accuracy drop minus the mean drop of the other classes.\n",
]

directions = '''def get_output_directions(outputs):
    """Principal directions of an operator's outputs on training images."""

    mean = outputs.mean(dim=0)
    _, values, directions = torch.linalg.svd(outputs - mean, full_matrices=False)
    return mean, values, directions.T


def project_out(outputs, mean, directions, removed):
    picked = directions[:, removed]
    return outputs - ((outputs - mean) @ picked) @ picked.T


train_k_outputs = train_phi @ operator
train_l_outputs = train_x[:, 0].double() @ linear_weight + linear_bias
k_mean, k_data_values, k_data = get_output_directions(train_k_outputs)
l_mean, l_data_values, l_data = get_output_directions(train_l_outputs)


def edit_kae_data(removed):
    return decode(project_out(val_phi @ operator, k_mean, k_data, removed))


def edit_linear_data(removed):
    outputs = val_x[:, 0].double() @ linear_weight + linear_bias
    return project_out(outputs, l_mean, l_data, removed)
'''

results = '''k_coefficients = train_phi @ k_left
eigen_coefficients = train_phi.to(eigenvalues.dtype) @ eigenvectors
l_coefficients = train_x[:, 0].double() @ l_left
k_data_coefficients = (train_k_outputs - k_mean) @ k_data
l_data_coefficients = (train_l_outputs - l_mean) @ l_data

rows = {}
for m in NUM_REMOVED:
    rows[(m, "K singular")] = run(edit_kae_svd, k_coefficients, kae_accs, m)
    rows[(m, "K eigen")] = run(edit_kae_eigen, eigen_coefficients, kae_accs, m)
    rows[(m, "K data")] = run(edit_kae_data, k_data_coefficients, kae_accs, m)
    rows[(m, "K data random")] = run(edit_kae_data, k_data_coefficients, kae_accs, m, random=True)
    rows[(m, "linear singular")] = run(edit_linear_svd, l_coefficients, linear_accs, m)
    rows[(m, "linear data")] = run(edit_linear_data, l_data_coefficients, linear_accs, m)
    rows[(m, "linear data random")] = run(edit_linear_data, l_data_coefficients, linear_accs, m, random=True)

results = pd.DataFrame(rows).T
results.round(3)
'''

summary = '''results["specificity"].unstack().round(3)
'''

spectrum = '''def get_shares(values):
    shares = values.pow(2) / values.pow(2).sum()
    return {f"top {m}": shares[:m].sum().item() for m in NUM_REMOVED}


pd.DataFrame({"K data": get_shares(k_data_values), "linear data": get_shares(l_data_values)}).T.round(3)
'''

per_class = '''rows = {}
for c, name in enumerate(CLASSES):
    removed = get_scores(k_data_coefficients, c).argsort(descending=True)[:5]
    k_drops = kae_accs - get_class_accs(edit_kae_data(removed))

    removed = get_scores(l_data_coefficients, c).argsort(descending=True)[:5]
    l_drops = linear_accs - get_class_accs(edit_linear_data(removed))

    rows[name] = {
        "K target": k_drops[c].item(),
        "K other": ((k_drops.sum() - k_drops[c]) / 9).item(),
        "linear target": l_drops[c].item(),
        "linear other": ((l_drops.sum() - l_drops[c]) / 9).item(),
    }

pd.DataFrame(rows).T.round(3)
'''

nb["cells"] += [
    cell("markdown", "##### Directions from the data\n"),
    cell("code", directions),
    cell("markdown", "##### Results\n"),
    cell("code", results),
    cell("markdown", "##### Specificity\n"),
    cell("code", summary),
    cell("markdown", "##### Share of output variance in the top directions\n"),
    cell("code", spectrum),
    cell("markdown", "##### Per class, removing the top 5 data directions\n"),
    cell("code", per_class),
]

json.dump(nb, open("second_analysis/15_data_svd.ipynb", "w"), indent=1)

# Same notebook on the older checkpoint, kept out of the repo
text = json.dumps(nb, indent=1)
text = text.replace('NAME = \\"b6_novc\\"', 'NAME = \\"b6_30v\\"')
text = text.replace('load_dotenv(\\"../.env\\")', 'load_dotenv(\\"/home/nsa325/work/koopmann/.env\\")')
open(f"{scratch}/15_data_svd_30v.ipynb", "w").write(text)
