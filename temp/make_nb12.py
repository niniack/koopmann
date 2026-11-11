import json

source = json.load(open("second_analysis/16_mode_images.ipynb"))
code = ["".join(c["source"]) for c in source["cells"] if c["cell_type"] == "code"]
setup, load, modes, table, images, concepts, correlations, baselines, novelty = code


def cell(kind, text):
    new = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        new.update({"outputs": [], "execution_count": None})
    return new


intro = """### Is K just a rotation of the linear map's directions?

In `16_mode_images`, K's modes tracked scene-level concepts better than the linear map's top directions, even though those directions explain most of each mode. So K may simply be choosing better axes. Here we rotate the linear map's top directions with ICA, which needs no training, and repeat both tests.

Nothing is trained here. Concepts are measured on training images, within each class.
"""

setup = setup.replace("from dotenv import load_dotenv\n", "from dotenv import load_dotenv\nfrom sklearn.decomposition import FastICA\n")
setup += """NUM_ROTATED = [7, 20, 50]
ICA_SEEDS = [0, 1, 2]
"""

rotation = '''def get_ica(features, seed):
    ica = FastICA(whiten="unit-variance", max_iter=2000, random_state=seed)
    sources = ica.fit_transform(features.cpu().numpy())
    return ica, torch.tensor(sources, device=device, dtype=torch.double)


results = {
    "K top 7, seed 0": correlations_a.amax(dim=0).cpu(),
    "K top 7, seed 1": correlations_b.amax(dim=0).cpu(),
    "linear top 20": get_correlations(linear_features[:, :20]).abs().amax(dim=0).cpu(),
}

rotations = {}
for num in NUM_ROTATED:
    scores = []
    for seed in ICA_SEEDS:
        rotations[(num, seed)] = get_ica(linear_features[:, :num], seed)
        scores.append(get_correlations(rotations[(num, seed)][1]).abs().amax(dim=0))
    scores = torch.stack(scores)
    results[f"ICA of {num}"] = scores.mean(dim=0).cpu()
    results[f"ICA of {num}, spread"] = (scores.amax(dim=0) - scores.amin(dim=0)).cpu()

pd.DataFrame(results, index=concepts).round(2)
'''

editing = '''classifier = fit_linear(train_x[:, 1], F.one_hot(train_labels, 10).float())


def get_class_accs(states):
    predictions = (F.pad(states.double(), (0, 1), value=1) @ classifier).argmax(dim=1)
    correct = (predictions == labels).double()
    return torch.stack([correct[labels == c].mean() for c in range(10)])


def get_scores(coefficients, c):
    """How well each direction separates class `c`, on training images."""

    is_class = train_labels == c
    gap = coefficients[is_class].mean(dim=0) - coefficients[~is_class].mean(dim=0)
    spread = (coefficients - coefficients.mean(dim=0)).pow(2).mean(dim=0).sqrt()
    return gap.abs() / (spread + 1e-12)


def get_specificity(coefficients, edit, m, base=None):
    """Mean target drop, mean other-class drop, and their difference."""

    base = get_class_accs(val_outputs) if base is None else base
    target_drops, other_drops = [], []
    for c in range(10):
        drops = base - get_class_accs(edit(get_scores(coefficients, c).argsort(descending=True)[:m]))
        target_drops.append(drops[c])
        other_drops.append((drops.sum() - drops[c]) / 9)

    target, other = torch.stack(target_drops).mean().item(), torch.stack(other_drops).mean().item()
    return {"target drop": target, "other drop": other, "specificity": target - other}


directions = linear_directions[:, :]
val_coordinates = (val_outputs - val_outputs.mean(dim=0)) @ directions


def edit_principal(removed):
    return val_outputs - val_coordinates[:, removed] @ directions[:, removed].T


def get_ica_edit(num, seed):
    """Removes independent components from the outputs, leaving the mean in place."""

    ica, _ = rotations[(num, seed)]
    mixing = torch.tensor(ica.mixing_, device=device, dtype=torch.double)
    sources = torch.tensor(ica.transform(val_coordinates[:, :num].cpu().numpy()), device=device, dtype=torch.double)

    def edit(removed):
        return val_outputs - (sources[:, removed] @ mixing[:, removed].T) @ directions[:, :num].T

    return edit


rows = {}
for m in [1, 2, 5]:
    rows[(m, "linear principal")] = get_specificity(linear_features, edit_principal, m)
    for num in NUM_ROTATED:
        trials = [get_specificity(rotations[(num, seed)][1], get_ica_edit(num, seed), m) for seed in ICA_SEEDS]
        rows[(m, f"ICA of {num}")] = pd.DataFrame(trials).mean().to_dict()

pd.DataFrame(rows).T.round(3)
'''

kae_rotation = '''def get_kae_rotations(name):
    \"\"\"The same rotation as the linear map, applied to K's outputs in the latent space.\"\"\"

    kae, _ = KoopmanAutoencoder.load_model(file_path=save_dir / f"{name}.safetensors")
    kae = kae.to(device).eval()
    operator = kae.koopman_weights.T.double().contiguous()

    train_out = kae.encode(train_x[:, 0]).double() @ operator
    val_out = kae.encode(val_x[:, 0]).double() @ operator
    out_directions = get_principal_directions(train_out)
    features = (train_out - train_out.mean(dim=0)) @ out_directions
    coordinates = (val_out - val_out.mean(dim=0)) @ out_directions
    base = get_class_accs(kae.decode(val_out.float()))

    rows, sources = {}, {}
    for num in NUM_ROTATED:
        ica, train_sources = get_ica(features[:, :num], 0)
        sources[num] = train_sources
        mixing = torch.tensor(ica.mixing_, device=device, dtype=torch.double)
        val_sources = torch.tensor(ica.transform(coordinates[:, :num].cpu().numpy()), device=device, dtype=torch.double)

        def edit(removed):
            change = (val_sources[:, removed] @ mixing[:, removed].T) @ out_directions[:, :num].T
            return kae.decode((val_out - change).float())

        for m in [1, 2, 5]:
            rows[(m, f"K ICA of {num}")] = get_specificity(train_sources, edit, m, base)

    return rows, sources


kae_rows_a, kae_sources_a = get_kae_rotations("b6_novc")
kae_rows_b, kae_sources_b = get_kae_rotations("b6_novc_s1")

summary = {}
for m in [1, 2, 5]:
    summary[m] = {"linear principal": rows[(m, "linear principal")]["specificity"]}
    for num in NUM_ROTATED:
        summary[m][f"linear ICA of {num}"] = rows[(m, f"ICA of {num}")]["specificity"]
        summary[m][f"K ICA of {num}, seed 0"] = kae_rows_a[(m, f"K ICA of {num}")]["specificity"]
        summary[m][f"K ICA of {num}, seed 1"] = kae_rows_b[(m, f"K ICA of {num}")]["specificity"]

pd.DataFrame(summary).round(3)
'''

neutral = '''# NOTE: This list was fixed before seeing any result, so it favours no method
NEUTRAL = [
    "the sky", "grass", "water", "snow", "night", "an indoor room", "the outdoors", "a crowd", "a face", "hands",
    "text", "a black and white image", "a close-up", "a landscape", "a white background", "a building", "a vehicle",
    "an animal", "a machine", "metal",
]

with torch.no_grad():
    plain = F.normalize(clip.encode_text(tokenizer(["a photo"]).to(device)), dim=-1)[0]
    prompts = tokenizer([f"a photo of {concept}" for concept in NEUTRAL]).to(device)
    neutral_values = (embeddings @ (F.normalize(clip.encode_text(prompts), dim=-1) - plain).T).double()


def get_neutral_best(features):
    \"\"\"The best absolute correlation with each neutral concept, among images of the same class.\"\"\"

    x = F.normalize(remove_class_means(features), dim=0)
    y = F.normalize(remove_class_means(neutral_values), dim=0)
    return (x.T @ y).abs().amax(dim=0).cpu()


methods = {
    "K top 7, seed 0": get_neutral_best(features_a),
    "K top 7, seed 1": get_neutral_best(features_b),
    "linear top 7": get_neutral_best(linear_features[:, :7]),
    "linear top 20": get_neutral_best(linear_features[:, :20]),
    "linear ICA of 7": get_neutral_best(rotations[(7, 0)][1]),
    "linear ICA of 20": get_neutral_best(rotations[(20, 0)][1]),
    "linear ICA of 50": get_neutral_best(rotations[(50, 0)][1]),
    "K ICA of 7, seed 0": get_neutral_best(kae_sources_a[7]),
    "K ICA of 7, seed 1": get_neutral_best(kae_sources_b[7]),
    "K ICA of 20, seed 0": get_neutral_best(kae_sources_a[20]),
    "K ICA of 20, seed 1": get_neutral_best(kae_sources_b[20]),
    "K ICA of 50, seed 0": get_neutral_best(kae_sources_a[50]),
    "K ICA of 50, seed 1": get_neutral_best(kae_sources_b[50]),
}

neutral_table = pd.DataFrame(methods, index=NEUTRAL)
print(neutral_table.mean().round(3).to_string())
neutral_table.round(2)
'''

nb = {
    "cells": [
        cell("markdown", intro),
        cell("code", setup),
        cell("markdown", "##### Load data\n"),
        cell("code", load),
        cell("markdown", "##### K's modes\n"),
        cell("code", modes),
        cell("code", table),
        cell("markdown", "##### Concepts\n"),
        cell("code", concepts),
        cell("code", correlations),
        cell("markdown", "##### Linear directions\n"),
        cell("code", baselines),
        cell("markdown", "##### Does rotating the linear directions find the same concepts?\n\nThe best absolute correlation for each concept, among each method's directions. ICA results are averaged over three initializations, and `spread` is the gap between the best and worst of them.\n"),
        cell("code", rotation),
        cell("markdown", "##### Do rotated directions edit better?\n\nThe same editing test as `15_data_svd`, removing the top `m` class-separating directions.\n"),
        cell("code", editing),
        cell("markdown", "##### The same rotation on K\\n\\nOnly the linear map was rotated above, which isn't a fair comparison. Here K's outputs get the same treatment.\\n"),
        cell("code", kae_rotation),
        cell("markdown", "##### A neutral set of concepts\\n\\nThe earlier concepts were guessed from K's modes, which favours K. These twenty were chosen without looking at any method. Each is a CLIP text score, compared with the plain prompt `a photo`.\\n"),
        cell("code", neutral),
    ],
    "metadata": source["metadata"],
    "nbformat": 4,
    "nbformat_minor": 5,
}
json.dump(nb, open("second_analysis/17_rotation.ipynb", "w"), indent=1)
