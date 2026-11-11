import json
import subprocess
import sys

SP = "/home/nsa325/work/koopmann/temp"
target = "/home/nsa325/work/koopmann/second_analysis/24_topk_modes.ipynb"
models = """{
    "im_jordan128": {"top_k": 128, "jordan": True},
    "im_jordan_2k_v3_both": {"top_k": 0, "jordan": True, "width": 2048},
    "im_jordan256_2k_both": {"top_k": 256, "jordan": True, "width": 2048},
}"""
subprocess.run([sys.executable, f"{SP}/make_nb18.py", models, target], check=True)
nb = json.load(open(target))


def get(i):
    return "".join(nb["cells"][i]["source"])


def put(i, text):
    nb["cells"][i]["source"] = text.splitlines(keepends=True)


def swap(i, old, new):
    text = get(i)
    assert text.count(old) == 1, old
    put(i, text.replace(old, new))


def add(cell_type, text):
    cell = {"cell_type": cell_type, "metadata": {}, "source": text.splitlines(keepends=True)}
    if cell_type == "code":
        cell.update({"execution_count": None, "outputs": []})
    nb["cells"].append(cell)


put(0, """### TopK at 2049 wide: does a single mode carry snow?

Rows named after a model are its raw modes: one feature per eigenvalue of K. Rows ending in `ICA` are directions in K's outputs found by ICA; they mix many modes and are not part of K's spectrum.

- `im_jordan128`: 1025 wide, top 128 coordinates (64 of 513 modes), 20 epochs, no CLIP losses. Its mode 414 was the snow mode.
- `im_jordan_2k_v3_both`: 2049 wide, no TopK, with the embedding and KL losses. Checkpoint from epoch 38.
- `im_jordan256_2k_both`: the same, keeping the top 256 coordinates (128 of 1025 modes). Checkpoint from its best epoch, stopped at epoch 12.

Removing a raw mode sets it to its training mean, which is the same edit as setting its eigenvalue to zero.
""")

swap(5, '''    kae.load_state_dict(load_file(save_dir / f"{name}.safetensors"))
    return kae.to(device).eval()''', '''    kae.load_state_dict(load_file(save_dir / f"{name}.safetensors"))

    # NOTE: Models trained on augmented views have their own standardization, so we convert to and from it
    own = torch.load(save_dir / f"{name}_stats.pt", map_location=device)
    scale, shift = stats["std"] / own["std"], (stats["mean"] - own["mean"]) / own["std"]
    encode, decode = kae.encode, kae.decode
    kae.encode = lambda x: encode(x * scale + shift)
    kae.decode = lambda phi: (decode(phi) - shift) / scale

    return kae.to(device).eval()''')

add("markdown", """##### As a surrogate

CLIP's own zero-shot classifier on each model's prediction of block 12. `agrees with CLIP` is how often the prediction gets CLIP's answer, right or wrong.
""")
add("code", '''clip.to("cuda")
head = open_clip.build_zero_shot_classifier(
    clip,
    tokenizer,
    open_clip.IMAGENET_CLASSNAMES,
    open_clip.OPENAI_IMAGENET_TEMPLATES,
    num_classes_per_batch=10,
    device="cuda",
    use_tqdm=False,
).cpu()
clip.to("cpu")


def zero_shot(states):
    return (to_embeddings(states) @ head).argmax(dim=1)


surrogates = {name: bundles[f"{name} ICA"][2] for name in MODELS}
surrogates["linear"] = bundles["linear ICA"][2]

true_val, true_snow = zero_shot(val_x[:, 1]), zero_shot(x[:, 1])
rows = {
    "real CLIP": {
        "zero-shot": (true_val == val_y).double().mean().item(),
        "agrees with CLIP": 1.0,
        "snowy": (true_snow == y)[is_snow].double().mean().item(),
        "normal": (true_snow == y)[~is_snow].double().mean().item(),
    }
}
for name, predict in surrogates.items():
    answers, snow_answers = zero_shot(predict(val_x[:, 0])), zero_shot(predict(x[:, 0]))
    rows[name] = {
        "zero-shot": (answers == val_y).double().mean().item(),
        "agrees with CLIP": (answers == true_val).double().mean().item(),
        "snowy": (snow_answers == y)[is_snow].double().mean().item(),
        "normal": (snow_answers == y)[~is_snow].double().mean().item(),
    }

pd.DataFrame(rows).T.round(3)
''')
add("markdown", """##### The snow edit, scored with CLIP's head

As in the snow test: the snow feature is picked on one half of the images, and the edit is scored on the other half. Raw modes and ICA features both.
""")
add("code", '''rows = {}
for name in bundles:
    features, edit, predict, labels = bundles[name]
    scores = features(x[:, 0])
    correct = zero_shot(predict(x[:, 0])) == y
    halves = []
    for half in [0, 1]:
        pick, test = fold == half, fold != half
        best = get_auc(scores[pick], is_snow[pick]).argmax().item()
        fixed = zero_shot(edit(x[:, 0], best)) == y
        halves.append(
            {
                "snowy before": correct[test & is_snow].double().mean().item(),
                "snowy after": fixed[test & is_snow].double().mean().item(),
                "normal before": correct[test & ~is_snow].double().mean().item(),
                "normal after": fixed[test & ~is_snow].double().mean().item(),
            }
        )
    rows[name] = pd.DataFrame(halves).mean().to_dict()

pd.DataFrame(rows).T.round(3)
''')

json.dump(nb, open(target, "w"), indent=1)
print("written", target)
