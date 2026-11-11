import json, os
os.chdir("/home/nsa325/work/koopmann/second_analysis")
nb = json.load(open("20_steering.ipynb"))
cells = ["".join(c["source"]) for c in nb["cells"]]
for i in [1, 3, 5, 6, 8]:
    exec(cells[i], globals())
exec(cells[10].split("linear = fit_linear")[0], globals())
exec(cells[12].split("true_scores =")[0], globals())

pd.set_option("display.float_format", lambda v: f"{v:.3f}")


def predict(states):
    return (F.pad(states.double(), (0, 1), value=1) @ classifier).argmax(dim=1)


def effects(base, edited, active):
    """Same quantities as the notebook, but split by which images the edit really touches."""
    change = edited.double() - base.double()
    size = change.norm(dim=1) / base.double().norm(dim=1)
    base_emb, edit_emb = to_embeddings(base), to_embeddings(edited)
    base_panel, edit_panel = base_emb @ panel_directions, edit_emb @ panel_directions
    delta = edit_panel - base_panel
    variance = remove_class_means(base_panel, val_y).pow(2).mean(dim=0)

    def other(mask):
        return ((delta[mask].pow(2).mean(dim=0)) / variance).median().item()

    flipped = predict(base) != predict(edited)
    colour = ~val_grey
    grey_axis = F.normalize(
        remove_class_means(base.double(), val_y)[val_grey].mean(0) - remove_class_means(base.double(), val_y)[colour].mean(0), dim=0
    )
    on_axis = ((change[val_grey] @ grey_axis).pow(2) / change[val_grey].pow(2).sum(1).clamp_min(1e-12)).mean()
    return {
        "active b&w": active[val_grey].double().mean().item(),
        "active colour": active[colour].double().mean().item(),
        "size b&w": size[val_grey].mean().item(),
        "size colour": size[colour].mean().item(),
        "other, all": other(torch.ones_like(val_grey)),
        "other, b&w": other(val_grey),
        "other, active": other(active),
        "flip b&w %": 100 * flipped[val_grey].double().mean().item(),
        "flip colour %": 100 * flipped[colour].double().mean().item(),
        "flip active colour %": 100 * flipped[colour & active].double().mean().item() if (colour & active).any() else float("nan"),
        "on b&w axis": on_axis.item(),
    }


CONFIGS = {
    "b6_jordan128": (100, 128, True),
    "b6_jordan128_s1": (171, 128, True),
    "b6_jordan": (442, 0, True),
    "b6_jordan_s1": (71, 0, True),
    "b6_novc": (6, 0, False),
    "b6_novc_s1": (12, 0, False),
    "b6_k128": (6, 128, False),
}

rows, controls, precision = {}, {}, {}
for name, (mode, top_k, jordan) in CONFIGS.items():
    train_f, val_f, edit, base, labels = get_mode_features(name, top_k, jordan)
    index = labels.index(f"mode {mode}")
    if jordan:
        modes = get_jordan_modes(load_model(name, top_k, jordan))[1](val_x[:, 0])
    if jordan and top_k:
        active = modes[:, mode].abs() > 0
    else:
        active = torch.ones_like(val_grey)
    rows[name] = effects(base, edit(index), active)

    if jordan and top_k:
        count = int(val_grey.sum())
        top = val_f[:, index].abs().argsort(descending=True)[:count]
        precision[name] = {
            "images active": active.sum().item(),
            "b&w among active %": 100 * val_grey[active].double().mean().item(),
            "b&w in top 154 %": 100 * val_grey[top].double().mean().item(),
            "median colour, active non-b&w": val_colour[active & ~val_grey].median().item(),
            "median colour, all": val_colour.median().item(),
        }
        for i, label in enumerate(labels):
            m = int(label.split()[1])
            a = modes[:, m].abs() > 0
            e = edit(i)
            r = effects(base, e, a)
            controls[(name, label)] = {
                "active %": 100 * a.double().mean().item(),
                "notebook other": measure(base, e)["other concepts moved"],
                "other, active": r["other, active"],
                "flip colour %": r["flip colour %"],
                "flip active %": 100 * (predict(base) != predict(e))[a].double().mean().item(),
                "colour acc change": 100 * (measure(base, e)["accuracy, colour"] - measure(base, base)["accuracy, colour"]),
            }

print("\n=== where the edit lands ===")
print(pd.DataFrame(rows).T.to_string())
print("\n=== what the mode fires on ===")
print(pd.DataFrame(precision).T.to_string())
print("\n=== control: remove each of the top 20 modes ===")
print(pd.DataFrame(controls).T.to_string())
