"""
What does the eigenvalue angle track that the eigenvalue size does not? (DINOv3 run im_dino_w21_wd, feature modes)

Joins every per-mode measurement made so far and, for each one, gives its rank correlation over modes with
  |angle of λ|, holding the size of λ and the mode's usage fixed
  the size of λ, holding |angle| and usage fixed.
A property that only the angle tracks would be a meaning for the angle that the size cannot claim.
With about 430 modes, a correlation under about 0.10 is noise (two standard errors).
"""

import pandas as pd
import torch

table = torch.load("dino_explore.pt", weights_only=False)["modes"].copy()
table = table.rename(columns={"R²": "λ·z6 explains z12 (R²)", "turn agreement": "true turn alike across images", "kept": "usage"})
table["|angle|"] = table["angle"].abs()
table["true gain / size of λ"] = table["gain"] / table["|λ|"]
table["|true turn - angle|"] = ((table["turn"] - table["angle"] + 180) % 360 - 180).abs()

effects = pd.read_csv("phase_size_all.csv").set_index("mode")
table["phase spread at block 6"] = effects["phase spread°"]
table["size spread at block 6"] = effects["size spread"]
table["accuracy lost: removed"] = effects["accuracy, base"] - effects["accuracy, removed"]
table["accuracy lost: phase set to its average"] = effects["accuracy, base"] - effects["accuracy, mean phase"]
table["accuracy lost: size set to its average"] = effects["accuracy, base"] - effects["accuracy, mean size"]
table["answers changed: removed"] = effects["changed, removed"]
table["answers changed: phase set to its average"] = effects["changed, mean phase"]
table["answers changed: size set to its average"] = effects["changed, mean size"]

views = torch.load("view_consistency.pt", weights_only=False)
views = pd.DataFrame(views["table"].numpy(), columns=views["columns"]).set_index("mode")
for key in ["kept by both", "size at 6", "phase at 6", "size at 12", "phase at 12", "turn 6 to 12"]:
    table[f"two views agree: {key}"] = views[key]

clock = torch.load("depth_clock.pt", weights_only=False)
multiplier = clock["multiplier"]
final_turn, final_growth = torch.angle(multiplier[:, -1]), multiplier[:, -1].abs().log()
clock_table = pd.DataFrame(
    {
        "share of the turn done by block 9": (torch.angle(multiplier[:, 3]) / final_turn).numpy(),
        "share of the growth done by block 9": (multiplier[:, 3].abs().log() / final_growth).numpy(),
        "share of the growth done by block 11": (multiplier[:, 5].abs().log() / final_growth).numpy(),
    },
    index=clock["modes"].tolist(),
)
# NOTE: A share of a turn or growth close to zero is a ratio of noise
clock_table.loc[(final_turn.abs() < 0.087).numpy(), "share of the turn done by block 9"] = float("nan")
clock_table.loc[(final_growth.abs() < 0.2).numpy(), ["share of the growth done by block 9", "share of the growth done by block 11"]] = float("nan")
table = table.join(clock_table)

ties = torch.load("angle_vs_class_im_dino_w21_wd.pt")
ties = pd.DataFrame({"class tie at block 6": ties["tie start"].numpy(), "class tie at block 12": ties["tie end"].numpy(), "class tie of both coordinates at block 6": ties["tie plane"].numpy()}, index=ties["modes"].tolist())
ties["class tie gained from 6 to 12"] = ties["class tie at block 12"] - ties["class tie at block 6"]
table = table.join(ties)

factors = torch.load("alter_factors.pt", weights_only=False)["factors"]
for key in ["gray", "sepia"]:
    table[f"{key} copy: change of size"] = [abs(factors[key][m].abs().item() - 1) for m in table.index]
    table[f"{key} copy: turn"] = [abs(torch.rad2deg(torch.angle(factors[key][m])).item()) for m in table.index]

parts = pd.DataFrame(torch.load("mode_parts.pt", weights_only=False)).set_index("mode")
table["push lands in the CLS"] = parts["mode: cls"]
table["push lands in the patch mean"] = parts["mode: patch mean"]
table["push lands in the patch std"] = parts["mode: patch std"]
table["push size"] = parts["mode: push size"]
table["built part lands in the CLS more than the kept part"] = parts["built: cls"] - parts["kept: cls"]

features = table[(table["usage"] < 0.99) & (table["usage"] >= 0.004)]
print(f"{len(features)} feature modes")
rank = lambda a: torch.tensor(a.rank().to_numpy(), dtype=torch.double)


def partial(a, b, controls):
    x = torch.stack([rank(c) for c in controls] + [torch.ones(len(a), dtype=torch.double)], dim=1)
    residual = lambda y: y - x @ torch.linalg.lstsq(x, y[:, None]).solution[:, 0]
    return torch.corrcoef(torch.stack([residual(rank(a)), residual(rank(b))]))[0, 1].item()


rows = []
skip = {"|λ|", "angle", "|angle|", "usage", "gain", "turn", "share 6", "share 12"}
for key in [c for c in features.columns if c not in skip]:
    has = features[features[key].notna()]
    rows.append(
        {
            "property": key,
            "modes": len(has),
            "with |angle| (size, usage fixed)": partial(has["|angle|"], has[key], [has["|λ|"], has["usage"]]),
            "with size (|angle|, usage fixed)": partial(has["|λ|"], has[key], [has["|angle|"], has["usage"]]),
            "with |angle| alone": partial(has["|angle|"], has[key], []),
            "with size alone": partial(has["|λ|"], has[key], []),
        }
    )
scan = pd.DataFrame(rows).sort_values("with |angle| (size, usage fixed)", key=abs, ascending=False)
pd.set_option("display.width", 250, "display.max_colwidth", 60, "display.max_rows", 100)
print(scan.round(3).to_string(index=False))
scan.to_csv("angle_scan.csv", index=False)
table.to_csv("dino_mode_table.csv")
