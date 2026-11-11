"""Where DINOv3 (ViT-B/16 + its ImageNet probe) fails on ImageNet val, and whether editing a KAE mode fixes it. CPU only."""
import math
import os
import sys
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
from dotenv import load_dotenv
from safetensors.torch import load_file

sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
import train  # noqa: E402

load_dotenv("/home/nsa325/work/koopmann/.env")
torch.set_grad_enabled(False)
torch.set_num_threads(20)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 90)
SP = Path(__file__).parent
NAME, ENCODER = "im_dino_w21_wd", "dinov3-vitb16"
names = open_clip.IMAGENET_CLASSNAMES

ckpt, root = Path(os.getenv("WEIGHTS_CACHE")) / "tokae", Path(os.getenv("DATASETS_CACHE"))
stats = torch.load(ckpt / f"{NAME}_stats.pt")
mean, std = stats["mean"], stats["std"]
val = torch.load(root / "imagenet" / f"clip_cls_patch_std_{ENCODER}_val_blocks_5_11.pt")
val_x, val_y = (val["states"].float() - mean) / std, val["labels"]
kae = train.JordanKoopmanAutoencoder(256, shared_modes=True, in_features=2304, latent_features=2049, hidden_config=stats["hidden_config"], bias=True, batchnorm=False, nonlinearity="leaky_relu")
kae.load_state_dict(load_file(ckpt / f"{NAME}.safetensors"))
kae.eval()
head = train.DinoHead(ENCODER, mean, std, torch.device("cpu"), zero_shot=True)
lam = torch.cat([kae.real_eigenvalue.to(torch.cfloat), torch.complex(kae.block_real, kae.block_imag)])


def to_latent(z):
    return torch.cat([z[..., :1].real, torch.view_as_real(z[..., 1:]).flatten(-2)], dim=-1)


def decode(z):
    return torch.cat([kae.decode(to_latent(batch)) for batch in z.split(5_000)])


z = torch.load(SP / "dino_explore.pt", weights_only=False)["z"]
z6, z12 = z[:, 0], z[:, 1]
kept = z6 != 0
share = kept.double().mean(dim=0)

# ---------------------------------------------------------------- 1. DINOv3's errors
log_probs = head.get_log_probs(val_x[:, 1])
answers = log_probs.argmax(dim=1)
wrong = answers != val_y
top5 = log_probs.topk(5).indices
in_top5 = (top5 == val_y[:, None]).any(dim=1)
second = top5[:, 1] == val_y
print("=== 1. DINOv3 on the 50,000 validation images ===")
print(f"accuracy {1 - wrong.double().mean():.4f}, top-5 accuracy {in_top5.double().mean():.4f}; {int(wrong.sum())} wrong answers")
print(f"of the wrong answers: the label is DINOv3's second choice for {second[wrong].double().mean():.3f}, and among its top 5 for {in_top5[wrong].double().mean():.3f}")
confidence = log_probs.max(dim=1).values.exp()
print(f"median confidence when right {confidence[~wrong].median():.3f}, when wrong {confidence[wrong].median():.3f}")

confusion = torch.zeros(1000, 1000, dtype=torch.long)
confusion.index_put_((val_y[wrong], answers[wrong]), torch.ones(int(wrong.sum()), dtype=torch.long), accumulate=True)
flat = confusion.flatten().argsort(descending=True)[:25]
print("\nmost common mistakes (label -> DINOv3's answer, count out of 50, and the count the other way round):")
for index in flat.tolist():
    a, b = divmod(index, 1000)
    print(f"   {names[a]:32s} -> {names[b]:32s} {int(confusion[a, b]):3d}   back {int(confusion[b, a]):3d}")
both = confusion + confusion.T
print(f"wrong answers inside a pair of classes confused 10+ times in total: {int(confusion[both >= 10].sum())} of {int(wrong.sum())} ({confusion[both >= 10].sum() / wrong.sum():.3f}), over {int((both >= 10).sum() // 2)} pairs")
print(f"wrong answers inside a pair confused 5+ times: {confusion[both >= 5].sum() / wrong.sum():.3f}")
errors = torch.zeros(1000).index_add_(0, val_y, wrong.float())
print("\nworst classes (accuracy, most common wrong answer):")
for c in errors.argsort(descending=True)[:20].tolist():
    print(f"   {names[c]:32s} {1 - errors[c] / 50:.2f}   {names[confusion[c].argmax()]} ({int(confusion[c].max())})")
print(f"classes under 50% accuracy: {int((errors > 25).sum())}; under 70%: {int((errors > 15).sum())}; at 100%: {int((errors == 0).sum())}")
print(f"share of all wrong answers in the 100 worst classes: {errors.sort(descending=True).values[:100].sum() / errors.sum():.3f}")

# ---------------------------------------------------------------- 2. the KAE on DINOv3's errors
base = lam * z6
kae_answers = head.classify(decode(base))
kae_right = kae_answers == val_y
print("\n=== 2. the KAE on the images DINOv3 gets wrong ===")
print(f"KAE accuracy {kae_right.double().mean():.4f}; on images DINOv3 gets right {kae_right[~wrong].double().mean():.4f}; on images DINOv3 gets wrong {kae_right[wrong].double().mean():.4f}")
print(f"on DINOv3's {int(wrong.sum())} wrong answers the KAE gives: the label {int(kae_right[wrong].sum())}, DINOv3's wrong answer {int((kae_answers == answers)[wrong].sum())}, a third answer {int((~kae_right & (kae_answers != answers))[wrong].sum())}")

# ---------------------------------------------------------------- 3. modes that flag DINOv3's errors
# For each mode: DINOv3's wrong answers among the images that keep it, against the count expected from each image's class
# alone (drawing the same number of images from each class at random).
print("\n=== 3. modes that flag DINOv3's errors, beyond what the image's class explains ===")
feature = (share > 0.005) & (share < 0.99)
modes = torch.nonzero(feature).flatten()
onehot = torch.zeros(len(val_y), 1000).scatter_(1, val_y[:, None], 1.0)


def get_excess(rows, per_class):
    on = kept[rows][:, modes].float()
    hot, bad = onehot[rows], wrong[rows].float()
    count = hot.T @ on
    seen = (on * bad[:, None]).sum(dim=0)
    rate = (hot.T @ bad) / per_class
    expected = (count * rate[:, None]).sum(dim=0)
    variance = (count * (rate * (1 - rate))[:, None] * (per_class - count) / (per_class - 1)).sum(dim=0)
    return on.sum(dim=0), seen, expected, (seen - expected) / variance.sqrt()


everything = torch.arange(len(val_y))
count, seen, expected, score = get_excess(everything, 50)
# held-out check: each class's images alternate between the two halves
position = torch.zeros(len(val_y), dtype=torch.long)
for c in range(1000):
    members = torch.nonzero(val_y == c).flatten()
    position[members] = torch.arange(len(members))
half_a, half_b = torch.nonzero(position % 2 == 0).flatten(), torch.nonzero(position % 2 == 1).flatten()
_, seen_a, expected_a, score_a = get_excess(half_a, 25)
_, seen_b, expected_b, score_b = get_excess(half_b, 25)
print(f"{len(modes)} feature modes tested. Scores (in standard deviations) above 3.7 would be surprising for one of {len(modes)} modes by chance.")
print(f"modes with score > 3.7: {int((score > 3.7).sum())}; > 3: {int((score > 3).sum())}; < -3.7: {int((score < -3.7).sum())}; < -3: {int((score < -3).sum())}")
print(f"score on one half of the images against the other half, over modes: correlation {np.corrcoef(score_a.numpy(), score_b.numpy())[0, 1]:.3f}")
table = pd.DataFrame(
    {
        "mode": modes.tolist(),
        "kept": share[modes].numpy().round(3),
        "|λ|": lam[modes].abs().numpy().round(2),
        "angle°": (torch.angle(lam[modes]) * 180 / math.pi).numpy().round(1),
        "DINOv3 wrong": seen.long().tolist(),
        "expected": expected.numpy().round(1),
        "excess": (seen - expected).numpy().round(1),
        "score": score.numpy().round(2),
        "score half A": score_a.numpy().round(2),
        "score half B": score_b.numpy().round(2),
    }
).set_index("mode")


def describe(mode):
    on = kept[:, mode] & wrong
    pairs = val_y[on] * 1000 + answers[on]
    values, counts = pairs.unique(return_counts=True)
    order = counts.argsort(descending=True)[:3]
    return "; ".join(f"{names[int(values[i]) // 1000]} -> {names[int(values[i]) % 1000]} ({int(counts[i])})" for i in order)


top = table.sort_values("score", ascending=False).head(15)
top["top mistakes on its images"] = [describe(m) for m in top.index]
print("\nmodes whose images DINOv3 gets wrong more often than their classes explain:")
print(top.to_string())
bottom = table.sort_values("score").head(6)
print("\nand the other end (DINOv3 gets these images right more often than their classes explain):")
print(bottom.to_string())

# ---------------------------------------------------------------- 4. does editing the mode fix them?
print("\n=== 4. editing the flagged modes ===")
print("'through K' edits the KAE's own prediction. 'true block 12' edits the mode in DINOv3's real block-12 state, encoded on the")
print("modes picked at block 6, which is the best any K could do. Fixed and broken are against the unedited answers, on the images that keep the mode.")
EDITS = {"removed": 0, "x0.5": 0.5, "x2": 2, "+90°": 1j, "-90°": -1j, "180°": -1}
flagged = table.sort_values("score", ascending=False).head(12).index.tolist()
rows, effects = [], {}
for mode in flagged:
    on = torch.nonzero(kept[:, mode]).flatten()
    labels, dino_wrong = val_y[on], wrong[on]
    for source, latent in [("through K", base[on]), ("true block 12", z12[on])]:
        before = head.classify(decode(latent)) == labels
        row = {"mode": mode, "edit of": source, "images": len(on), "DINOv3 wrong": int(dino_wrong.sum()), "right before": int(before.sum()), "right before, where DINOv3 wrong": int(before[dino_wrong].sum())}
        for label, factor in EDITS.items():
            edited = latent.clone()
            edited[:, mode] = edited[:, mode] * factor
            after = head.classify(decode(edited)) == labels
            row[label] = f"{int((~before & after).sum())}/{int((before & ~after).sum())}"
            if label == "removed":
                row["removed, where DINOv3 wrong"] = f"{int((~before & after)[dino_wrong].sum())}/{int((before & ~after)[dino_wrong].sum())}"
                effects[(mode, source)] = (on, before, after)
        rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))

# ---------------------------------------------------------------- 5. held-out: pick modes on half A, score on half B
print("\n=== 5. held-out: flag modes on half of the images, remove them, score on the other half ===")
in_a = position % 2 == 0
picked = table.sort_values("score half A", ascending=False).head(30).index.tolist()
for source, latent_all in [("through K", base), ("true block 12", z12)]:
    before_all = head.classify(decode(latent_all)) == val_y
    kept_modes, totals = [], np.zeros(4, dtype=int)
    for mode in picked:
        on = torch.nonzero(kept[:, mode]).flatten()
        edited = latent_all[on].clone()
        edited[:, mode] = 0
        after = head.classify(decode(edited)) == val_y[on]
        before = before_all[on]
        fixed, broken = ~before & after, before & ~after
        a = in_a[on]
        net_a = int(fixed[a].sum()) - int(broken[a].sum())
        if net_a > 0:
            kept_modes.append(mode)
            totals += np.array([int(fixed[a].sum()), int(broken[a].sum()), int(fixed[~a].sum()), int(broken[~a].sum())])
    print(f"{source}: of the 30 modes flagged on half A, removal helps on half A for {len(kept_modes)} ({kept_modes}).")
    print(f"   those modes, one at a time: half A fixed {totals[0]} / broken {totals[1]}; half B fixed {totals[2]} / broken {totals[3]}")
    if kept_modes:
        edited = latent_all.clone()
        edited[:, kept_modes] = 0
        after_all = head.classify(decode(edited)) == val_y
        for label, rows_ in [("half A", in_a), ("half B", ~in_a)]:
            print(f"   all removed together, {label}: accuracy {before_all[rows_].double().mean():.4f} -> {after_all[rows_].double().mean():.4f} (fixed {int((~before_all & after_all)[rows_].sum())}, broken {int((before_all & ~after_all)[rows_].sum())})")
torch.save({"table": table, "flagged": flagged}, SP / "dino_failures.pt")
