"""
How do grayscale and sepia copies of an image show up in the block-6 modes: as a change of size, or as a turn?

For each alteration we fit one complex number per mode, c, so that z6(altered) ≈ c · z6(original) on the images that keep the mode,
on half of the images, and test on the other half. |c| is a scaling and the angle of c is a turn.
A real c (scaling only, with a sign) is the comparison: it is all a latent without 2x2 blocks could do.
"""
import math, os
from pathlib import Path
import torch
import open_clip
import inversion
from inversion import train, to_complex, to_latent

torch.set_num_threads(16)
torch.set_grad_enabled(False)
kae, stats = inversion.load_kae()
head = train.DinoHead(inversion.ENCODER, stats["mean"], stats["std"], torch.device("cpu"), zero_shot=True)
lam = torch.cat([kae.real_eigenvalue.to(torch.cfloat), torch.complex(kae.block_real, kae.block_imag)])
data = torch.load("alter_states.pt")
labels = data["labels"]
x = {key: (data[key] - stats["mean"]) / stats["std"] for key in ["original", "gray", "sepia"]}
z = {key: to_complex(train.encode_pairs(kae, value)) for key, value in x.items()}
z6 = {key: value[:, 0] for key, value in z.items()}


def predict(z_start):
    return kae.decode(to_latent(lam * z_start))


pred = {key: predict(value) for key, value in z6.items()}
half = torch.arange(len(labels)) % 2 == 0  # fit on even rows, test on odd rows
test = ~half

print("== accuracy on the 2,000 images: DINOv3 itself | KAE prediction | KAE agrees with DINOv3")
for key in x:
    real_answers, kae_answers = head.classify(x[key][:, 1]), head.classify(pred[key])
    print(f"{key:9s} {(real_answers == labels).double().mean():.3f} | {(kae_answers == labels).double().mean():.3f} | {(kae_answers == real_answers).double().mean():.3f}")

print("== colourfulness of the originals:", data["colourfulness"].quantile(torch.tensor([0.1, 0.5, 0.9])).tolist())
# NOTE: Images that are already close to gray tell us nothing about losing colour
coloured = data["colourfulness"] > data["colourfulness"].quantile(0.25)

results = {}
for key in ["gray", "sepia"]:
    a, b = z6["original"], z6[key]
    both = (a != 0) & (b != 0)
    kept_a, kept_b = (a != 0), (b != 0)
    jaccard = ((kept_a & kept_b).sum(dim=1) / (kept_a | kept_b).sum(dim=1).clamp(min=1)).mean()
    print(f"\n==== {key}: modes kept by both copies / by either: {jaccard:.3f}")

    # Exact split of the change of every mode value: |a - b|² = (|a| - |b|)² + 2|a||b|(1 - cos Δφ)
    size_part = (a.abs() - b.abs()).pow(2)
    turn_part = 2 * a.abs() * b.abs() * (1 - torch.cos(torch.angle(b) - torch.angle(a)))
    total = (a - b).abs().pow(2)
    assert torch.allclose(size_part + turn_part, total, atol=1e-3)
    print(f"change of the block-6 latent, as a share of its size: {(total.sum() / a.abs().pow(2).sum()):.3f}")
    print(f"  of that change: size {size_part.sum() / total.sum():.3f} (modes switched on or off {size_part[~both].sum() / total.sum():.3f}), turn {turn_part.sum() / total.sum():.3f}")

    # One factor per mode, fitted on half of the images (least squares over images that keep the mode in the original)
    fit = half[:, None] & (a != 0)
    c = (torch.where(fit, a.conj() * b, 0).sum(dim=0)) / torch.where(fit, a.abs().pow(2), 0).sum(dim=0).clamp(min=1e-8)
    counts = fit.sum(dim=0)
    c = torch.where(counts >= 20, c, torch.ones_like(c))  # modes seen too rarely are left alone
    c_real = c.real.to(torch.cfloat)
    c_size = c.abs().to(torch.cfloat)
    results[key] = c

    used = counts >= 20
    weight = torch.where(fit, a.abs().pow(2), 0).sum(dim=0)
    angles = torch.rad2deg(torch.angle(c[used]))
    print(f"factors fitted for {int(used.sum())} modes; |c| quartiles {c[used].abs().quantile(torch.tensor([0.25, 0.5, 0.75])).tolist()}; |angle| quartiles {angles.abs().quantile(torch.tensor([0.25, 0.5, 0.75])).tolist()}")
    print(f"  modes with |angle| > 20°: {int((angles.abs() > 20).sum())}, > 45°: {int((angles.abs() > 45).sum())}; with |c| < 0.7: {int((c[used].abs() < 0.7).sum())}, > 1.3: {int((c[used].abs() > 1.3).sum())}")

    # Test half: how much of the way from the original's prediction to the altered copy's prediction does each knob go?
    def closed(z_start, rows):
        out = predict(z_start[rows])
        gap = (pred[key][rows] - pred["original"][rows]).pow(2).sum(dim=1)
        left = (pred[key][rows] - out).pow(2).sum(dim=1)
        answers = head.classify(out)
        return (1 - left.sum() / gap.sum()).item(), (answers == head.classify(pred[key][rows])).double().mean().item(), (answers == labels[rows]).double().mean().item()

    rows = test & coloured
    print(f"  test half, {int(rows.sum())} coloured images. columns: share of the way to the {key} copy's predicted block 12 | same answer as the {key} copy's prediction | accuracy")
    for name, factors in {"nothing": torch.ones_like(c), "complex factor (size and turn)": c, "real factor (size and sign)": c_real, "size only": c_size, "turn only": c / c.abs().clamp(min=1e-8)}.items():
        print(f"    {name:32s}", [round(v, 3) for v in closed(a * factors, rows)])
    # Ceilings: the altered copy's own values on the modes the original keeps, and everything
    print(f"    {'copy values, original modes':32s}", [round(v, 3) for v in closed(torch.where(a != 0, b, 0), rows)])

    # Latent level, test half: error left in z6 after each knob, on modes kept by both
    for name, factors in {"nothing": torch.ones_like(c), "complex": c, "real": c_real, "size only": c_size}.items():
        err = torch.where(both & rows[:, None], (a * factors - b).abs().pow(2), 0).sum() / torch.where(both & rows[:, None], (a - b).abs().pow(2), 0).sum()
        print(f"    latent change left on shared modes, {name}: {err:.3f}")

    # The modes that carry most of the change
    change = torch.where(fit, (a - b).abs().pow(2), 0).sum(dim=0)
    top = change.argsort(descending=True)[:12]
    print("  modes with most change (mode, share of change, |c|, angle°, |λ|, usage):")
    for m in top.tolist():
        print(f"    {m:5d} {change[m] / change.sum():.3f} {c[m].abs():.2f} {math.degrees(torch.angle(c[m])):+7.1f} {lam[m].abs():.2f} {(a[:, m] != 0).double().mean():.2f}")

torch.save({"factors": results, "indices": data["indices"]}, "alter_factors.pt")
