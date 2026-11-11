"""
Reads the blocks between 6 and 12 through the KAE's encoder, to see how each mode gets from its block-6 value to its block-12 value.

For every block b and mode m, the "multiplier so far" is the one complex number that best maps the mode's block-6 values to its
block-b values over the images that keep it. It is 1 at block 6 and close to the eigenvalue at block 12.
If the eigenvalue's angle were a clock for depth, the multiplier's angle would move steadily from 0 to the eigenvalue's angle.

NOTE: The KAE never saw blocks 7 to 11, so the first thing printed is how well it reconstructs them.
"""

import numpy as np
import torch

import inversion
from inversion import to_complex, to_latent, train

torch.set_num_threads(6)
torch.set_grad_enabled(False)
OFFSET, GROWTH, INERT = [861, 338, 144, 560, 117, 988, 871, 746, 652, 533], [331, 77], [68, 653, 877, 465]
MIN_IMAGES = 40

kae, stats = inversion.load_kae()
data = torch.load("depth_states.pt")
states = (data["states"] - stats["mean"]) / stats["std"]
blocks = [block + 1 for block in data["blocks"]]
lam = torch.cat([kae.real_eigenvalue.to(torch.cfloat), torch.complex(kae.block_real, kae.block_imag)])

# shape: [images, blocks, modes], complex. Every block is read on the modes picked at block 6
z, free_error, shared_error = [], [], []
for b in range(len(blocks)):
    phi = train.encode_pairs(kae, torch.stack([states[:, 0], states[:, b]], dim=1))[:, 1]
    z.append(to_complex(phi))
    spread = (states[:, b] - states[:, b].mean(dim=0)).pow(2).sum()
    shared_error.append(((kae.decode(phi) - states[:, b]).pow(2).sum() / spread).item())
    free_error.append(((kae.decode(kae.encode(states[:, b])) - states[:, b]).pow(2).sum() / spread).item())
z = torch.stack(z, dim=1)
print("block:                                        ", "  ".join(f"{b:5d}" for b in blocks))
print("reconstruction error, own modes:              ", "  ".join(f"{v:5.3f}" for v in free_error))
print("reconstruction error, block 6's modes:        ", "  ".join(f"{v:5.3f}" for v in shared_error))
size = states.pow(2).sum(dim=-1).mean(dim=0).sqrt()
print("size of the standardized state:               ", "  ".join(f"{v:5.1f}" for v in size.tolist()))
centre = states.mean(dim=0)
cosine = torch.nn.functional.cosine_similarity(centre, centre[:1], dim=-1)
print("cosine of the average state with block 6's:   ", "  ".join(f"{v:5.2f}" for v in cosine.tolist()))

kept = z[:, 0] != 0
modes = [m for m in (kept.sum(dim=0) >= MIN_IMAGES).nonzero()[:, 0].tolist() if m > 0]
multiplier, follows, path = [], [], []
for m in modes:
    on = kept[:, m]
    start = z[on, 0, m]
    mu = (z[on, :, m] * start.conj()[:, None]).sum(dim=0) / start.abs().pow(2).sum()
    multiplier.append(mu)
    # How much of the mode's value at each block is its block-6 value times one number
    follows.append(1 - (z[on, :, m] - mu * start[:, None]).abs().pow(2).sum(dim=0) / z[on, :, m].abs().pow(2).sum(dim=0))
    # The same with the number set by the eigenvalue alone: λ to the power of the share of the depth covered
    share = torch.linspace(0, 1, len(blocks))
    steady = torch.polar(lam[m].abs() ** share, torch.angle(lam[m]) * share)
    path.append(1 - (z[on, :, m] - steady * start[:, None]).abs().pow(2).sum(dim=0) / z[on, :, m].abs().pow(2).sum(dim=0))
multiplier, follows, path = torch.stack(multiplier), torch.stack(follows), torch.stack(path)
modes = torch.tensor(modes)
turn = torch.from_numpy(np.unwrap(torch.angle(multiplier).numpy(), axis=1))
growth = multiplier.abs().log()
feature = ~torch.isin(modes, torch.tensor(OFFSET + GROWTH + INERT))
print(f"\n{len(modes)} modes kept on at least {MIN_IMAGES} of {len(states)} images, {int(feature.sum())} of them feature modes")
quartiles = torch.tensor([0.25, 0.5, 0.75])


def show(label, values):
    rows = values.quantile(quartiles, dim=0)
    for name, row in zip(["25%", "median", "75%"], rows):
        print(f"  {label:46s} {name:7s}", "  ".join(f"{v:6.2f}" for v in row.tolist()))


print("\n== feature modes, by block:", "  ".join(f"{b:6d}" for b in blocks))
show("share of the value explained by one multiplier", follows[feature])
show("share explained by λ^(share of depth)", path[feature])
turning = feature & (turn[:, -1].abs() > np.radians(5))
show(f"share of the final turn done ({int(turning.sum())} modes over 5°)", (turn / turn[:, -1:])[turning])
growing = feature & (growth[:, -1].abs() > np.log(1.5))
show(f"share of the final growth done, log scale ({int(growing.sum())})", (growth / growth[:, -1:])[growing])
show("turn so far, degrees (absolute)", torch.rad2deg(turn[feature]).abs())
show("size of the multiplier so far", multiplier[feature].abs())

gap = torch.rad2deg(turn[:, -1]) - torch.rad2deg(torch.angle(lam[modes]))
gap = (gap + 180) % 360 - 180
print(f"\nmultiplier at block 12 against the eigenvalue: |angle gap| median {gap[feature].abs().median():.1f}°, "
      f"size ratio median {(multiplier[feature, -1].abs() / lam[modes][feature].abs()).median():.2f}")

# When is a mode built? The depth (in blocks, interpolated) at which half of its final turn, or growth, is done
def get_half_depth(progress):
    progress = progress.numpy()
    depths = np.full(len(progress), np.nan)
    for i, row in enumerate(progress):
        above = np.nonzero(row >= 0.5)[0]
        if len(above) and above[0] > 0:
            first = above[0]
            depths[i] = blocks[first - 1] + (0.5 - row[first - 1]) / (row[first] - row[first - 1])
    return torch.from_numpy(depths)


half_turn = get_half_depth((turn / turn[:, -1:])[turning])
half_growth = get_half_depth((growth / growth[:, -1:])[growing])
for label, values in [("half of the turn", half_turn), ("half of the growth", half_growth)]:
    ok = values[~values.isnan()]
    print(f"block by which {label} is done: 10/25/50/75/90% of modes = {[round(v, 2) for v in ok.quantile(torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]).double()).tolist()]} ({len(ok)} modes)")
steps = (turn[turning][:, 1:] - turn[turning][:, :-1]) * torch.sign(turn[turning][:, -1:])
print(f"modes whose turn never goes backwards by over 2°: {(steps > -np.radians(2)).all(dim=1).float().mean():.2f}")
steps = (growth[growing][:, 1:] - growth[growing][:, :-1]) * torch.sign(growth[growing][:, -1:])
print(f"modes whose growth never goes backwards by over 5%: {(steps > -np.log(1.05)).all(dim=1).float().mean():.2f}")

print("\n== always-on modes: turn so far in degrees | size of the multiplier, by block")
for name, group in [("offset", OFFSET), ("growth", GROWTH)]:
    for m in group:
        row = (modes == m).nonzero()
        if len(row):
            i = row[0, 0]
            print(f"  {name} {m:4d} (λ {lam[m].abs():5.2f} at {np.degrees(torch.angle(lam[m]).item()):6.1f}°): "
                  + "  ".join(f"{np.degrees(t):6.1f}" for t in turn[i].tolist()) + "  |  " + "  ".join(f"{v:5.2f}" for v in multiplier[i].abs().tolist()))
torch.save({"modes": modes, "blocks": blocks, "multiplier": multiplier, "follows": follows, "path": path, "feature": feature,
            "turning": turning, "growing": growing, "eigenvalues": lam[modes]}, "depth_clock.pt")
