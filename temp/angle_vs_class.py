"""
Do the modes that K turns hold objects, and the modes that K only rescales hold low-level qualities?

The classic finding on depth (Network Dissection): early layers hold colours and textures, late layers hold objects.
If the eigenvalue angle means "how much blocks 7-12 rework this feature", modes with a large angle should be tied to
the class of the image more than modes with a small angle. Class tie of a mode = share of the image-to-image variation
of its block-6 size that the class label explains (0 = no tie, 1 = the class decides it). With 50 validation images per
class, a mode with no tie scores about 0.02.

Usage: angle_vs_class.py dino | angle_vs_class.py <CLIP run name> <real_start: 0 or 1>
       angle_vs_class.py <weights name of a cls_train.py model>        such as lift_attend_step62500
"""

import os
import sys

import torch

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 2)))
torch.set_grad_enabled(False)
if sys.argv[1] == "dino":
    import inversion

    kae, stats = inversion.load_kae()
    cache = torch.load("dino_explore.pt", weights_only=False)
    z, lam = cache["z"], cache["eigenvalues"]
    from pathlib import Path

    labels = torch.load(Path(os.getenv("DATASETS_CACHE")) / "imagenet" / f"clip_cls_patch_std_{inversion.ENCODER}_val_blocks_5_11.pt")["labels"]
    name = "im_dino_w21_wd"
else:
    sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
    from dotenv import load_dotenv
    from safetensors.torch import load_file

    load_dotenv("/home/nsa325/work/koopmann/.env")
    import train

    name = sys.argv[1]
    save_dir = os.getenv("WEIGHTS_CACHE") + "/tokae"
    stats = torch.load(f"{save_dir}/{name.partition('_step')[0]}_stats.pt")
    if "arm" in stats:
        import run_codes

        _, _, labels, z, lam = run_codes.get_codes(name)
    else:
        states, labels = train.get_states("ViT-B-32", "imagenet", "val", "cpu", [5, 11], tokens="cls_patch_std", extract=False)
        states = (states.float() - stats["mean"]) / stats["std"]
        kae = train.JordanKoopmanAutoencoder(256, shared_modes=True, real_start=sys.argv[2] == "1", in_features=states.shape[-1], latent_features=2049, hidden_config=stats["hidden_config"], bias=True, batchnorm=False, nonlinearity="leaky_relu")
        kae.load_state_dict(load_file(f"{save_dir}/{name}.safetensors"))
        kae.eval()
        to_complex = lambda latent: torch.cat([latent[..., :1].to(torch.cfloat), torch.complex(latent[..., 1::2], latent[..., 2::2])], dim=-1)
        z = torch.cat([to_complex(train.encode_pairs(kae, states[first : first + 5000])) for first in range(0, len(states), 5000)])
        lam = torch.cat([kae.real_eigenvalue.to(torch.cfloat), torch.complex(kae.block_real, kae.block_imag)])


def get_class_tie(values):
    """Share of the variance of each column that the class label explains. values: [images, modes] real"""

    one_hot = torch.nn.functional.one_hot(labels, 1000).double()
    counts = one_hot.sum(dim=0)
    values = values.double()
    class_means = (one_hot.T @ values) / counts[:, None]
    between = (counts[:, None] * (class_means - values.mean(dim=0)).pow(2)).sum(dim=0)
    return (between / (values - values.mean(dim=0)).pow(2).sum(dim=0).clamp(min=1e-12)).float()


z6, z12 = z[:, 0], z[:, 1]
kept = z6 != 0
usage = kept.float().mean(dim=0)
modes = ((kept.sum(dim=0) >= 200) & (usage < 0.99)).nonzero()[:, 0]
tie_start, tie_end = get_class_tie(z6.abs())[modes], get_class_tie(z12.abs())[modes]
# The same for the mode's two coordinates taken as they are, so a mode whose phase holds the class is counted too
tie_plane = (get_class_tie(z6.real)[modes] * z6.real.var(dim=0)[modes] + get_class_tie(z6.imag)[modes] * z6.imag.var(dim=0)[modes]) / (z6.real.var(dim=0)[modes] + z6.imag.var(dim=0)[modes])
angle, size, use = torch.rad2deg(torch.angle(lam[modes])).abs(), lam[modes].abs(), usage[modes]
correlate = lambda a, b: torch.corrcoef(torch.stack([a.float(), b.float()]))[0, 1].item()
rank = lambda a: a.argsort().argsort().float()
print(f"== {name}: {len(modes)} feature modes (kept on at least 200 validation images, not on every image)")
print(f"class tie of the block-6 size at 10/50/90% of modes: {[round(v, 3) for v in tie_start.quantile(torch.tensor([0.1, 0.5, 0.9])).tolist()]}; of the true block-12 size: {[round(v, 3) for v in tie_end.quantile(torch.tensor([0.1, 0.5, 0.9])).tolist()]}")
print("rank correlations over modes:")
print(f"  |angle of λ| with the class tie at block 6: {correlate(rank(angle), rank(tie_start)):.3f}; with the class tie of both coordinates at block 6: {correlate(rank(angle), rank(tie_plane)):.3f}; with the class tie at block 12: {correlate(rank(angle), rank(tie_end)):.3f}")
print(f"  size of λ with the class tie at block 6: {correlate(rank(size), rank(tie_start)):.3f}; at block 12: {correlate(rank(size), rank(tie_end)):.3f}")
print(f"  usage with the class tie at block 6: {correlate(rank(use), rank(tie_start)):.3f}; usage with |angle|: {correlate(rank(use), rank(angle)):.3f}; usage with size of λ: {correlate(rank(use), rank(size)):.3f}; |angle| with size of λ: {correlate(rank(angle), rank(size)):.3f}")
print("by |angle of λ|: modes | median size of λ | median usage | median class tie at block 6 | at block 12 | median gain of the class tie from 6 to 12")
for low, high in [(0, 2), (2, 5), (5, 10), (10, 20), (20, 45), (45, 180)]:
    rows = (angle >= low) & (angle < high)
    if rows.sum() == 0:
        continue
    print(f"  {low:3d}-{high:3d}°: {rows.sum().item():4d} | {size[rows].median():5.2f} | {use[rows].median():.3f} | {tie_start[rows].median():.3f} | {tie_end[rows].median():.3f} | {(tie_end[rows] - tie_start[rows]).median():+.3f}")
print("by size of λ: modes | median |angle| | median usage | median class tie at block 6 | at block 12")
for part, rows in enumerate(size.argsort().chunk(5)):
    print(f"  fifth {part + 1} (size {size[rows].min():.2f} to {size[rows].max():.2f}): {len(rows):4d} | {angle[rows].median():5.1f}° | {use[rows].median():.3f} | {tie_start[rows].median():.3f} | {tie_end[rows].median():.3f}")
torch.save({"modes": modes, "angle": angle, "size": size, "usage": use, "tie start": tie_start, "tie end": tie_end, "tie plane": tie_plane}, f"angle_vs_class_{name}.pt")
print("done")
