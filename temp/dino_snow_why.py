"""Why the KAE's step fails on snowy images: is it the modes picked at block 6, or the one eigenvalue per mode? CPU only."""
import math
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv
from safetensors.torch import load_file

sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
import train  # noqa: E402

load_dotenv("/home/nsa325/work/koopmann/.env")
torch.set_grad_enabled(False)
torch.set_num_threads(32)
pd.set_option("display.width", 250)
SP = Path(__file__).parent
NAME, ENCODER = "im_dino_w21_wd", "dinov3-vitb16"
FOLDERS = {"armytank": 847, "carwheel": 479, "firetruck": 555, "motorscooter": 670, "racingcar": 751, "schoolbus": 779, "trafficlight": 920}
SNOW_MODES = [667, 644, 37, 247, 7, 461]

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


def to_complex(phi):
    return torch.cat([phi[..., :1].to(torch.cfloat), torch.view_as_complex(phi[..., 1:].unflatten(-1, (-1, 2)).contiguous())], dim=-1)


def to_latent(z):
    return torch.cat([z[..., :1].real, torch.view_as_real(z[..., 1:]).flatten(-2)], dim=-1)


def decode(z):
    return torch.cat([kae.decode(to_latent(batch)) for batch in z.split(5_000)])


saved = torch.load(SP / "dino_explore.pt", weights_only=False)
snowy_x, snowy_y = (saved["snowy_raw"] - mean) / std, saved["snowy_y"]
normal = torch.isin(val_y, torch.tensor(list(FOLDERS.values())))
x, y = torch.cat([val_x[normal], snowy_x]), torch.cat([val_y[normal], snowy_y])
is_snowy = torch.cat([torch.zeros(int(normal.sum()), dtype=torch.bool), torch.ones(len(snowy_x), dtype=torch.bool)])

z = to_complex(train.encode_pairs(kae, x))
z6, z12 = z[:, 0], z[:, 1]
kept = z6 != 0
print("modes kept per image: min", int(kept.sum(dim=1).min()), "median", int(kept.sum(dim=1).median()), "max", int(kept.sum(dim=1).max()), "| real mode kept on", kept[:, 0].double().mean().item())


def report(label, z_end):
    right = head.classify(decode(z_end)) == y
    print(f"{label:70s} snowy {right[is_snowy].double().mean():.3f}  normal {right[~is_snowy].double().mean():.3f}")
    return right


print("\n--- which part of the step fails ---")
base = report("K as trained (λ·z6)", lam * z6)
report("block 12 encoded on the modes picked at block 6 (no K)", z12)
report("block 12 encoded with its own TopK (reconstruction)", to_complex(kae.encode(x[:, 1])))
for count in [1, 2, 3, 4, 6]:
    mixed = lam * z6
    mixed[:, SNOW_MODES[:count]] = z12[:, SNOW_MODES[:count]]
    report(f"K as trained, but the top {count} snow modes take their true block-12 value", mixed)
# The other way round: true block-12 values everywhere except the snow modes, which follow K
for count in [4, 6]:
    mixed = z12.clone()
    mixed[:, SNOW_MODES[:count]] = (lam * z6)[:, SNOW_MODES[:count]]
    report(f"true block-12 values, but the top {count} snow modes follow K", mixed)
# Gains only: keep K's direction for every mode, but the true size, and the reverse
true_size = z12.abs() * torch.exp(1j * torch.angle(lam * z6))
report("K's turn, true size, every mode", torch.where(kept, true_size, torch.zeros_like(true_size)))
true_turn = (lam * z6).abs() * torch.exp(1j * torch.angle(z12))
report("K's size, true turn, every mode", torch.where(kept, true_turn, torch.zeros_like(true_turn)))

print("\n--- how the snow modes really move, by group of images ---")
full = to_complex(torch.cat([train.encode_pairs(kae, batch) for batch in val_x.split(2_500)]))
groups = {
    "snowy vehicles": (z6[is_snowy], z12[is_snowy]),
    "normal vehicles": (z6[~is_snowy], z12[~is_snowy]),
    "val snowplow + snowmobile": (full[(val_y == 803) | (val_y == 802), 0], full[(val_y == 803) | (val_y == 802), 1]),
    "all val": (full[:, 0], full[:, 1]),
}
rows = []
for mode in SNOW_MODES:
    for label, (a, b) in groups.items():
        on = a[:, mode] != 0
        a_, b_ = a[on, mode], b[on, mode]
        pooled = (b_ * a_.conj()).sum()
        rows.append(
            {
                "mode": mode,
                "images": label,
                "kept": int(on.sum()),
                "|λ|": lam[mode].abs().item(),
                "angle": math.degrees(torch.angle(lam[mode]).item()),
                "median |z6|": a_.abs().median().item(),
                "measured gain": (b_.abs() / a_.abs()).median().item(),
                "measured turn": math.degrees(torch.angle(pooled).item()),
                "turn agreement": (pooled.abs() / (b_.abs() * a_.abs()).sum()).item(),
            }
        )
print(pd.DataFrame(rows).set_index(["mode", "images"]).round(2).to_string())

print("\n--- all validation: where the gap to DINOv3 opens ---")
f6, f12 = full[:, 0], full[:, 1]
true_answers = head.classify(val_x[:, 1])


def score(label, z_end):
    found = head.classify(decode(z_end))
    print(f"{label:62s} accuracy {(found == val_y).double().mean():.4f}  agreement {(found == true_answers).double().mean():.4f}")


def mix(z_start, z_end, part):
    predicted = lam * z_start
    mixed = z_end.abs() * torch.exp(1j * torch.angle(predicted)) if part == "size" else predicted.abs() * torch.exp(1j * torch.angle(z_end))
    return torch.where(z_start != 0, mixed, torch.zeros_like(mixed))


print(f"DINOv3 block 12: accuracy {(true_answers == val_y).double().mean():.4f}")
score("block 12 through the KAE, its own modes (no K)", to_complex(torch.cat([kae.encode(b) for b in val_x[:, 1].split(5_000)])))
score("block 12 through the KAE, modes picked at block 6 (no K)", f12)
score("K's size, true turn", mix(f6, f12, "turn"))
score("K's turn, true size", mix(f6, f12, "size"))
score("K as trained", lam * f6)
# Split by group of modes: true values for one group, K for the rest
usage = (f6 != 0).double().mean(dim=0)
big = lam.abs() > 5
groups = {"always-on modes (16)": usage >= 0.99, "feature modes, |λ| > 5": (usage < 0.99) & big, "feature modes, |λ| ≤ 5": (usage < 0.99) & ~big}
for label, mask in groups.items():
    mixed = lam * f6
    mixed[:, mask] = f12[:, mask]
    score(f"true values for the {label} ({int((mask & (usage > 0)).sum())} used), K for the rest", mixed)
