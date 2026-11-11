"""
The DINOv3 KAE's eigenvalues, mode usage and latents on the 50,000 validation images, saved like inside_pass.py does.

NOTE: DINOv3's head needs the `transformers` package, which this environment does not have. So no accuracy is
measured here, and the review quotes this model's accuracy from its training log.
"""

import torch

from inside_load import DATA, NAMES, get_eigenvalues, get_encoder, load_model, load_val, train  # isort: skip  (sets the path)

import angle_tests as tests

NAME = "im_dino_w21_wd_e30"
model, stats = load_model(NAME)
states, _, labels = load_val(get_encoder(NAME))
x = (states - stats["mean"]) / stats["std"]
# shape: [images, 2 (block 6, block 12), latent]
phi = torch.cat([train.encode_pairs(model, part) for part in x.split(5_000)])
z6 = phi[:, 0]
results = {"name": NAME, "label": NAMES[NAME], "images": len(x), "eigenvalues": get_eigenvalues(model), "usage": (tests.to_complex(z6) != 0).sum(dim=0), "real_start": bool(stats.get("real_start", False))}
torch.save(results, DATA / f"inside_{NAME}.pt")
torch.save({"z6": z6, "z12": model.koopman_forward(z6, 1), "end": phi[:, 1]}, DATA / f"inside_latents_{NAME}.pt")
print(NAME, results["images"], "images,", int((results["usage"][1:] >= 50).sum()), "modes used by 50+ images")
