"""How often does the real DINOv3 (with its ImageNet probe) mix up ImageNet's antelope-like classes, and does the KAE do worse there?"""

import os
import sys

import open_clip
import torch
from dotenv import load_dotenv
from safetensors.torch import load_file

sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
load_dotenv("/home/nsa325/work/koopmann/.env")
import train  # noqa: E402

torch.set_grad_enabled(False)
ENCODER, RUN = "dinov3-vitb16", "im_dino_w21_wd_e30"
GROUP = list(range(345, 354))  # ox, water buffalo, bison, ram, bighorn, ibex, hartebeest, impala, gazelle

folder = os.getenv("WEIGHTS_CACHE") + "/tokae"
stats = torch.load(f"{folder}/{RUN}_stats.pt")
states, labels = train.get_states(ENCODER, "imagenet", "val", "cpu", [5, 11], tokens="cls_patch_std", extract=False)
states = (states.float() - stats["mean"]) / stats["std"]
head = train.DinoHead(ENCODER, stats["mean"], stats["std"], "cpu", zero_shot=True)
kae = train.JordanKoopmanAutoencoder(256, shared_modes=True, in_features=2304, latent_features=2049, hidden_config=stats["hidden_config"], bias=True, batchnorm=False, nonlinearity="leaky_relu")
kae.load_state_dict(load_file(f"{folder}/{RUN}.safetensors"))
kae.eval()

inside = torch.isin(labels, torch.tensor(GROUP))
states, labels = states[inside], labels[inside]
answers = {"real DINOv3": head.classify(states[:, 1]), "KAE": head.classify(kae.decode(kae.koopman_forward(train.encode_pairs(kae, states)[:, 0], 1)))}
for name, answer in answers.items():
    wrong = answer != labels
    in_group = torch.isin(answer, torch.tensor(GROUP)) & wrong
    print(f"{name}: accuracy on the 9 classes {100 * (~wrong).float().mean():.1f}% | wrong but inside the group {100 * in_group.float().mean():.1f}% | wrong, outside {100 * (wrong & ~in_group).float().mean():.1f}%")
    for c in GROUP:
        rows = labels == c
        print(f"   {open_clip.IMAGENET_CLASSNAMES[c]:14s} {100 * (answer[rows] == c).float().mean():.0f}%")
