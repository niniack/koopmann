"""How far do single-mode edits move the KAE's predicted block-12 state, next to the gap between the prediction and the real state?"""
import math, os
from pathlib import Path
import torch
import inversion
from inversion import train, to_complex, to_latent, turn

torch.set_num_threads(4)
torch.set_grad_enabled(False)
kae, stats = inversion.load_kae()
head = train.DinoHead(inversion.ENCODER, stats["mean"], stats["std"], torch.device("cpu"), zero_shot=True)
root = Path(os.getenv("DATASETS_CACHE"))
val = torch.load(root / "imagenet" / f"clip_cls_patch_std_{inversion.ENCODER}_val_blocks_5_11.pt")
lam = torch.cat([kae.real_eigenvalue.to(torch.cfloat), torch.complex(kae.block_real, kae.block_imag)])
import open_clip
names = open_clip.IMAGENET_CLASSNAMES
IMAGES = {42917: 667, 39437: 667, 12995: 644, 15059: 247, 26388: 37, 32814: 71, 46610: 535, 33079: 211, 27631: 211}
for index, mode in IMAGES.items():
    x = ((val["states"][index].float() - stats["mean"]) / stats["std"])[None]
    z = to_complex(train.encode_pairs(kae, x))[0]
    z6 = z[0]
    pred = kae.decode(to_latent(lam * z6))[None][0]
    recon = kae.decode(to_latent(z[1]))
    real = x[0, 1]

    def gap(a, b, part=slice(0, 2304)):
        return ((a[part] - b[part]).pow(2).sum() / real[part].pow(2).sum()).item()

    line = f"{index} {names[val['labels'][index]]}: real-recon {gap(real, recon):.3f}, real-pred {gap(real, pred):.3f} (cls {gap(real, pred, slice(0, 768)):.3f})"
    for name, factor in {"removed": 0, "x2": 2, "+90": turn(90), "180": -1}.items():
        edited = z6.clone()
        edited[mode] = edited[mode] * factor
        out = kae.decode(to_latent(lam * edited))
        answer = names[head.classify(out[None]).item()]
        line += f" | {name}: {gap(pred, out):.3f} (cls {gap(pred, out, slice(0, 768)):.3f}) -> {answer}"
    print(line)
    print("   answers: real", names[head.classify(real[None]).item()], "| pred", names[head.classify(pred[None]).item()], "| kept modes", int((z6 != 0).sum()), "| mode's share of predicted latent energy", round(((lam * z6)[mode].abs().pow(2) / (lam * z6).abs().pow(2).sum()).item(), 4))
