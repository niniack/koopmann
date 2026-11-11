import json, os, sys
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
os.chdir("/home/nsa325/work/koopmann/second_analysis")
nb = json.load(open("jordan_topk_modes.ipynb"))
cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
for i in [0, 1, 2]:
    exec(cells[i], globals())
torch.set_num_threads(48)
SP = sys.argv[1]
saved = torch.load(f"{SP}/shift_states.pt"); data, indices = saved["data"], saved["indices"]
labels = val_y[indices]
original = data["original"]
base = get_activity(original[:, 0])
acc = lambda model, s: (classify(predict(model, s)) == labels).double().mean().item()


def remove_modes(modes, alpha=0.0):
    edited = copy.deepcopy(kae)
    for mode in modes:
        if mode == 0:
            edited.real_eigenvalue.mul_(alpha)
        else:
            edited.block_real[mode - 1].mul_(alpha)
            edited.block_imag[mode - 1].mul_(alpha)
    return edited


for name in ["grayscale", "sepia", "motion blur"]:
    shifted = data[name]
    act = get_activity(shifted[:, 0])
    # modes switched on by the shift: on far more often for shifted images than for the same images unshifted
    gain = (act > 0).double().mean(0) - (base > 0).double().mean(0)
    order = gain.argsort(descending=True)
    lost = (-gain).argsort(descending=True)
    same_modes = ((act > 0) == (base > 0)).double().mean().item()
    print(f"\n{name}: KAE shifted {acc(kae, shifted[:, 0]):.3f}, original {acc(kae, original[:, 0]):.3f} | share of mode on/off decisions unchanged by the shift: {same_modes:.3f}")
    print("  modes switched on most:", [(int(m), round(gain[m].item(), 2)) for m in order[:6]], "| switched off most:", [(int(m), round(gain[m].item(), 2)) for m in lost[:4]])
    for n in [1, 3, 10, 30]:
        edited = remove_modes(order[:n].tolist())
        print(f"  remove top {n:2d} switched-on modes: shifted {acc(edited, shifted[:, 0]):.3f}, original {acc(edited, original[:, 0]):.3f}")
    # upper bound for a latent-space fix: the shifted image's latent, but with the original image's mode selection
    phi_s, phi_o = kae.encode(shifted[:, 0]), kae.encode(original[:, 0])
    print(f"  reference, decode K applied to the ORIGINAL image's latent: {acc(kae, original[:, 0]):.3f}; latent cosine shifted vs original: {torch.nn.functional.cosine_similarity(phi_s, phi_o).mean():.3f}")
    print(f"  CLIP: original {(classify(original[:, 1]) == labels).double().mean():.3f}, shifted {(classify(shifted[:, 1]) == labels).double().mean():.3f}; KAE agrees with CLIP on shifted: {(classify(predict(kae, shifted[:, 0])) == classify(shifted[:, 1])).double().mean():.3f} (original {(classify(predict(kae, original[:, 0])) == classify(original[:, 1])).double().mean():.3f})")
