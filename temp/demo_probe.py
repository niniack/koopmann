"""Numbers for the demo notebook, before it is written (CPU)."""
import sys, time
sys.path.insert(0, "/home/nsa325/work/koopmann/second_analysis")
import torch
from sklearn.metrics import roc_auc_score
from demo_utils import *
torch.set_num_threads(24)
states, labels = load_val()
kae = Kae("im_iso_2x4k")
z6 = kae.encode(states)
answers, clip = kae.answers(z6), kae.clip_answers(states)
print("accuracy CLIP", (clip == labels).float().mean().item(), "KAE", (answers == labels).float().mean().item(), "agree", (answers == clip).float().mean().item())
size, angle = kae.eigenvalues.abs(), kae.eigenvalues.angle().rad2deg().abs()
count = (z6 != 0).sum(dim=0)
live = count >= 200; live[0] = False
usage = count / len(z6)
print("live", int(live.sum()), "always on (>=99%)", int((usage >= .99).sum()))
tie = class_tie(z6.abs(), labels)
feature = live & (usage < 0.5)
print("size vs class tie (live):", rank_correlation(size[live], tie[live]), "usage fixed:", rank_correlation(size[live], tie[live], (usage[live],)), "| angle vs tie:", rank_correlation(angle[live], tie[live]), "| usage vs tie", rank_correlation(usage[live], tie[live]))
print("size quantiles", size[live].quantile(torch.tensor([0., .1, .5, .9, 1.])).tolist(), "tie quantiles", tie[live].quantile(torch.tensor([0., .1, .5, .9, 1.])).tolist())
order = size.clone(); order[~feature] = float("nan")
for label, pick in [("smallest", torch.where(feature, size, torch.inf).argsort()[:6]), ("largest", torch.where(feature, size, -torch.inf).argsort(descending=True)[:6])]:
    print(label, [(int(m), round(size[m].item(), 2), round(tie[m].item(), 3), round(usage[m].item(), 3)) for m in pick])
# snow
snow_states, snow_labels, snow_paths = load_snowy()
normal = torch.isin(labels, torch.tensor(list(SNOW_FOLDERS.values())))
test_states, test_labels = torch.cat([states[normal], snow_states]), torch.cat([labels[normal], snow_labels])
is_snowy = torch.arange(len(test_labels)) >= int(normal.sum())
test_z6 = kae.encode(test_states)
before = kae.answers(test_z6) == test_labels
clip_right = kae.clip_answers(test_states) == test_labels
print("CLIP snowy/normal", clip_right[is_snowy].float().mean().item(), clip_right[~is_snowy].float().mean().item(), "KAE", before[is_snowy].float().mean().item(), before[~is_snowy].float().mean().item())
activity = test_z6.abs()
def get_aucs(rows):
    return torch.tensor([roc_auc_score(is_snowy[rows].numpy(), activity[rows, m].numpy()) if activity[rows, m].any() else 0.5 for m in range(activity.shape[1])])
half = torch.arange(len(test_labels)) % 2 == 0
for pick, name in [(half, "pick even"), (~half, "pick odd")]:
    aucs = get_aucs(pick)
    top = aucs.argsort(descending=True)
    print(name, "top modes", [(int(m), round(aucs[m].item(), 3), round(size[m].item(), 2), round(angle[m].item(), 1), round(usage[m].item(), 3)) for m in top[:4]])
    for n in [1, 2, 4]:
        after = kae.answers(test_z6, top[:n].tolist()) == test_labels
        t = ~pick
        print(f"   remove top {n}: held-out snowy {before[t & is_snowy].float().mean():.3f} -> {after[t & is_snowy].float().mean():.3f} (CLIP {clip_right[t & is_snowy].float().mean():.3f}); normal {before[t & ~is_snowy].float().mean():.3f} -> {after[t & ~is_snowy].float().mean():.3f}")
aucs = get_aucs(torch.ones_like(half)); top = aucs.argsort(descending=True)
print("all images top", top[:4].tolist())
for n in [1, 2]:
    for f in [1, .75, .5, .25, 0]:
        after = kae.answers(test_z6, top[:n].tolist(), f) == test_labels
        print(f"   top {n} factor {f}: snowy {after[is_snowy].float().mean():.3f} normal {after[~is_snowy].float().mean():.3f}")
    after = kae.answers(z6, top[:n].tolist())
    print(f"   cost on all val, top {n}: accuracy {(answers == labels).float().mean():.4f} -> {(after == labels).float().mean():.4f}; fixed {((after == labels) & (answers != labels)).sum().item()}, broken {((after != labels) & (answers == labels)).sum().item()}; images that keep one {(z6[:, top[:n]] != 0).any(dim=1).sum().item()}")
# angle
a, b = Kae("im_realstart_2x4k_wd"), Kae("im_realstart_2x4k_wd_s1")
za, zb = a.encode(states), b.encode(states)
ma, mb, alike = match_modes(za, zb)
sa, sb = a.eigenvalues.abs()[ma], b.eigenvalues.abs()[mb]
aa, ab = a.eigenvalues.angle().rad2deg().abs()[ma], b.eigenvalues.angle().rad2deg().abs()[mb]
print("pairs", len(ma), "size", rank_correlation(sa, sb), "angle", rank_correlation(aa, ab), "angle | size", rank_correlation(aa, ab, (sa, sb)), "shuffled", rank_correlation(aa, ab[torch.randperm(len(ab))]))
print("imag at block 6 max", za.imag.abs().max().item())
for k, z in [(a, za), (b, zb)]:
    live = ((z != 0).sum(dim=0) >= 200).nonzero()[:, 0]
    x = z.real[:, live].double(); x = x - x.mean(dim=0)
    y = ((states - k.mean) / k.std); y = (y - y.mean(dim=0)).double()
    gram = x.T @ x; gram = gram + 1e-3 * gram.diagonal().mean() * torch.eye(len(gram), dtype=torch.double)
    effect = torch.linalg.solve(gram, x.T @ y.flatten(1)).view(len(live), 2, -1)
    persistence = torch.nn.functional.cosine_similarity(effect[:, 0], effect[:, 1], dim=-1).float()
    keep = live > 0
    s, g, u = k.eigenvalues.abs()[live], k.eigenvalues.angle().rad2deg().abs()[live], (z != 0).float().mean(dim=0)[live]
    print("persistence vs angle", rank_correlation(g[keep], persistence[keep]), "fixed size, usage", rank_correlation(g[keep], persistence[keep], (s[keep], u[keep])), "n", int(keep.sum()), "live includes mode 0:", bool((live == 0).any()))
print("done")
