"""Linear (ridge) and small-MLP probes from candidate block-6 summaries to block-12 CLS, scored with CLIP's zero-shot head."""
import os, sys, torch, pandas as pd, open_clip
from pathlib import Path
from dotenv import load_dotenv
load_dotenv("/home/nsa325/work/koopmann/.env")
torch.manual_seed(0); torch.set_num_threads(24)
f = torch.load(sys.argv[1]); mlp_sets = sys.argv[2].split(",") if len(sys.argv) > 2 else []
clip, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", force_quick_gelu=True, cache_dir=f"{os.getenv('HF_HOME')}/open_clip"); clip.eval()
head = torch.load(Path(os.getenv("WEIGHTS_CACHE")) / "tokae/zero_shot_head_ViT-B-32.pt")
y = f["labels"]; target = f["target"].float()
n = len(y); perm = torch.randperm(n)
fit, sel, test = perm[: int(0.7 * n)], perm[int(0.7 * n): int(0.8 * n)], perm[int(0.8 * n):]
t_mean, t_std = target[fit].mean(0), target[fit].std(0)
T = (target - t_mean) / t_std
with torch.no_grad():
    def answers(pred):
        o = clip.visual.ln_post(pred * t_std + t_mean) @ clip.visual.proj
        return ((o / o.norm(dim=-1, keepdim=True)) @ head).argmax(-1)
    clip_ans = answers(T)
SETS = {"CLS": ["cls"], "CLS + mean": ["cls", "mean"], "CLS + mean + std (current)": ["cls", "mean", "std"],
        "current + max": ["cls", "mean", "std", "max"], "current + 2x2 grid": ["cls", "mean", "std", "quad2"],
        "current + 3x3 grid": ["cls", "mean", "std", "grid3"], "current + attention-pooled": ["cls", "mean", "std", "attn"],
        "current + 2x2 grid + attention-pooled": ["cls", "mean", "std", "quad2", "attn"], "CLS + 2x2 grid + std (mean dropped)": ["cls", "quad2", "std"]}
def score(pred, idx):
    a = answers(pred)
    return {"zero-shot": (a == y[idx]).double().mean().item(), "agrees with CLIP": (a == clip_ans[idx]).double().mean().item(),
            "CLS FVU": ((pred - T[idx]).pow(2).sum() / (T[idx] - T[idx].mean(0)).pow(2).sum()).item()}
rows = {}
for label, keys in SETS.items():
    X = torch.cat([f[k].float() for k in keys], -1)
    X = (X - X[fit].mean(0)) / X[fit].std(0).clamp_min(1e-6)
    X = torch.cat([X, torch.ones(n, 1)], -1).double()
    G, B = X[fit].T @ X[fit], X[fit].T @ T[fit].double()
    best = None
    with torch.no_grad():
        for lam in [1e1, 1e2, 1e3, 1e4, 1e5]:
            W = torch.linalg.solve(G + lam * torch.eye(len(G), dtype=G.dtype), B)
            s = score((X[sel] @ W).float(), sel)
            if best is None or s["zero-shot"] > best[0]["zero-shot"]:
                best = (s, lam, W)
        rows[(label, "ridge")] = {**score((X[test] @ best[2]).float(), test), "dims": X.shape[1] - 1, "ridge strength": best[1]}
    print(label, rows[(label, "ridge")], flush=True)
    if label in mlp_sets or "all" in mlp_sets:
        Xf = X[:, :-1].float()
        net = torch.nn.Sequential(torch.nn.Linear(Xf.shape[1], 2048), torch.nn.GELU(), torch.nn.Dropout(0.2), torch.nn.Linear(2048, 768))
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05)
        best_mlp = None
        for epoch in range(40):
            net.train()
            for b in fit[torch.randperm(len(fit))].split(256):
                loss = (net(Xf[b]) - T[b]).pow(2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                s = score(net(Xf[sel]), sel)
                if best_mlp is None or s["zero-shot"] > best_mlp[0]["zero-shot"]:
                    best_mlp = (s, epoch, score(net(Xf[test]), test))
        rows[(label, "MLP")] = {**best_mlp[2], "dims": Xf.shape[1], "best epoch": best_mlp[1]}
        print(label, rows[(label, "MLP")], flush=True)
print(pd.DataFrame(rows).T.round(3).to_string())
