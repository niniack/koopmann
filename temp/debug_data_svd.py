import json, sys, warnings
warnings.filterwarnings("ignore", category=FutureWarning)
nb = json.load(open("15_data_svd.ipynb"))
code = [ "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
ns = {}
# setup, load, references, directions, data directions
for src in code[:5]:
    exec(src, ns)
torch = ns["torch"]; val_phi = ns["val_phi"]; operator = ns["operator"]; decode = ns["decode"]
k_mean, k_data, project_out = ns["k_mean"], ns["k_data"], ns["project_out"]
get_class_accs = ns["get_class_accs"]; kae = ns["kae"]
print("k_data", k_data.shape, k_data.dtype, "orthonormal err", (k_data.T @ k_data - torch.eye(k_data.shape[1], device=k_data.device, dtype=k_data.dtype)).abs().max().item())
out = val_phi @ operator
print("val output norm", out.norm(dim=1).mean().item(), "train mean norm", k_mean.norm().item())
print("base acc", get_class_accs(decode(out)).mean().item())
empty = torch.tensor([], dtype=torch.long, device=out.device)
same = project_out(out, k_mean, k_data, empty)
print("empty removal: max change", (same - out).abs().max().item(), "acc", get_class_accs(decode(same)).mean().item())
for idx in [0, 1, 5, 50, 300, 700, 1023]:
    removed = torch.tensor([idx], device=out.device)
    edited = project_out(out, k_mean, k_data, removed)
    change = (edited - out).norm(dim=1) / out.norm(dim=1)
    dec_change = (decode(edited) - decode(out)).norm(dim=1) / decode(out).norm(dim=1)
    print(f"dir {idx:>4}: relative latent change {change.mean().item():.2e}  relative decoded change {dec_change.mean().item():.2e}  acc {get_class_accs(decode(edited)).mean().item():.3f}")
print("edit_kae_data source check:")
removed = torch.tensor([700], device=out.device)
print(" via edit_kae_data acc", get_class_accs(ns["edit_kae_data"](removed)).mean().item())
print(" decode id", ns["edit_kae_data"].__globals__["decode"] is decode, "operator same", ns["edit_kae_data"].__globals__["operator"] is operator)
print(" k_data global shape", ns["edit_kae_data"].__globals__["k_data"].shape)
