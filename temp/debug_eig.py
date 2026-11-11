import json, warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
print("torch", torch.__version__)

# Minimal repro, no project code
torch.manual_seed(0)
base = torch.randn(64, 64, device="cuda", dtype=torch.float64)
for label, make in [("transposed view (.T.double())", lambda: base.float().T.double()),
                    ("contiguous", lambda: base.clone()),
                    ("transposed .contiguous()", lambda: base.float().T.double().contiguous())]:
    x = make(); before = x.clone()
    print(f"{label:<32} strides {x.stride()}", end="  ")
    torch.linalg.svd(x); a = (x - before).abs().max().item()
    torch.linalg.eig(x); b = (x - before).abs().max().item()
    print(f"changed after svd {a:.2e}, after eig {b:.2e}")
x = base.float().T.double().cpu(); before = x.clone(); torch.linalg.eig(x)
print("cpu transposed: changed after eig", (x - before).abs().max().item())

# The real operator, inside the notebook's own flow
nb = json.load(open("15_data_svd.ipynb"))
code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
ns = {}
for src in code[:4]:
    exec(src, ns)
kae = ns["kae"]; operator = ns["operator"]; val_phi = ns["val_phi"]
fresh = kae.koopman_weights.detach().T.double().contiguous()
print("operator vs fresh K: max diff", (operator - fresh).abs().max().item(), " K scale", fresh.abs().max().item())
get_class_accs, decode = ns["get_class_accs"], ns["decode"]
print("acc, stored operator", get_class_accs(decode(val_phi @ operator)).mean().item())
print("acc, fresh K        ", get_class_accs(decode(val_phi @ fresh)).mean().item())

# Are the decompositions themselves still right?
k_left, k_values, k_right = ns["k_left"], ns["k_values"], ns["k_right"]
print("svd reconstructs fresh K: max diff", ((k_left * k_values) @ k_right - fresh).abs().max().item())
vals, vecs, inv = ns["eigenvalues"], ns["eigenvectors"], ns["eigenvectors_inverse"]
print("eig reconstructs fresh K: max diff", ((vecs * vals) @ inv - fresh.to(vecs.dtype)).abs().max().item())
empty = torch.tensor([], dtype=torch.long, device=fresh.device)
print("acc, eig edit with nothing removed", get_class_accs(ns["edit_kae_eigen"](empty)).mean().item())
print("acc, svd edit with nothing removed", get_class_accs(ns["edit_kae_svd"](empty)).mean().item())
print("kae_accs mean", ns["kae_accs"].mean().item())
