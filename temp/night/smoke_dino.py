"""CPU smoke test of the DINOv3 path in train.py: extraction against a direct computation, the head on true states, then one training epoch on fake states."""
import os, sys, tempfile, torch
sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
os.environ["WANDB_MODE"] = "disabled"
import train
from dotenv import load_dotenv
load_dotenv("/home/nsa325/work/koopmann/.env")
from transformers import AutoModel
torch.manual_seed(0)
cpu = torch.device("cpu")

class Few(train.ImageNet):
    def __init__(self, split, transform):
        super().__init__(split, transform)
        self.data = self.data.select(range(0, 48 * 1000, 1000))
train.ImageNet = Few

# One training epoch on fake states
tmp = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(__file__)))
os.makedirs(f"{tmp}/tokae")
os.environ["WEIGHTS_CACHE"] = tmp
train.get_device = lambda: cpu
def fake_states(encoder, dataset, split, device, blocks, views=1, tokens="cls", extract=True):
    n = 20_000 if split == "train" else 2_000
    return torch.randn(n, 2, 2304) * 3 + 1, torch.randint(0, 1000, (n,))
train.get_states = fake_states
train.main(encoder="dinov3-vitb16", dataset="imagenet", start_block=6, end_block=12, views=3, num_epochs=1, lr=0.0075, adam_lr=0.00075, lambda_variance=0, lambda_covariance=0,
           lambda_isometric=0.1, ae_dim=2049, ae_hidden=256, ae_layers=2, lambda_embedding=1, lambda_kl=0.03, jordan=True, top_k=256, shared_modes=True,
           tokens="cls_patch_std", name="smoke_dino", batch_size=1024)
print("SMOKE OK", sorted(os.listdir(f"{tmp}/tokae")))
import shutil; shutil.rmtree(tmp)
