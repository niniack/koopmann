"""Zero-shot accuracy and agreement with CLIP for a saved KAE, on ImageNet validation."""
import os, sys
import torch
from dotenv import load_dotenv
from safetensors.torch import load_file
sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
load_dotenv("/home/nsa325/work/koopmann/.env")
from train import JordanKoopmanAutoencoder, ClipHead, get_states

name, top_k, width, patches = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "1"
save_dir = os.getenv("WEIGHTS_CACHE") + "/tokae"
stats = torch.load(f"{save_dir}/{name}_stats.pt")
val, labels = get_states("ViT-B-32", "imagenet", "val", "cpu", [5, 11], tokens="cls_patch" if patches else "cls")
val = (val - stats["mean"]) / stats["std"]
kae = JordanKoopmanAutoencoder(top_k, in_features=val.shape[-1], latent_features=width, hidden_config=[2048], bias=True, batchnorm=False, nonlinearity="leaky_relu")
kae.load_state_dict(load_file(f"{save_dir}/{name}.safetensors"))
kae = kae.cuda().eval()
head = ClipHead("ViT-B-32", stats["mean"], stats["std"], "cuda", zero_shot=True)
val, labels = val.cuda(), labels.cuda()
with torch.no_grad():
    pred = kae.decode(kae.koopman_forward(kae.encode(val[:, 0]), 1))
    answers, clip = head.classify(pred), head.classify(val[:, 1])
print(name, "zero-shot", round((answers == labels).float().mean().item(), 4), "agrees", round((answers == clip).float().mean().item(), 4), "clip", round((clip == labels).float().mean().item(), 4))
