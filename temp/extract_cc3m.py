import sys
sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
from train import get_states
states, labels = get_states("ViT-B-32", "cc3m", "train", "cuda", [5, 11], tokens="cls_patch_std")
print("done", len(states), flush=True)
