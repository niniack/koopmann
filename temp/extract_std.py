import sys
sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
from train import get_states
for split, views in [("val", 1), ("train", 3)]:
    states, labels = get_states("ViT-B-32", "imagenet", split, "cuda", [5, 11], views=views, tokens="cls_patch_std")
    print("done", split, states.shape, flush=True)
    del states
