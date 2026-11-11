import sys
sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
from train import get_states
states, labels = get_states("ViT-B-32", "imagenet", "train", "cuda", [5, 11], views=3, patches=True)
print("done", states.shape, labels.shape, flush=True)
