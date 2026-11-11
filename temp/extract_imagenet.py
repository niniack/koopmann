import sys
import time

sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
from train import get_states

from koopmann.utils import get_device

for split in ["val", "train"]:
    start = time.time()
    states, labels = get_states("ViT-B-32", "imagenet", split, get_device(), [5, 11])
    print(split, tuple(states.shape), tuple(labels.shape), states.dtype, f"{time.time() - start:.0f}s", flush=True)
