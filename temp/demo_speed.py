import os, sys, time
sys.path.insert(0, "/home/nsa325/work/koopmann/second_analysis")
import torch, open_clip
from demo_utils import *
torch.set_num_threads(8)
kae = Kae("im_iso_2x4k")
clip, _, _ = open_clip.create_model_and_transforms(ENCODER, pretrained="openai", force_quick_gelu=True, cache_dir=f"{os.getenv('HF_HOME')}/open_clip")
blocks = clip.visual.transformer.resblocks[6:]
print(len(blocks), "blocks; batch_first:", getattr(clip.visual.transformer, "batch_first", None))
tokens = torch.randn(256, 50, 768)
def clock(run):
    best = 1e9
    for _ in range(5):
        began = time.perf_counter(); run(); best = min(best, time.perf_counter() - began)
    return best
def run_clip():
    x = tokens
    for block in blocks: x = block(x)
def run_kae():
    summary = torch.cat([tokens[:, 0], tokens[:, 1:].mean(dim=1), tokens[:, 1:].std(dim=1)], dim=1)
    kae.model.decode(kae.model.koopman_forward(kae.model.encode(summary), 1))
c, k = clock(run_clip), clock(run_kae)
print(f"256 images, 8 CPU threads: CLIP blocks 7-12 {1000*c:.0f} ms, KAE {1000*k:.0f} ms, ratio {c/k:.1f}")
count = lambda m: sum(p.numel() for p in m.parameters())
print("params: blocks", count(blocks), "KAE", count(kae.model))
