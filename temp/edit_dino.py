import sys
p = sys.argv[1]
s = open(p).read()
def swap(old, new):
    global s
    assert s.count(old) == 1, old
    s = s.replace(old, new)
swap('''def report(name, feats, labels, val, vlabels):
    """feats: dict of block-6 inputs plus 'end' (block 12 CLS)."""''', '''def report(name, feats, labels, val, vlabels, head=None):
    """feats: dict of block-6 inputs plus 'end' (block 12 CLS). head: the model's own classifier on a raw block 12 CLS."""''')
swap('''probe on true block 12 CLS {(true == vlabels).float().mean():.3f}", flush=True)''', '''probe on true block 12 CLS {(true == vlabels).float().mean():.3f}", flush=True)
    if head is not None:
        head_true = head(val["end"]).argmax(-1)
        print(f"  official head on true block 12 CLS {(head_true == vlabels).float().mean():.3f}", flush=True)''')
swap('''        answers = read(pred)
''', '''        answers = read(pred)
        extra = ""
        if head is not None:
            head_answers = head(pred * sd["end"] + mu["end"]).argmax(-1)
            extra = f" | official head {(head_answers == vlabels).float().mean():.3f}, agrees {(head_answers == head_true).float().mean():.3f}"
''')
swap('''agrees with probe on true {(answers == true).float().mean():.3f}", flush=True)''', '''agrees with probe on true {(answers == true).float().mean():.3f}{extra}", flush=True)''')
swap('''report("DINOv3 ViT-B/16", train, ty, val, vy)''', '''# The official linear head from github.com/m2b3/dinov3-in1k-probes: final layer norm, then 768 -> 1000, trained at 512 px
from glob import glob
from safetensors.torch import load_file
probe = load_file(glob(os.getenv("HF_HOME") + "/hub/models--yberreby--dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe/snapshots/*/model.safetensors")[0])
head = lambda cls: model.norm(cls) @ probe["weight"].T + probe["bias"]
report("DINOv3 ViT-B/16", train, ty, val, vy, head)''')
swap('''saved = torch.load(f"{SP}/pool_check.pt")
report("CLIP ViT-B/32", saved["train"], saved["ty"], saved["val"], saved["vy"])
''', '''# NOTE: CLIP's numbers were printed by an earlier run of this script
''')
open(p, "w").write(s)
print("edited")
