import os, glob, torch, open_clip, numpy as np
from PIL import Image
root = os.environ['DATASETS_CACHE'] + '/snowy_imagenet'
names = {'armytank':'tank','carwheel':'car wheel','firetruck':'fire truck','motorscooter':'vespa','racingcar':'race car','schoolbus':'school bus','trafficlight':'traffic light'}
C = list(open_clip.IMAGENET_CLASSNAMES); print('ok' if all(v in C for v in names.values()) else [v for v in names.values() if v not in C], len(C), len(open_clip.OPENAI_IMAGENET_TEMPLATES))
folders = sorted(names); ids = torch.tensor([C.index(names[f]) for f in folders]); print(dict(zip(folders, ids.tolist())))
paths = [(p, i) for i, f in enumerate(folders) for p in sorted(glob.glob(f'{root}/{f}/*'))]
y = torch.tensor([i for _, i in paths]); dev = 'cuda'
single = [lambda c: f'a photo of a {c}.']
paper = {'ViT-B-16': '55.2 +- 24.6', 'ViT-B-32': '53.4 +- 29.9', 'ViT-L-14': '60.2 +- 22.5'}
for name in ('ViT-B-16', 'ViT-B-32', 'ViT-L-14'):
    m, _, pre = open_clip.create_model_and_transforms(name, pretrained='openai', cache_dir=os.environ['HF_HOME'] + '/open_clip')
    m = m.to(dev).eval(); tok = open_clip.get_tokenizer(name)
    with torch.no_grad():
        f = torch.cat([m.encode_image(torch.stack([pre(Image.open(p).convert('RGB')) for p, _ in paths[i:i+32]]).to(dev)) for i in range(0, len(paths), 32)]).float()
        f = f / f.norm(dim=-1, keepdim=True)
        for label, T in (('1 template', single), ('80 templates', open_clip.OPENAI_IMAGENET_TEMPLATES)):
            W = []
            for c in C:
                e = m.encode_text(tok([t(c) for t in T]).to(dev)).float(); e = e / e.norm(dim=-1, keepdim=True); e = e.mean(0); W.append(e / e.norm())
            logits = f @ torch.stack(W).T
            for way, pred in (('1000-way', logits.argmax(1).cpu()), ('7-way', ids[logits[:, ids].argmax(1).cpu()])):
                ok = pred == ids[y]
                per = torch.stack([ok[y == k].float().mean() for k in range(7)]) * 100
                print(f"{name} {label:12s} {way:8s} class-mean {per.mean():5.1f} +- {per.std():4.1f} (sample-std {per.std(unbiased=False):4.1f}) | image acc {ok.float().mean()*100:5.1f} | paper {paper[name]}")
    del m; torch.cuda.empty_cache()
