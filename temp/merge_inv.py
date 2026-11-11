"""
Merges the nine shards of painted pictures into inv/results.pt (without the generators) and measures what each edit did.

For an edit painted by continuing the picture of the KAE's prediction ("pred"):
  asked    = the edit's target state less the prediction's target state
  got      = the state DINOv3 reads from the edited picture less the state it reads from the "same" picture
             ("same" continues the prediction's picture for as many steps without any edit, so it is the fair reference)
  share    = how much of the asked change arrived: <got, asked> / <asked, asked>  (1 = all of it, 0 = none)
  cosine   = whether what changed is what was asked: cos(got, asked)
  asked size = |asked| / |prediction's target|; an edit that asks for less than the painting's own error cannot show
and a second opinion from a model that took no part in the painting: the cosine between CLIP ViT-B/32's embedding of the
picture and CLIP's text embedding of one word for the mode's meaning (snow, fruit, water, sky, knit, fire).

Usage: merge_inv.py
"""

import os
from pathlib import Path

import open_clip
import pandas as pd
import torch
from dotenv import load_dotenv
from PIL import Image
from torchvision.transforms.functional import to_tensor

load_dotenv("/home/nsa325/work/koopmann/.env")
torch.set_grad_enabled(False)
SP = Path(__file__).parent
WORDS = {667: "snow", 644: "snow", 247: "fruit", 37: "water", 71: "sky", 535: "knit", 211: "fire"}
EDITS = ["removed", "x2", "+90", "-90", "control", "knob gray", "knob sepia", "pred of gray copy"]
notes = torch.load(SP / "inv/notes.pt", weights_only=False)
results = {}
for index in notes:
    path = SP / f"inv/shard_{index}/results.pt"
    if not path.exists():
        print(f"no results for image {index}")
        continue
    for name, result in torch.load(path, weights_only=False).items():
        results[name] = {key: value for key, value in result.items() if key != "generator"}
torch.save(results, SP / "inv/results.pt")
print(len(results), "pictures merged")

clip, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", force_quick_gelu=True, cache_dir=f"{os.getenv('HF_HOME')}/open_clip")
clip.eval()
clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:, None, None]
clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:, None, None]
vocabulary = torch.load(SP / "clip_vocab.pt", weights_only=False)


def get_word_score(picture, word):
    image = torch.nn.functional.normalize(clip.encode_image(((picture - clip_mean) / clip_std)[None]), dim=-1)[0]
    text = torch.nn.functional.normalize(vocabulary["embeddings"][vocabulary["words"].index(word)].float(), dim=-1)
    return (image @ text).item()


rows = []
for index, note in notes.items():
    word = WORDS[note["mode"]]
    if f"{index}_same" not in results:
        continue
    pred, same = results[f"{index}_pred"], results[f"{index}_same"]
    photo = to_tensor(Image.open(SP / f"inv/photo_{index}.png").convert("RGB"))
    base = {"image": index, "mode": note["mode"], "word": word}
    rows.append({**base, "kind": "photo", "word score": get_word_score(photo, word), "colour": float("nan")})
    for kind in ["real", "recon", "pred", "same"]:
        if f"{index}_{kind}" in results:
            result = results[f"{index}_{kind}"]
            rows.append({**base, "kind": kind, "state error left": result["error"], "word score": get_word_score(result["picture"], word), "colour": result["colourfulness"]})
    for kind in EDITS:
        if f"{index}_{kind}" not in results:
            continue
        result = results[f"{index}_{kind}"]
        asked, got = result["target"] - pred["target"], result["state"] - same["state"]
        rows.append(
            {
                **base,
                "kind": kind,
                "state error left": result["error"],
                "asked size": (asked.norm() / pred["target"].norm()).item(),
                "share": ((got @ asked) / (asked @ asked)).item(),
                "cosine": torch.nn.functional.cosine_similarity(got, asked, dim=0).item(),
                "word score": get_word_score(result["picture"], word),
                "word score change": get_word_score(result["picture"], word) - get_word_score(same["picture"], word),
                "colour": result["colourfulness"],
                "colour change": result["colourfulness"] - same["colourfulness"],
            }
        )
    for key in ["gray", "sepia"]:
        if f"{index}_real {key}" in results:
            result, real = results[f"{index}_real {key}"], results[f"{index}_real"]
            asked, got = result["target"] - real["target"], result["state"] - real["state"]
            # NOTE: No "real, continued" picture was painted, so the reference here is the uncontinued picture of the real state
            rows.append({**base, "kind": f"real {key}", "state error left": result["error"], "asked size": (asked.norm() / real["target"].norm()).item(), "share": ((got @ asked) / (asked @ asked)).item(), "cosine": torch.nn.functional.cosine_similarity(got, asked, dim=0).item(), "colour": result["colourfulness"], "colour change": result["colourfulness"] - real["colourfulness"]})
table = pd.DataFrame(rows)
pd.set_option("display.width", 250, "display.max_rows", 500)
print(table.round(3).to_string(index=False))
print("\nmean over images, by kind")
print(table.groupby("kind", sort=False)[["state error left", "asked size", "share", "cosine", "word score", "word score change", "colour", "colour change"]].mean().round(3))
print("\nmode edits: in how many images does the word score move the expected way (down when removed, up when doubled), and by more than for the control?")
wide = table.pivot(index="image", columns="kind", values="word score change")
if {"removed", "x2", "control"} <= set(wide.columns):
    print(f"  removed: down in {(wide['removed'] < 0).sum()} of {len(wide)}; below the control in {(wide['removed'] < wide['control']).sum()}")
    print(f"  doubled: up in {(wide['x2'] > 0).sum()} of {len(wide)}; above the control in {(wide['x2'] > wide['control']).sum()}")
    print(wide.round(4).to_string())
table.to_csv(SP / "inv/edit_numbers.csv", index=False)
print("done")
