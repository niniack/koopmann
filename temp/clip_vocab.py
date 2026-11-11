"""
CLIP text embeddings of 20,000 common English words, for naming modes by their nearest words (as Discover-then-Name and SpLiCE do).

Each word is embedded alone and as "a photo of {word}", and the two unit vectors are averaged.
Usage: python -I clip_vocab.py <word list> <out file>   (the word list is a download, so it is only ever read as text)
"""

import os
import sys

import open_clip
import torch
from dotenv import load_dotenv

load_dotenv("/home/nsa325/work/koopmann/.env")
torch.set_num_threads(6)
torch.set_grad_enabled(False)
ENCODER = "ViT-B-32"

with open(sys.argv[1]) as file:
    words = [line.strip() for line in file]
words = [word for word in words if word.isascii() and word.isalpha() and len(word) >= 3]
model, _, _ = open_clip.create_model_and_transforms(
    ENCODER, pretrained="openai", force_quick_gelu=True, cache_dir=f"{os.getenv('HF_HOME')}/open_clip"
)
model.eval()
tokenizer = open_clip.get_tokenizer(ENCODER)
embeddings = []
for first in range(0, len(words), 500):
    chunk = words[first : first + 500]
    both = [
        torch.nn.functional.normalize(model.encode_text(tokenizer([template.format(word) for word in chunk])), dim=-1)
        for template in ("{}", "a photo of {}")
    ]
    embeddings.append(torch.nn.functional.normalize(both[0] + both[1], dim=-1))
    print(first, flush=True)
torch.save({"words": words, "embeddings": torch.cat(embeddings)}, sys.argv[2])
print("done", len(words))
