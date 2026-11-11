"""Contact sheet of the from-scratch pictures: one row per photo; columns photo | real | recon | pred. Usage: contact.py <out>"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw

SP = Path(__file__).parent
kinds = ["photo", "real", "recon", "pred"]
indices = [int(p.name.split("_")[1]) for p in sorted((SP / "inv").glob("shard_*")) if p.is_dir() and (p / f"{p.name.split('_')[1]}_pred.png").exists()]
sheet = Image.new("RGB", (224 * len(kinds) + 90, 224 * len(indices) + 20), "white")
draw = ImageDraw.Draw(sheet)
for column, kind in enumerate(kinds):
    draw.text((90 + 224 * column + 80, 4), kind, fill="black")
for row, index in enumerate(indices):
    draw.text((4, 20 + 224 * row + 100), str(index), fill="black")
    for column, kind in enumerate(kinds):
        path = SP / f"inv/photo_{index}.png" if kind == "photo" else SP / f"inv/shard_{index}/{index}_{kind}.png"
        sheet.paste(Image.open(path).convert("RGB").resize((224, 224)), (90 + 224 * column, 20 + 224 * row))
sheet.save(sys.argv[1])
print(indices)
