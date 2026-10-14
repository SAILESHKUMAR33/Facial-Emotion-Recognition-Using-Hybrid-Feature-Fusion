"""Optional helper - builds a picture of 8 flagged near-duplicate pairs so you can judge them by eye.
LEFT = the image that was flagged (would be removed), RIGHT = the image it was matched with (kept).
Output: results/near_duplicate_pairs.jpg
"""
import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw

from config import *
from io_utils import read_rgb

rows = [r for r in csv.DictReader(open(RESULTS_DIR / "dataset_issues.csv", encoding="utf-8"))
        if r["issue"].startswith("near_duplicate_of")]
print(f"near-duplicate rows: {len(rows)}")
index = {}
for p in DATASET_DIR.rglob("*"):
    if p.is_file():
        index.setdefault(p.name, []).append(p)

random.seed(1)
sample = random.sample(rows, min(8, len(rows)))
T = 224
sheet = Image.new("RGB", (T * 2 + 10, (T + 24) * len(sample)), "white")
d = ImageDraw.Draw(sheet)
cross = 0
for i, r in enumerate(sample):
    a = Path(r["path"])
    name = r["issue"].split(":", 1)[1]
    cands = index.get(name, [])
    b = next((c for c in cands if c.parent == a.parent), cands[0] if cands else None)
    if b is not None and b.parent != a.parent:
        cross += 1
    y = i * (T + 24)
    d.text((4, y + 4), f"{r['class']}: {a.name}  vs  {name}", fill="black")
    for k, pth in enumerate((a, b)):
        if pth is None:
            continue
        im = Image.fromarray(read_rgb(pth, 800))
        im.thumbnail((T, T))
        sheet.paste(im, (k * (T + 10), y + 24))
sheet.save(RESULTS_DIR / "near_duplicate_pairs.jpg", quality=90)
print(f"pairs whose match is in a DIFFERENT emotion folder: {cross} of {len(sample)}")
print("Saved results/near_duplicate_pairs.jpg  -> open it in VS Code")
