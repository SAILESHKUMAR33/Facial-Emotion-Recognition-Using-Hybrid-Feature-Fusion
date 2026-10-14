"""STEP 1 - Audit the raw dataset (and convert HEIC/HEIF -> JPG).

Expected layout:  dataset/<emotion_name>/image files
Checks: image count, classes, images per class, formats, corrupt files,
exact + near duplicates, images with no face / multiple faces.
Outputs: results/dataset_issues.csv, results/dataset_report.json
"""
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter

import cv2
from PIL import Image, ImageOps
from tqdm import tqdm

from config import *
from face_utils import FaceCropper
from io_utils import read_rgb

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    print("WARNING: pillow-heif not installed - HEIC files cannot be converted "
          "(pip install pillow-heif)")


def dhash(rgb, size=8):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def main():
    if not DATASET_DIR.exists():
        sys.exit(f"Dataset folder not found: {DATASET_DIR}\n"
                 "Create it as dataset/<emotion>/images...")
    class_dirs = sorted(d for d in DATASET_DIR.iterdir() if d.is_dir())
    if not class_dirs:
        sys.exit("No class sub-folders found inside dataset/")

    # ---- 1. HEIC/HEIF -> JPG --------------------------------------------
    converted = 0
    for cd in class_dirs:
        for p in sorted(cd.rglob("*")):
            if p.is_file() and p.suffix.lower() in HEIC_EXTS:
                out = p.with_suffix(".jpg")
                if out.exists():
                    out = p.with_name(p.stem + "_heic.jpg")
                try:
                    ImageOps.exif_transpose(Image.open(p)).convert("RGB").save(out, quality=95)
                except Exception as e:
                    print(f"  could not convert {p.name}: {e}")
                    continue
                bk = HEIC_BACKUP_DIR / cd.name
                bk.mkdir(parents=True, exist_ok=True)
                shutil.move(str(p), str(bk / p.name))
                converted += 1
    print(f"HEIC/HEIF files converted to JPG: {converted}")

    # ---- 2. scan everything ---------------------------------------------
    records = [(p, cd.name) for cd in class_dirs for p in sorted(cd.rglob("*")) if p.is_file()]
    ext_count, per_class_total, per_class_clean = Counter(), Counter(), Counter()
    issues, seen_md5, seen_dh = [], {}, {}
    cropper = FaceCropper(min_conf=0.6)

    for p, cls in tqdm(records, desc="Checking images"):
        ext = p.suffix.lower()
        ext_count[ext] += 1
        if ext not in IMG_EXTS:
            issues.append((p, cls, "unsupported_format"))
            continue
        try:
            with Image.open(p) as im:
                im.verify()
            rgb = read_rgb(p, 1280)
        except Exception:
            issues.append((p, cls, "corrupt"))
            continue
        per_class_total[cls] += 1

        md5 = hashlib.md5(p.read_bytes()).hexdigest()
        if md5 in seen_md5:
            issues.append((p, cls, f"duplicate_of:{seen_md5[md5].name}"))
            continue
        seen_md5[md5] = p
        dh = dhash(rgb)
        if dh in seen_dh:
            issues.append((p, cls, f"near_duplicate_of:{seen_dh[dh].name}"))
            continue
        seen_dh[dh] = p

        n_faces = len(cropper.detections(rgb))
        if n_faces == 0:
            issues.append((p, cls, "no_face"))
        elif n_faces > 1:
            issues.append((p, cls, "multiple_faces"))
        else:
            per_class_clean[cls] += 1

    # ---- 3. report ---------------------------------------------------------
    with open(RESULTS_DIR / "dataset_issues.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "class", "issue"])
        w.writerows([(str(p), c, i) for p, c, i in issues])

    issue_kinds = Counter(i.split(":")[0] for _, _, i in issues)
    report = {
        "classes": [c.name for c in class_dirs],
        "total_files": len(records),
        "formats": dict(ext_count),
        "images_per_class_readable": dict(per_class_total),
        "images_per_class_clean": dict(per_class_clean),
        "issue_counts": dict(issue_kinds),
        "heic_converted": converted,
    }
    json.dump(report, open(RESULTS_DIR / "dataset_report.json", "w"), indent=2)

    print("\n================ DATASET REPORT ================")
    print(f"Total files        : {len(records)}")
    print(f"Emotion categories : {len(class_dirs)} -> {report['classes']}")
    print(f"Formats            : {dict(ext_count)}")
    print(f"\n{'class':<15}{'readable':>10}{'clean':>10}")
    for c in report["classes"]:
        print(f"{c:<15}{per_class_total[c]:>10}{per_class_clean[c]:>10}")
    print(f"\nIssues found: {dict(issue_kinds) or 'none'}")
    print(f"Details in {RESULTS_DIR / 'dataset_issues.csv'}  (open it and review!)")
    clean = [per_class_clean[c] for c in report["classes"] if per_class_clean[c]]
    if clean and max(clean) / min(clean) > 3:
        print("NOTE: classes are imbalanced (>3x) - class weights are used in training.")
    if clean and min(clean) < 20:
        print("WARNING: some class has <20 clean images - results will be unreliable.")
    print("Flagged files are NOT deleted; step 2 simply skips them.")


if __name__ == "__main__":
    main()
