"""STEP 2 - Face detection + cropping, 224x224 RGB, stratified train/val/test split.

- skips files flagged in step 1 (corrupt / duplicates / no face / multiple faces)
- keeps only faces where BOTH face detection and landmark detection succeed, so that
  all three approaches are trained and tested on exactly the same images (fair comparison)
- augmentation is NOT done here: it is applied on-the-fly to the training set only
- pixel normalisation happens inside each model (Rescaling layer)
"""
import csv
import json
import shutil
import sys
from collections import Counter

from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import *
from face_utils import FaceCropper, LandmarkExtractor
from io_utils import read_rgb


def excluded_paths():
    f = RESULTS_DIR / "dataset_issues.csv"
    if not f.exists():
        print("WARNING: run 01_check_dataset.py first - no issues file, nothing excluded.")
        return set()
    drop = set()
    for row in csv.DictReader(open(f, encoding="utf-8")):
        kind = row["issue"].split(":")[0]
        if kind == "near_duplicate_of" and not EXCLUDE_NEAR_DUPLICATES:
            continue
        if kind == "multiple_faces" and not EXCLUDE_MULTIPLE_FACES:
            continue
        drop.add(row["path"])
    return drop


def main():
    has_train_test_split = (DATASET_DIR / "train").exists() and (DATASET_DIR / "test").exists()
    skip = excluded_paths()

    for sub in ("train", "val", "test", "_all", "landmarks"):
        shutil.rmtree(PROCESSED_DIR / sub, ignore_errors=True)

    cropper, lmx = FaceCropper(0.5), LandmarkExtractor()
    items, dropped, counter = [], [], Counter()

    if has_train_test_split:
        train_classes = sorted(d.name for d in (DATASET_DIR / "train").iterdir() if d.is_dir())
        manifest = []
        for orig_split in ("train", "test"):
            for cname in train_classes:
                cd = DATASET_DIR / orig_split / cname
                if not cd.exists():
                    continue
                files = [p for p in sorted(cd.rglob("*")) if p.suffix.lower() in IMG_EXTS]
                for p in tqdm(files, desc=f"{orig_split}/{cname}"):
                    if str(p) in skip:
                        dropped.append((str(p), "flagged_in_step1"))
                        continue
                    try:
                        rgb = read_rgb(p, MAX_SIDE, enhance=True)
                    except Exception:
                        dropped.append((str(p), "unreadable"))
                        continue
                    crop = cropper.crop(rgb)
                    if crop is None:
                        dropped.append((str(p), "no_face"))
                        continue
                    counter[cname] += 1
                    items.append((p, cname, orig_split, crop))

        if not items:
            sys.exit("No usable faces found in dataset.")
        classes = train_classes

        # Build train/val from train set, and test from test set
        train_items = [it for it in items if it[2] == "train"]
        test_items = [it for it in items if it[2] == "test"]

        # If test set exists, split train_items into train and val (e.g. 15% for val)
        tr_indices = list(range(len(train_items)))
        tr_labels = [it[1] for it in train_items]
        tr_idx, val_idx = train_test_split(
            tr_indices, test_size=VAL_FRAC, stratify=tr_labels, random_state=SEED)

        split_dict = {
            "train": [train_items[i] for i in tr_idx],
            "val": [train_items[i] for i in val_idx],
            "test": test_items,
        }

        manifest = []
        save_counter = Counter()
        for split_name, split_list in split_dict.items():
            for src_p, cname, _, crop in split_list:
                save_counter[cname] += 1
                dst_dir = PROCESSED_DIR / split_name / cname
                dst_dir.mkdir(parents=True, exist_ok=True)
                out_path = dst_dir / f"{cname}_{save_counter[cname]:05d}.jpg"
                Image.fromarray(crop).save(out_path, quality=95)
                manifest.append((split_name, cname, str(out_path), str(src_p)))

    else:
        class_dirs = sorted(d for d in DATASET_DIR.iterdir() if d.is_dir())
        for cd in class_dirs:
            (PROCESSED_DIR / "_all" / cd.name).mkdir(parents=True, exist_ok=True)
            files = [p for p in sorted(cd.rglob("*")) if p.suffix.lower() in IMG_EXTS]
            for p in tqdm(files, desc=cd.name):
                if str(p) in skip:
                    dropped.append((str(p), "flagged_in_step1"))
                    continue
                try:
                    rgb = read_rgb(p, MAX_SIDE, enhance=True)
                except Exception:
                    dropped.append((str(p), "unreadable"))
                    continue
                crop = cropper.crop(rgb)
                if crop is None:
                    dropped.append((str(p), "no_face"))
                    continue
                counter[cd.name] += 1
                out = PROCESSED_DIR / "_all" / cd.name / f"{cd.name}_{counter[cd.name]:05d}.jpg"
                Image.fromarray(crop).save(out, quality=95)
                items.append((out, cd.name, str(p)))

        if not items:
            sys.exit("No usable faces found. Check results/dropped_step2.csv for the reasons.")
        classes = sorted({c for _, c, _ in items})

        paths = [i[0] for i in items]
        labels = [i[1] for i in items]
        tr_p, tmp_p, tr_y, tmp_y = train_test_split(
            paths, labels, test_size=VAL_FRAC + TEST_FRAC, stratify=labels, random_state=SEED)
        va_p, te_p, va_y, te_y = train_test_split(
            tmp_p, tmp_y, test_size=TEST_FRAC / (VAL_FRAC + TEST_FRAC), stratify=tmp_y,
            random_state=SEED)

        src_of = {i[0]: i[2] for i in items}
        manifest = []
        for split, ps, ys in (("train", tr_p, tr_y), ("val", va_p, va_y), ("test", te_p, te_y)):
            for p, y in zip(ps, ys):
                dst = PROCESSED_DIR / split / y
                dst.mkdir(parents=True, exist_ok=True)
                shutil.move(str(p), str(dst / p.name))
                manifest.append((split, y, str(dst / p.name), src_of[p]))
        shutil.rmtree(PROCESSED_DIR / "_all", ignore_errors=True)

    json.dump(classes, open(PROCESSED_DIR / "classes.json", "w"))
    with open(PROCESSED_DIR / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "class", "processed_path", "source_path"])
        w.writerows(manifest)
    with open(RESULTS_DIR / "dropped_step2.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "reason"])
        w.writerows(dropped)

    print("\n================ SPLIT SUMMARY ================")
    print(f"{'class':<15}{'train':>8}{'val':>8}{'test':>8}")
    for c in classes:
        n = [sum(1 for m in manifest if m[0] == s and m[1] == c) for s in ("train", "val", "test")]
        print(f"{c:<15}{n[0]:>8}{n[1]:>8}{n[2]:>8}")
    print(f"\nKept {len(items)} images, dropped {len(dropped)} "
          f"(see results/dropped_step2.csv)")


if __name__ == "__main__":
    main()
