"""STEP 10 - Predict the emotion of one image with every model.

Usage:  python 10_predict.py --image path/to/photo.jpg
"""
import argparse
import sys

import cv2
import joblib
import numpy as np
import tensorflow as tf

from config import *
from face_utils import FaceCropper, LandmarkExtractor
from common import get_classes, read_rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--save", default=str(RESULTS_DIR / "prediction.jpg"))
    a = ap.parse_args()

    classes = get_classes()
    rgb = read_rgb(a.image, MAX_SIDE)
    crop = FaceCropper().crop(rgb)
    if crop is None:
        sys.exit("No face detected in the image.")

    feats = LandmarkExtractor().features(crop)
    img = crop[None].astype("float32")
    lm = None
    if feats is not None:
        lm = joblib.load(MODELS_DIR / "landmark_scaler.pkl").transform(feats[None]).astype("float32")

    best_fusion = (MODELS_DIR / "best_fusion.txt").read_text().strip() \
        if (MODELS_DIR / "best_fusion.txt").exists() else "fusion_all_frozen"
    models = [("Custom CNN", "custom_cnn"), ("Transfer Learning", "transfer_learning"),
              ("Landmarks MLP", "landmark_mlp"), ("FUSION (final)", best_fusion)]

    print(f"\n{'Model':<20}{'Prediction':<14}Confidence")
    final = None
    for label, fname in models:
        path = MODELS_DIR / f"{fname}.keras"
        if not path.exists():
            continue
        m = tf.keras.models.load_model(path, compile=False)
        if "landmarks" in m.input_names and lm is None:
            print(f"{label:<20}(landmarks not detected - skipped)")
            continue
        x = {n: (img if n == "image" else lm) for n in m.input_names}
        p = m.predict(x, verbose=0)[0]
        print(f"{label:<20}{classes[p.argmax()]:<14}{p.max():.1%}")
        final = (classes[p.argmax()], float(p.max()))

    if final:
        vis = cv2.cvtColor(cv2.resize(crop, (448, 448)), cv2.COLOR_RGB2BGR)
        cv2.putText(vis, f"{final[0]} {final[1]:.0%}", (10, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2)
        cv2.imwrite(a.save, vis)
        print(f"\nFinal prediction: {final[0]} ({final[1]:.1%})  | image saved to {a.save}")


if __name__ == "__main__":
    main()
