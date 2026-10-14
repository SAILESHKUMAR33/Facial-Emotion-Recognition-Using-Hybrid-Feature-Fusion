"""STEP 11 - Live webcam demo: draws the facial landmarks on your face and shows the
predicted emotion + confidence bars.

Run AFTER training (needs the models/ folder):
    python 11_webcam_demo.py                  # fusion model (best), camera 0
    python 11_webcam_demo.py --model lm       # landmarks-only model (fastest)
    python 11_webcam_demo.py --model tl       # also: cnn | fusion
    python 11_webcam_demo.py --cam 1          # another camera
    python 11_webcam_demo.py --source video.mp4

Keys:  q = quit   l = landmarks on/off   s = save screenshot (results/)
"""
import argparse
import os
import time
from collections import deque

import cv2
import joblib
import numpy as np
import tensorflow as tf

from config import *
from common import get_classes
from face_utils import COORD_IDX, FaceCropper, LandmarkExtractor, geometric_features

KEY_PTS = set(COORD_IDX)       # landmarks that feed the geometry features


def pick_model(name):
    if name == "fusion":
        f = MODELS_DIR / "best_fusion.txt"
        return f.read_text().strip() if f.exists() else "fusion_all_frozen"
    return {"cnn": "custom_cnn", "tl": "transfer_learning", "lm": "landmark_mlp"}[name]


def draw_bars(frame, classes, probs):
    h = 22
    for i, (c, p) in enumerate(zip(classes, probs)):
        y = 70 + i * (h + 6)
        cv2.rectangle(frame, (10, y), (10 + 180, y + h), (60, 60, 60), -1)
        cv2.rectangle(frame, (10, y), (10 + int(180 * p), y + h),
                      (0, 200, 0) if p == probs.max() else (200, 120, 0), -1)
        cv2.putText(frame, f"{c} {p:.0%}", (14, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fusion", choices=["fusion", "cnn", "tl", "lm"])
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--source", default=None, help="video/image path instead of a camera")
    ap.add_argument("--smooth", type=int, default=8, help="average predictions over N frames")
    ap.add_argument("--headless", action="store_true", help="no window (testing)")
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()

    classes = get_classes()
    model_name = pick_model(a.model)
    model = tf.keras.models.load_model(MODELS_DIR / f"{model_name}.keras", compile=False)
    scaler = joblib.load(MODELS_DIR / "landmark_scaler.pkl")
    need_lm = "landmarks" in model.input_names
    need_img = "image" in model.input_names
    print(f"Model: {model_name} | inputs: {model.input_names}")

    cropper, lmx = FaceCropper(0.5), LandmarkExtractor()
    if a.source:
        cap = cv2.VideoCapture(a.source)
    else:
        cap = cv2.VideoCapture(a.cam, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
    if not cap.isOpened():
        raise SystemExit("Cannot open camera/video. Try --cam 1 or close other apps using the camera.")

    WIN_NAME = "Emotion demo  (q quit, l landmarks, s save)"
    if not a.headless:
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME, 800, 600)
        cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_TOPMOST, 1)   # always on top

    show_lm, hist, n, t_prev, fps = True, deque(maxlen=a.smooth), 0, time.time(), 0.0
    last = None
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        n += 1
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = cropper.crop_box(rgb)
        label = "no face"
        if res is not None:
            crop, (x0, y0, x1, y1) = res
            # Apply real-time quality enhancement to live crop (CLAHE + unsharp)
            from io_utils import enhance_face_quality
            enhanced_crop = enhance_face_quality(crop)
            
            pts = lmx.raw(enhanced_crop)
            cv2.rectangle(bgr, (x0, y0), (x1, y1), (0, 255, 255), 2)
            if pts is not None and show_lm:
                sc = (x1 - x0) / IMG_SIZE
                for i, (px, py) in enumerate(pts):
                    X, Y = int(x0 + px * sc), int(y0 + py * sc)
                    if i in KEY_PTS:
                        cv2.circle(bgr, (X, Y), 2, (0, 0, 255), -1)
                    else:
                        cv2.circle(bgr, (X, Y), 1, (0, 255, 0), -1)
            if need_lm and pts is None:
                label = "face found, no landmarks"
            else:
                x = {}
                if need_img:
                    x["image"] = enhanced_crop[None].astype("float32")
                if need_lm:
                    x["landmarks"] = scaler.transform(geometric_features(pts)[None]).astype("float32")
                raw_pred = model.predict_on_batch(x)[0]
                hist.append(raw_pred)
                
                # Exponential weighting gives recent frames more influence while dampening noise
                weights = np.exp(np.linspace(-1.0, 0.0, len(hist)))
                weights /= weights.sum()
                probs = np.average(np.array(hist), axis=0, weights=weights)
                
                top_idx = int(probs.argmax())
                conf = probs[top_idx]
                label = f"{classes[top_idx].upper()}  {conf:.1%}"
                draw_bars(bgr, classes, probs)
        else:
            hist.clear()

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
        t_prev = now
        cv2.putText(bgr, label, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(bgr, f"{fps:.1f} fps | {model_name}", (10, bgr.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        last = bgr
        if not a.headless:
            cv2.imshow(WIN_NAME, bgr)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("l"):
                show_lm = not show_lm
            if k == ord("s"):
                out = RESULTS_DIR / f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(str(out), bgr)
                print("saved", out)
        if a.max_frames and n >= a.max_frames:
            break
    cap.release()
    cv2.destroyAllWindows()
    if a.headless and last is not None:
        out = RESULTS_DIR / "webcam_demo_last.jpg"
        cv2.imwrite(str(out), last)
        print("saved", out, "| label:", label)


if __name__ == "__main__":
    main()
