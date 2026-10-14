"""Shared data pipeline, training and evaluation helpers (TensorFlow side)."""
import json
import os
import random
import shutil
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import matplotlib
matplotlib.use("Agg")                      # save figures to files, never open windows
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.utils.class_weight import compute_class_weight

from config import *            # noqa: F401,F403
from io_utils import read_rgb   # noqa: F401

AUTOTUNE = tf.data.AUTOTUNE
RESULTS_FILE = RESULTS_DIR / "results.json"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ------------------------------------------------------------------ data ---
def get_classes():
    for p in (PROCESSED_DIR / "classes.json", MODELS_DIR / "classes.json"):
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return ["angry", "fear", "happy", "sad", "surprise"]


def split_files(split):
    """Deterministic (paths, int labels) for train / val / test."""
    paths, labels = [], []
    for idx, c in enumerate(get_classes()):
        for p in sorted((PROCESSED_DIR / split / c).glob("*.jpg")):
            paths.append(str(p))
            labels.append(idx)
    return paths, np.array(labels, dtype="int32")


def load_landmark_features(split):
    """Scaled landmark features, same order as split_files(split)."""
    raw = np.load(PROCESSED_DIR / "landmarks" / f"{split}.npy").astype("float64")
    scaler = joblib.load(MODELS_DIR / "landmark_scaler.pkl")
    bad = np.isnan(raw).any(axis=1)
    raw[bad] = scaler.mean_
    return scaler.transform(raw).astype("float32")


def _load_img(path):
    img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return tf.cast(img, tf.float32)         # 0..255 RGB; each model normalises internally


def _make_aug(flip):
    L = tf.keras.layers
    layers = [L.RandomFlip("horizontal")] if flip else []
    layers += [L.RandomRotation(0.06), L.RandomZoom(0.1), L.RandomContrast(0.15),
               L.RandomBrightness(0.1, value_range=(0, 255))]
    return tf.keras.Sequential(layers)


def make_dataset(split, use_image=True, use_landmarks=False, augment=False, shuffle=False):
    """tf.data pipeline. Augmentation is applied only when augment=True (training set).
    Horizontal flip is skipped when landmarks are used (it would swap left/right features)."""
    paths, labels = split_files(split)
    feats = {"path": np.array(paths)}
    if use_landmarks:
        feats["landmarks"] = load_landmark_features(split)
    ds = tf.data.Dataset.from_tensor_slices((feats, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)

    def _map(f, y):
        out = {}
        if use_image:
            out["image"] = _load_img(f["path"])
        if use_landmarks:
            out["landmarks"] = f["landmarks"]
        return out, y

    ds = ds.map(_map, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)
    if augment and use_image:
        aug = _make_aug(flip=not use_landmarks)

        def _aug_map(x, y):
            x = dict(x)
            x["image"] = aug(x["image"], training=True)
            return x, y

        ds = ds.map(_aug_map, num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


# -------------------------------------------------------------- training ---
def fit_model(model, tag, train_ds, val_ds, epochs, lr, patience=8, class_weight=True):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cw = None
    if class_weight:
        _, y = split_files("train")
        cls = np.unique(y)
        w = compute_class_weight("balanced", classes=cls, y=y)
        cw = {int(c): float(v) for c, v in zip(cls, w)}
    cbs = [
        tf.keras.callbacks.EarlyStopping("val_loss", patience=patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=max(2, patience // 3),
                                             min_lr=1e-7, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(str(MODELS_DIR / f"{tag}.keras"), monitor="val_loss",
                                           save_best_only=True, verbose=1),
    ]
    t0 = time.perf_counter()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cbs,
                     class_weight=cw, verbose=2)
    secs = time.perf_counter() - t0
    plot_history(hist, tag)
    print(f"[{tag}] training time: {secs:.1f}s")
    return hist, secs


def plot_history(hist, tag):
    h = hist.history
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(h["loss"], label="train"); ax[0].plot(h["val_loss"], label="val")
    ax[0].set_title(f"{tag} - loss"); ax[0].legend()
    ax[1].plot(h["accuracy"], label="train"); ax[1].plot(h["val_accuracy"], label="val")
    ax[1].set_title(f"{tag} - accuracy"); ax[1].legend()
    fig.tight_layout(); fig.savefig(RESULTS_DIR / f"history_{tag}.png", dpi=120); plt.close(fig)


# ------------------------------------------------------------ evaluation ---
def load_results():
    return json.load(open(RESULTS_FILE)) if RESULTS_FILE.exists() else {}


def save_result(name, entry):
    res = load_results()
    res[name] = entry
    json.dump(res, open(RESULTS_FILE, "w"), indent=2)


def plot_confusion(cm, classes, name):
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.0 * len(classes) + 3, 1.0 * len(classes) + 2.5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"Confusion matrix - {name}")
    fig.colorbar(im, fraction=0.046); fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"confusion_{name}.png", dpi=130); plt.close(fig)


def measure_inference_ms(model, use_image, use_landmarks, n=128):
    ds = make_dataset("test", use_image, use_landmarks).unbatch().take(n).batch(32)
    batches = [x for x, _ in ds]
    model.predict_on_batch(batches[0])                       # warm-up
    count = sum(len(next(iter(b.values()))) for b in batches)
    t0 = time.perf_counter()
    for b in batches:
        model.predict_on_batch(b)
    return (time.perf_counter() - t0) / count * 1000


def evaluate_model(name, model, use_image, use_landmarks, train_secs):
    classes = get_classes()
    labels = list(range(len(classes)))
    _, y_true = split_files("test")
    _, y_val = split_files("val")
    p_test = model.predict(make_dataset("test", use_image, use_landmarks), verbose=0)
    p_val = model.predict(make_dataset("val", use_image, use_landmarks), verbose=0)
    y_pred = p_test.argmax(1)

    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    _, _, f1_cls, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    entry = {
        "name": name,
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(pr), "recall_macro": float(rc), "f1_macro": float(f1),
        "val_accuracy": float(accuracy_score(y_val, p_val.argmax(1))),
        "train_time_s": float(train_secs),
        "inference_ms_per_image": float(measure_inference_ms(model, use_image, use_landmarks)),
        "params": int(model.count_params()),
        "per_class_f1": {c: float(v) for c, v in zip(classes, f1_cls)},
    }
    plot_confusion(confusion_matrix(y_true, y_pred, labels=labels), classes, name)
    np.save(RESULTS_DIR / f"preds_{name}.npy", y_pred)
    (RESULTS_DIR / f"report_{name}.txt").write_text(
        classification_report(y_true, y_pred, target_names=classes, zero_division=0))
    save_result(name, entry)
    print(f"\n=== {name} ===  test acc {entry['test_accuracy']:.4f} | macro-F1 {entry['f1_macro']:.4f} "
          f"| P {pr:.4f} | R {rc:.4f} | {entry['inference_ms_per_image']:.1f} ms/img")
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))
    return entry


def alias_result(src, dst):
    """Make results['dst'] a copy of results['src'] (incl. predictions + confusion plot)."""
    res = load_results()
    res[dst] = {**res[src], "name": dst, "source": src}
    json.dump(res, open(RESULTS_FILE, "w"), indent=2)
    shutil.copy(RESULTS_DIR / f"preds_{src}.npy", RESULTS_DIR / f"preds_{dst}.npy")
    shutil.copy(RESULTS_DIR / f"confusion_{src}.png", RESULTS_DIR / f"confusion_{dst}.png")
