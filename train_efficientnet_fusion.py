"""
Improved Hybrid Fusion: EfficientNetB0 + MediaPipe 468-pt Landmarks
Improvements over fusion_tl_lm:
  - EfficientNetB0 backbone (stronger than MobileNetV2)
  - Focal Loss (gamma=2) to focus on hard fear/sad/angry confusions
  - Label smoothing (0.1) to reduce overconfidence
  - Deeper fusion head (512 → 256)
  - Unfreeze top 80 EfficientNetB0 layers in Stage B
  - Stage A: pre-extract features → train head in seconds (not hours)
"""
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.utils.class_weight import compute_class_weight

from common import (set_seed, get_classes, split_files, load_landmark_features,
                    make_dataset, plot_history, evaluate_model, AUTOTUNE,
                    save_result, _load_img)
from config import MODELS_DIR, RESULTS_DIR, IMG_SIZE, SEED

UNFREEZE_LAST_N = 80   # top 80 of 238 EfficientNetB0 layers
FEATURE_DIM     = 256  # larger than MobileNetV2 baseline
BATCH_SIZE_FEAT = 32   # for feature extraction pass

set_seed(SEED)
classes    = get_classes()
n_classes  = len(classes)
lm_train   = load_landmark_features("train")
lm_val     = load_landmark_features("val")
lm_dim     = lm_train.shape[1]
print(f"Classes: {classes}  |  landmark dim: {lm_dim}")

# ── Focal Loss + Label Smoothing ────────────────────────────────────────────
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, label_smoothing=0.1, **kw):
        super().__init__(name="focal_loss", **kw)
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        n_cls  = tf.cast(tf.shape(y_pred)[-1], tf.float32)
        y_int  = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_oh   = tf.one_hot(y_int, tf.shape(y_pred)[-1])
        # label smoothing
        y_sm   = y_oh * (1.0 - self.label_smoothing) + self.label_smoothing / n_cls
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce     = -tf.reduce_sum(y_sm * tf.math.log(y_pred), axis=-1)
        # focal weight on un-smoothed prob of true class
        pt     = tf.reduce_sum(y_oh * y_pred, axis=-1)
        return tf.pow(1.0 - pt, self.gamma) * ce

    def get_config(self):
        cfg = super().get_config()
        cfg.update(gamma=self.gamma, label_smoothing=self.label_smoothing)
        return cfg

# ── Model builders ───────────────────────────────────────────────────────────
def build_effnet_branch():
    """EfficientNetB0 (frozen) → GlobalAvgPool → Dense(FEATURE_DIM)."""
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    # EfficientNetB0 expects raw uint8 or float32 0-255 (built-in preprocessing)
    inp = layers.Input((IMG_SIZE, IMG_SIZE, 3), name="image")
    x   = base(inp, training=False)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dropout(0.3)(x)
    x   = layers.Dense(FEATURE_DIM, activation="relu", name="features")(x)
    return Model(inp, x, name="effnet_branch")


def build_lm_branch():
    """Landmark MLP matching the same architecture trained earlier."""
    from models_lib import _BRANCH_FILES
    m = tf.keras.models.load_model(
        MODELS_DIR / f"{_BRANCH_FILES['lm']}.keras", compile=False)
    lm_feat_model = Model(m.input, m.get_layer("features").output,
                          name="lm_branch")
    lm_feat_model.trainable = False
    return lm_feat_model


def build_fusion(n_cls, effnet, lm_branch):
    img_inp = layers.Input((IMG_SIZE, IMG_SIZE, 3), name="image")
    lm_inp  = layers.Input((lm_dim,),               name="landmarks")

    f_img = effnet(img_inp, training=False)
    f_lm  = lm_branch(lm_inp, training=False)

    bn_img = layers.BatchNormalization(name="img_feat_bn")(f_img)
    bn_lm  = layers.BatchNormalization(name="lm_feat_bn")(f_lm)

    x = layers.Concatenate(name="fusion_concat")([bn_img, bn_lm])
    x = layers.Dense(512, activation="relu", name="fusion_dense1")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu", name="fusion_dense2")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_cls, activation="softmax", name="emotion")(x)
    return Model([img_inp, lm_inp], out, name="effnet_lm_fusion")


# ── Paths ────────────────────────────────────────────────────────────────────
stageA_path = MODELS_DIR / "effnet_lm_stageA.keras"
stageB_path = MODELS_DIR / "effnet_lm_fusion.keras"

s1 = s2 = 0

# ══════════════════════════════════════════════════════════════════════════════
# STAGE A – train fusion head on pre-extracted features (fast)
# ══════════════════════════════════════════════════════════════════════════════
if stageA_path.exists():
    print(f"\nLoading Stage A checkpoint: {stageA_path}")
    fusion = tf.keras.models.load_model(stageA_path, compile=False,
                                        custom_objects={"FocalLoss": FocalLoss})
else:
    print("\n── STAGE A: feature extraction + fusion head training ──")
    effnet    = build_effnet_branch()
    lm_branch = build_lm_branch()

    tr_paths, y_tr = split_files("train")
    va_paths, y_va = split_files("val")

    def img_ds(paths):
        return (tf.data.Dataset.from_tensor_slices(paths)
                .map(_load_img, num_parallel_calls=AUTOTUNE)
                .batch(BATCH_SIZE_FEAT).prefetch(AUTOTUNE))

    print(f"  Extracting EfficientNetB0 features for {len(tr_paths)} train images …")
    t0 = time.perf_counter()
    feat_img_tr = effnet.predict(img_ds(tr_paths), verbose=1)
    feat_img_va = effnet.predict(img_ds(va_paths), verbose=1)
    print(f"  Extracting landmark features …")
    feat_lm_tr  = lm_branch.predict(lm_train, batch_size=512, verbose=0)
    feat_lm_va  = lm_branch.predict(lm_val,   batch_size=512, verbose=0)
    print(f"  Feature extraction done in {time.perf_counter() - t0:.1f}s")

    # lightweight head trained on concatenated features
    in_img = layers.Input((FEATURE_DIM,), name="img_in")
    in_lm  = layers.Input((128,),         name="lm_in")    # lm_branch output = 128
    bn1    = layers.BatchNormalization(name="img_feat_bn")(in_img)
    bn2    = layers.BatchNormalization(name="lm_feat_bn")(in_lm)
    concat = layers.Concatenate(name="fusion_concat")([bn1, bn2])
    x      = layers.Dense(512, activation="relu", name="fusion_dense1")(concat)
    x      = layers.Dropout(0.4)(x)
    x      = layers.Dense(256, activation="relu", name="fusion_dense2")(x)
    x      = layers.Dropout(0.3)(x)
    out    = layers.Dense(n_classes, activation="softmax", name="emotion")(x)
    head   = Model([in_img, in_lm], out, name="fusion_head")

    cls = np.unique(y_tr)
    w   = compute_class_weight("balanced", classes=cls, y=y_tr)
    cw  = {int(c): float(v) for c, v in zip(cls, w)}

    head.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                 loss=FocalLoss(gamma=2.0, label_smoothing=0.1),
                 metrics=["accuracy"])

    t0_head = time.perf_counter()
    cbs = [
        tf.keras.callbacks.EarlyStopping("val_loss", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=2,
                                             min_lr=1e-7, verbose=1),
    ]
    hist = head.fit(
        [feat_img_tr, feat_lm_tr], y_tr,
        validation_data=([feat_img_va, feat_lm_va], y_va),
        epochs=20, batch_size=64, class_weight=cw,
        callbacks=cbs, verbose=2)
    s1 = time.perf_counter() - t0_head
    plot_history(hist, "effnet_lm_stageA")

    # Assemble end-to-end model and copy head weights
    print("\n  Assembling full end-to-end fusion model …")
    fusion = build_fusion(n_classes, effnet, lm_branch)
    for name in ["img_feat_bn", "lm_feat_bn",
                 "fusion_dense1", "fusion_dense2", "emotion"]:
        fusion.get_layer(name).set_weights(head.get_layer(name).get_weights())

    fusion.save(stageA_path)
    print(f"  Stage A saved → {stageA_path}  ({s1:.1f}s)")

# ══════════════════════════════════════════════════════════════════════════════
# STAGE B – unfreeze top 80 EfficientNetB0 layers + end-to-end fine-tuning
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n── STAGE B: Fine-tuning top {UNFREEZE_LAST_N} EfficientNetB0 layers ──")

effnet_sub = fusion.get_layer("effnet_branch")
effnet_sub.trainable = True
eff_base   = effnet_sub.get_layer("efficientnetb0")
for l in eff_base.layers[:-UNFREEZE_LAST_N]:
    l.trainable = False
for l in eff_base.layers:
    if isinstance(l, layers.BatchNormalization):
        l.trainable = False   # keep BN frozen to preserve normalisation stats

trainable = sum(np.prod(w.shape) for w in fusion.trainable_weights)
total     = sum(np.prod(w.shape) for w in fusion.weights)
print(f"  Trainable params: {trainable:,} / {total:,}")

train_ds = make_dataset("train", use_image=True, use_landmarks=True,
                        augment=True, shuffle=True)
val_ds   = make_dataset("val",   use_image=True, use_landmarks=True)

fusion.compile(
    optimizer=tf.keras.optimizers.Adam(5e-6),
    loss=FocalLoss(gamma=2.0, label_smoothing=0.1),
    metrics=["accuracy"])

_, y_tr = split_files("train")
cls = np.unique(y_tr)
w   = compute_class_weight("balanced", classes=cls, y=y_tr)
cw  = {int(c): float(v) for c, v in zip(cls, w)}

cbs = [
    tf.keras.callbacks.EarlyStopping("val_loss", patience=4, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=2,
                                         min_lr=1e-8, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(str(stageB_path), monitor="val_loss",
                                       save_best_only=True, verbose=1),
]

t0 = time.perf_counter()
hist = fusion.fit(train_ds, validation_data=val_ds, epochs=10,
                  class_weight=cw, callbacks=cbs, verbose=2)
s2 = time.perf_counter() - t0
plot_history(hist, "effnet_lm_stageB")

fusion.save(stageB_path)
(MODELS_DIR / "best_fusion.txt").write_text("effnet_lm_fusion")
print(f"\nModel saved → {stageB_path}")

# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Final Evaluation on Test Set ──")
evaluate_model("effnet_lm_fusion", fusion,
               use_image=True, use_landmarks=True,
               train_secs=s1 + s2)
print("\nDone!")
