"""Train the hybrid Transfer Learning (MobileNetV2) + MediaPipe Facial Landmarks fusion model."""
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from common import *
from common import _load_img
from models_lib import _load_branch, build_fusion, unfreeze_fusion_tl_top

set_seed()
classes = get_classes()
n_classes = len(classes)
print(f"Loaded {n_classes} classes: {classes}")

stageA_model_path = MODELS_DIR / "fusion_tl_lm_stageA.keras"
stageB_model_path = MODELS_DIR / "fusion_tl_lm.keras"

s1 = 0
s2 = 0

# Check if Stage A is already trained
if stageA_model_path.exists():
    print(f"Loading existing Stage A checkpoint from {stageA_model_path}...")
    model = tf.keras.models.load_model(stageA_model_path, compile=False)
else:
    print("\n--- STAGE A: Training Fusion Classifier Head ---")
    tl_branch = _load_branch("tl")
    lm_branch = _load_branch("lm")

    tr_paths, y_tr = split_files("train")
    va_paths, y_va = split_files("val")

    lm_tr = load_landmark_features("train")
    lm_va = load_landmark_features("val")

    print(f"Extracting MobileNetV2 image features for {len(tr_paths)} train images (batch 64)...")
    ds_img_tr = (
        tf.data.Dataset.from_tensor_slices(tr_paths)
        .map(_load_img, num_parallel_calls=AUTOTUNE)
        .batch(64)
        .prefetch(AUTOTUNE)
    )
    t0_ext = time.perf_counter()
    feat_tl_tr = tl_branch.predict(ds_img_tr, verbose=1)

    print(f"Extracting MobileNetV2 image features for {len(va_paths)} val images (batch 64)...")
    ds_img_va = (
        tf.data.Dataset.from_tensor_slices(va_paths)
        .map(_load_img, num_parallel_calls=AUTOTUNE)
        .batch(64)
        .prefetch(AUTOTUNE)
    )
    feat_tl_va = tl_branch.predict(ds_img_va, verbose=1)
    print(f"Feature extraction completed in {time.perf_counter() - t0_ext:.1f}s")

    print("Extracting Landmark features...")
    feat_lm_tr = lm_branch.predict(lm_tr, batch_size=256, verbose=0)
    feat_lm_va = lm_branch.predict(lm_va, batch_size=256, verbose=0)

    # Train fusion head directly on extracted features (lightning-fast)
    print("Training fusion head on precomputed features...")
    in_tl = layers.Input((128,), name="tl_feat_in")
    in_lm = layers.Input((128,), name="lm_feat_in")
    bn_tl = layers.BatchNormalization(name="tl_feat_bn")(in_tl)
    bn_lm = layers.BatchNormalization(name="lm_feat_bn")(in_lm)
    concat = layers.Concatenate(name="fusion_concat")([bn_tl, bn_lm])
    dense = layers.Dense(256, activation="relu", name="fusion_dense")(concat)
    drop = layers.Dropout(0.5)(dense)
    out = layers.Dense(n_classes, activation="softmax", name="emotion")(drop)
    head = Model([in_tl, in_lm], out, name="fusion_head")

    head.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    cls = np.unique(y_tr)
    w = compute_class_weight("balanced", classes=cls, y=y_tr)
    cw = {int(c): float(v) for c, v in zip(cls, w)}

    t0_head = time.perf_counter()
    cbs = [
        tf.keras.callbacks.EarlyStopping("val_loss", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1),
    ]
    hist = head.fit(
        [feat_tl_tr, feat_lm_tr],
        y_tr,
        validation_data=([feat_tl_va, feat_lm_va], y_va),
        epochs=15,
        batch_size=64,
        class_weight=cw,
        callbacks=cbs,
        verbose=2,
    )
    s1 = time.perf_counter() - t0_head
    plot_history(hist, "fusion_tl_lm_stageA")

    # Assemble full end-to-end model and transfer trained head weights
    print("Assembling full end-to-end Multi-Modal Fusion Model...")
    model = build_fusion(n_classes, ("tl", "lm"), name="fusion_tl_lm")
    for name in ["tl_feat_bn", "lm_feat_bn", "fusion_dense", "emotion"]:
        model.get_layer(name).set_weights(head.get_layer(name).get_weights())

    model.save(stageA_model_path)
    print(f"Stage A checkpoint saved to {stageA_model_path} (Training time: {s1:.1f}s)")

# Stage B: Fine-tune MobileNetV2 top layers inside fusion
print("\n--- STAGE B: Fine-Tuning MobileNetV2 Top Layers inside Multi-Modal Fusion ---")
unfreeze_fusion_tl_top(model, UNFREEZE_LAST_N)
model.summary()

train_ds = make_dataset("train", use_image=True, use_landmarks=True, augment=True, shuffle=True)
val_ds = make_dataset("val", use_image=True, use_landmarks=True)

_, s2 = fit_model(model, "fusion_tl_lm", train_ds, val_ds, epochs=8, lr=1e-5, patience=3)

# Save best model and marker
model.save(stageB_model_path)
(MODELS_DIR / "best_fusion.txt").write_text("fusion_tl_lm")
print(f"Final fine-tuned fusion model saved to {stageB_model_path}")

# Final Evaluation on Held-out Test Set
print("\n--- Running Final Evaluation on Test Set ---")
entry = evaluate_model("fusion_tl_lm", model, use_image=True, use_landmarks=True, train_secs=s1 + s2)
print("\nHybrid Multi-Modal Fusion Model training and evaluation complete!")
