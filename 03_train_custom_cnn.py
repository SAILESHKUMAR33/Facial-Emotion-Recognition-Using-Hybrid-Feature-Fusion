"""STEP 3 - Train + evaluate the custom CNN (baseline)."""
from common import *
from models_lib import build_custom_cnn

set_seed()
n_classes = len(get_classes())
train_ds = make_dataset("train", use_image=True, augment=True, shuffle=True)
val_ds = make_dataset("val", use_image=True)

model = build_custom_cnn(n_classes)
model.summary()
_, secs = fit_model(model, "custom_cnn", train_ds, val_ds, EPOCHS_CNN, lr=1e-3, patience=10)

model.save(MODELS_DIR / "custom_cnn.keras")
evaluate_model("custom_cnn", model, use_image=True, use_landmarks=False, train_secs=secs)
