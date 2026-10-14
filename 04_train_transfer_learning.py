"""STEP 4 - MobileNetV2 transfer learning.
Stage A: frozen base (feature extractor) + new classifier  -> baseline
Stage B: unfreeze the last layers, fine-tune with a small learning rate
The better model (by VALIDATION accuracy) is saved as models/transfer_learning.keras
"""
import shutil

from common import *
from models_lib import build_transfer, unfreeze_base_top

set_seed()
n_classes = len(get_classes())
train_ds = make_dataset("train", use_image=True, augment=True, shuffle=True)
val_ds = make_dataset("val", use_image=True)

model = build_transfer(n_classes)

# ---- Stage A: frozen base
_, s1 = fit_model(model, "transfer_frozen", train_ds, val_ds, EPOCHS_TL_HEAD, lr=1e-3, patience=6)
val1 = model.evaluate(val_ds, verbose=0)[1]
model.save(MODELS_DIR / "transfer_frozen.keras")
evaluate_model("transfer_frozen", model, True, False, s1)

# ---- Stage B: fine-tune last layers
unfreeze_base_top(model.get_layer("mobilenetv2_base"), UNFREEZE_LAST_N)
_, s2 = fit_model(model, "transfer_finetuned", train_ds, val_ds, EPOCHS_TL_FT, lr=1e-5, patience=5)
val2 = model.evaluate(val_ds, verbose=0)[1]
model.save(MODELS_DIR / "transfer_finetuned.keras")
evaluate_model("transfer_finetuned", model, True, False, s1 + s2)

# ---- keep the better one under the standard name
best = "transfer_finetuned" if val2 >= val1 else "transfer_frozen"
print(f"\nVal acc  frozen: {val1:.4f} | fine-tuned: {val2:.4f}  ->  using {best}")
shutil.copy(MODELS_DIR / f"{best}.keras", MODELS_DIR / "transfer_learning.keras")
alias_result(best, "transfer_learning")
