"""STEP 7 - Hybrid feature fusion: Custom CNN + Transfer Learning + Landmarks.

Stage A: all three trained branches frozen, train only fusion layers
         (BatchNorm per branch -> Concatenate -> Dense -> Dropout -> softmax)
Stage B: unfreeze the last MobileNetV2 layers, fine-tune with a small learning rate
"""
from common import *
from models_lib import build_fusion, unfreeze_fusion_tl_top

set_seed()
n_classes = len(get_classes())
BR = ("cnn", "tl", "lm")
train_ds = make_dataset("train", True, True, augment=True, shuffle=True)
val_ds = make_dataset("val", True, True)

model = build_fusion(n_classes, BR, name="fusion_all")
model.summary(expand_nested=False)

# ---- Stage A
_, s1 = fit_model(model, "fusion_all_frozen", train_ds, val_ds, EPOCHS_FUSION, lr=1e-3, patience=8)
val_a = model.evaluate(val_ds, verbose=0)[1]
model.save(MODELS_DIR / "fusion_all_frozen.keras")
evaluate_model("fusion_all_frozen", model, True, True, s1)

# ---- Stage B
unfreeze_fusion_tl_top(model, UNFREEZE_LAST_N)
_, s2 = fit_model(model, "fusion_all_finetuned", train_ds, val_ds, EPOCHS_FUSION_FT, lr=1e-5, patience=4)
val_b = model.evaluate(val_ds, verbose=0)[1]
model.save(MODELS_DIR / "fusion_all_finetuned.keras")
evaluate_model("fusion_all_finetuned", model, True, True, s1 + s2)

best = "fusion_all_finetuned" if val_b >= val_a else "fusion_all_frozen"
(MODELS_DIR / "best_fusion.txt").write_text(best)
print(f"\nVal acc  frozen: {val_a:.4f} | fine-tuned: {val_b:.4f}  ->  best fusion model: {best}")
