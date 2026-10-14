"""STEP 8 - Ablation study.
Trains the three 2-branch fusions (branches frozen, same recipe as step 7 stage A).
Single-branch results come from steps 3-5, the 3-branch result from step 7.
"""
from common import *
from models_lib import build_fusion

set_seed()
n_classes = len(get_classes())
COMBOS = [
    (("cnn", "tl"), "fusion_cnn_tl"),
    (("tl", "lm"), "fusion_tl_lm"),
    (("cnn", "lm"), "fusion_cnn_lm"),
]

for branches, tag in COMBOS:
    print(f"\n########## {tag}: {branches} ##########")
    use_lm = "lm" in branches
    use_img = any(b in branches for b in ("cnn", "tl"))
    train_ds = make_dataset("train", use_img, use_lm, augment=True, shuffle=True)
    val_ds = make_dataset("val", use_img, use_lm)
    model = build_fusion(n_classes, branches, name=tag)
    _, secs = fit_model(model, tag, train_ds, val_ds, EPOCHS_FUSION, lr=1e-3, patience=8)
    model.save(MODELS_DIR / f"{tag}.keras")
    evaluate_model(tag, model, use_img, use_lm, secs)

print("\nAblation models done. Run 09_final_evaluation.py for the full table.")
