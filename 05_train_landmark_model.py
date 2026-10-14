"""STEP 5 - Facial-landmark (geometry) model.
MediaPipe FaceMesh -> normalised distances / ratios / angles / coordinates -> small MLP.
"""
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from common import *
from face_utils import FEATURE_VECTOR_DIM, LandmarkExtractor
from models_lib import build_landmark_mlp

set_seed()
classes = get_classes()
n_classes = len(classes)

# ---- 1. extract features for every split (same file order as split_files)
lm_dir = PROCESSED_DIR / "landmarks"
lm_dir.mkdir(exist_ok=True)
ext = LandmarkExtractor()
dim = FEATURE_VECTOR_DIM
raw = {}
for split in ("train", "val", "test"):
    paths, _ = split_files(split)
    feats, failed = [], 0
    for p in tqdm(paths, desc=f"landmarks {split}"):
        f = ext.features(read_rgb(p))
        if f is None:
            failed += 1
            f = np.full(dim, np.nan, dtype="float32")
        feats.append(f)
    raw[split] = np.stack(feats)
    np.save(lm_dir / f"{split}.npy", raw[split])
    print(f"{split}: {raw[split].shape}, landmark failures: {failed}")

# ---- 2. fit scaler on TRAIN only
scaler = StandardScaler().fit(raw["train"][~np.isnan(raw["train"]).any(axis=1)])
joblib.dump(scaler, MODELS_DIR / "landmark_scaler.pkl")

# ---- 3. quick classical sanity check (not part of the final comparison)
_, y_tr = split_files("train")
_, y_te = split_files("test")
svm = SVC(kernel="rbf", C=5).fit(load_landmark_features("train"), y_tr)
print(f"[sanity] SVM on landmark features, test accuracy: "
      f"{svm.score(load_landmark_features('test'), y_te):.4f}")

# ---- 4. MLP classifier
train_ds = make_dataset("train", use_image=False, use_landmarks=True, shuffle=True)
val_ds = make_dataset("val", use_image=False, use_landmarks=True)
model = build_landmark_mlp(dim, n_classes)
model.summary()
_, secs = fit_model(model, "landmark_mlp", train_ds, val_ds, EPOCHS_LM, lr=1e-3, patience=15)
model.save(MODELS_DIR / "landmark_mlp.keras")
evaluate_model("landmark_mlp", model, use_image=False, use_landmarks=True, train_secs=secs)
