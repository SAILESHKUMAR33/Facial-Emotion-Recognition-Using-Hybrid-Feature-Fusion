"""Central configuration. Edit values here; every script imports this file."""
from pathlib import Path

BASE = Path(__file__).resolve().parent

# ---- folders -------------------------------------------------------------
DATASET_DIR = BASE / "dataset_face" if (BASE / "dataset_face").exists() else BASE / "dataset"
HEIC_BACKUP_DIR = BASE / "heic_originals"  # HEIC originals are moved here after conversion
PROCESSED_DIR = BASE / "processed"        # cropped faces split into train/val/test
MODELS_DIR = BASE / "models"
RESULTS_DIR = BASE / "results"
for _d in (PROCESSED_DIR, MODELS_DIR, RESULTS_DIR):
    _d.mkdir(exist_ok=True)

# ---- data ----------------------------------------------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42
VAL_FRAC = 0.15
TEST_FRAC = 0.15
FACE_MARGIN = 0.25        # extra context around the detected face box (each side)
MAX_SIDE = 1600           # big phone photos are down-scaled to this before detection
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HEIC_EXTS = {".heic", ".heif"}

# what step 1 flags is also removed in step 2 (set False to keep them)
EXCLUDE_NEAR_DUPLICATES = True
EXCLUDE_MULTIPLE_FACES = True

# ---- training ------------------------------------------------------------
FEATURE_DIM = 128
EPOCHS_CNN = 40
EPOCHS_TL_HEAD = 20
EPOCHS_TL_FT = 15
EPOCHS_LM = 150
EPOCHS_FUSION = 30
EPOCHS_FUSION_FT = 12
UNFREEZE_LAST_N = 30    # MobileNetV2 layers unfrozen during fine-tuning
