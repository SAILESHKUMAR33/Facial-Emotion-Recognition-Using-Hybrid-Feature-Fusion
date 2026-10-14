"""Small image helpers that do NOT need TensorFlow (used by steps 1-2)."""
import numpy as np
from PIL import Image, ImageOps


def enhance_face_quality(img_rgb):
    """Enhance face image quality using adaptive histogram equalization (CLAHE) and unsharp masking."""
    import cv2
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    # Mild unsharp masking for clarity
    gaussian = cv2.GaussianBlur(enhanced_rgb, (0, 0), 1.0)
    sharpened = cv2.addWeighted(enhanced_rgb, 1.25, gaussian, -0.25, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def read_rgb(path, max_side=None, enhance=False):
    """Read any image as an RGB uint8 array, fixing phone-camera rotation (EXIF) and optional enhancement."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if max_side and max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.LANCZOS)
    arr = np.array(img)
    if enhance:
        arr = enhance_face_quality(arr)
    return arr
