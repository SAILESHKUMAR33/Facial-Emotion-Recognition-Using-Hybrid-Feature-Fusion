"""Face detection/cropping + MediaPipe landmark geometry features."""
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions import face_mesh_connections as fmc

from config import FACE_MARGIN, IMG_SIZE

EPS = 1e-6


def _idx(*conns):
    return sorted({i for c in conns for pair in c for i in pair})


# landmark indices whose normalised (x, y) are used as raw coordinate features
COORD_IDX = _idx(fmc.FACEMESH_LIPS, fmc.FACEMESH_LEFT_EYE, fmc.FACEMESH_RIGHT_EYE,
                 fmc.FACEMESH_LEFT_EYEBROW, fmc.FACEMESH_RIGHT_EYEBROW)


class FaceCropper:
    """Detect the most confident face and return a square 224x224 RGB crop."""

    def __init__(self, min_conf=0.5):
        self.det = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=min_conf)

    def detections(self, rgb):
        return self.det.process(rgb).detections or []

    def crop(self, rgb):
        h, w = rgb.shape[:2]
        dets = self.detections(rgb)
        if not dets:
            # If the image is already a face crop (e.g. 48x48 cropped dataset), resize and enhance
            if min(h, w) <= 128:
                interp = cv2.INTER_CUBIC
                return cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=interp)
            return None
        d = max(dets, key=lambda k: k.score[0])
        bb = d.location_data.relative_bounding_box
        bx, by, bw, bh = bb.xmin * w, bb.ymin * h, bb.width * w, bb.height * h
        cx, cy = bx + bw / 2, by + bh / 2
        side = max(bw, bh) * (1 + 2 * FACE_MARGIN)
        x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
        x1, y1 = x0 + int(round(side)), y0 + int(round(side))
        pad = max(0, -x0, -y0, x1 - w, y1 - h)
        if pad > 0:
            rgb = cv2.copyMakeBorder(rgb, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
            x0, x1, y0, y1 = x0 + pad, x1 + pad, y0 + pad, y1 + pad
        crop = rgb[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        interp = cv2.INTER_AREA if crop.shape[0] > IMG_SIZE else cv2.INTER_CUBIC
        return cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=interp)

    def crop_box(self, rgb):
        h, w = rgb.shape[:2]
        dets = self.detections(rgb)
        if not dets:
            return None
        d = max(dets, key=lambda k: k.score[0])
        bb = d.location_data.relative_bounding_box
        bx, by, bw, bh = bb.xmin * w, bb.ymin * h, bb.width * w, bb.height * h
        cx, cy = bx + bw / 2, by + bh / 2
        side = max(bw, bh) * (1 + 2 * FACE_MARGIN)
        x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
        x1, y1 = x0 + int(round(side)), y0 + int(round(side))
        pad = max(0, -x0, -y0, x1 - w, y1 - h)
        padded = rgb
        if pad > 0:
            padded = cv2.copyMakeBorder(rgb, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
            x0_c, x1_c, y0_c, y1_c = x0 + pad, x1 + pad, y0 + pad, y1 + pad
        else:
            x0_c, x1_c, y0_c, y1_c = x0, x1, y0, y1
        crop = padded[y0_c:y1_c, x0_c:x1_c]
        if crop.size == 0:
            return None
        interp = cv2.INTER_AREA if crop.shape[0] > IMG_SIZE else cv2.INTER_CUBIC
        resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=interp)
        return resized, (max(0, x0), max(0, y0), min(w, x1), min(h, y1))


def normalize_points(pts):
    """Centre on the eye midpoint, rotate so the eye line is horizontal, scale by eye distance.
    Result is invariant to face position, size and in-plane rotation."""
    le = pts[[33, 133]].mean(0)
    re = pts[[362, 263]].mean(0)
    center = (le + re) / 2
    d = re - le
    angle = np.arctan2(d[1], d[0])
    c, s = np.cos(-angle), np.sin(-angle)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return ((pts - center) @ rot.T) / (np.linalg.norm(d) + EPS)


def geometric_features(pts):
    """pts: (468, 2) pixel landmarks -> 1-D float32 feature vector."""
    n = normalize_points(pts)

    def dist(a, b):
        return float(np.linalg.norm(n[a] - n[b]))

    def ang(a, b, mirror=False):
        v = n[b] - n[a]
        return float(np.arctan2(v[1], -v[0] if mirror else v[0]))

    f = []
    # eye openness (height, aspect ratio) for both eyes
    for outer, inner, up, low in [(33, 133, 159, 145), (263, 362, 386, 374)]:
        w, h = dist(outer, inner), dist(up, low)
        f += [w, h, h / (w + EPS)]
    # mouth width / height / ratios
    mw, mh_in, mh_out = dist(61, 291), dist(13, 14), dist(0, 17)
    f += [mw, mh_in, mh_out, mh_in / (mw + EPS), mh_out / (mw + EPS)]
    # smile / frown: mouth corners relative to lip centre (image y grows downwards)
    mid_y = (n[13, 1] + n[14, 1]) / 2
    f += [n[61, 1] - mid_y, n[291, 1] - mid_y, (n[61, 1] + n[291, 1]) / 2 - mid_y]
    # mouth corners to nose tip and to outer eye corners
    f += [dist(61, 1), dist(291, 1), dist(61, 33), dist(291, 263)]
    # eyebrow-to-eye distances (mid, inner, outer) both sides
    f += [dist(105, 159), dist(334, 386), dist(107, 133), dist(336, 362), dist(70, 33), dist(300, 263)]
    # eyebrow slopes and inner-brow gap (frown / raise)
    f += [ang(70, 107), ang(300, 336, mirror=True), dist(107, 336)]
    # nose
    f += [dist(1, 13), dist(129, 358), dist(168, 1)]
    # face proportions and mouth-chin
    fw, fh = dist(234, 454), dist(10, 152)
    f += [fw, fh, fw / (fh + EPS), dist(17, 152), float(np.linalg.norm(n[[13, 14]].mean(0)))]
    # raw normalised coordinates of lips / eyes / eyebrows
    f += n[COORD_IDX].ravel().tolist()
    return np.nan_to_num(np.asarray(f, dtype=np.float32))


FEATURE_VECTOR_DIM = len(geometric_features(
    np.random.RandomState(0).rand(468, 2).astype("float32") * 100 + 20))


class LandmarkExtractor:
    """FaceMesh with fallbacks, so fewer faces are lost:
    1) normal detection  2) relaxed detection threshold  3) zoom out (add border) + relaxed"""

    def __init__(self):
        def mk(conf):
            return mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1, refine_landmarks=False,
                min_detection_confidence=conf)
        self.mesh = mk(0.5)
        self.mesh_lo = mk(0.2)

    @staticmethod
    def _to_px(res, w, h, ox=0, oy=0):
        lm = res.multi_face_landmarks[0].landmark
        return np.array([[p.x * w - ox, p.y * h - oy] for p in lm], dtype=np.float32)

    def raw(self, rgb):
        h, w = rgb.shape[:2]
        for mesh in (self.mesh, self.mesh_lo):
            res = mesh.process(rgb)
            if res.multi_face_landmarks:
                return self._to_px(res, w, h)
        pad = int(0.5 * max(h, w))
        big = cv2.copyMakeBorder(rgb, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        res = self.mesh_lo.process(big)
        if res.multi_face_landmarks:
            return self._to_px(res, big.shape[1], big.shape[0], pad, pad)
        return None

    def features(self, rgb):
        pts = self.raw(rgb)
        return None if pts is None else geometric_features(pts)