"""Общие утилиты: калибровка (дисторсия + гомографии камера->план), маски, время."""
import json
import os
import numpy as np
import cv2

CAMERAS = [1, 2, 3, 4, 5]
CANON = (1600, 1200)          # канонический размер кадра (из ТЗ); все расчёты ведутся в нём
PLAN_SIZE = (180, 440)        # officePlan.jpg: ширина x высота, пиксели плана
CALIB_FILE = "calibration.json"


def video_path(root, cam):
    return os.path.join(root, f"camera_new_{cam}.mkv")


def fmt_time(frame, fps):
    """чч:мм:сс:мс (время от начала записи)."""
    ms = int(round(frame / fps * 1000.0))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}:{ms:03d}"


# ---------------------------------------------------------------- дисторсия
# Однопараметрическая "division model":  p_u = c + (p_d - c) / (1 + k * r_d^2),
# r_d = |p_d - c| / R,  R = половина диагонали кадра.  k < 0 -> бочка (широкоугольник).
def _geom():
    W, H = CANON
    return np.array([W / 2.0, H / 2.0]), float(np.hypot(W, H) / 2.0)


def undistort_px(pts, k):
    """pts: (N,2) пиксели канонического кадра (искажённые) -> неискажённые."""
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    c, R = _geom()
    d = pts - c
    r2 = (np.linalg.norm(d, axis=1) / R) ** 2
    return c + d / (1.0 + k * r2)[:, None]


def distort_px(pts, k):
    """Обратное преобразование (нужно для отрисовки проверок и remap)."""
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    c, R = _geom()
    d = pts - c
    ru = np.linalg.norm(d, axis=1) / R
    if abs(k) < 1e-9:
        return pts.copy()
    disc = np.clip(1.0 - 4.0 * k * ru ** 2, 0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        rd = np.where(ru > 1e-9, (1.0 - np.sqrt(disc)) / (2.0 * k * ru), 0.0)
    scale = np.where(ru > 1e-9, rd / np.maximum(ru, 1e-9), 1.0)
    return c + d * scale[:, None]


def undistort_image(img, k):
    """Для визуальной проверки коэффициента (кадр -> неискажённый кадр)."""
    h, w = img.shape[:2]
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    pts = np.stack([gx.ravel() * CANON[0] / w, gy.ravel() * CANON[1] / h], 1)
    src = distort_px(pts, k)
    mapx = (src[:, 0] * w / CANON[0]).reshape(h, w).astype(np.float32)
    mapy = (src[:, 1] * h / CANON[1]).reshape(h, w).astype(np.float32)
    return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)


# ---------------------------------------------------------------- калибровка
class Calibration:
    def __init__(self, data=None):
        self.data = data or {"k1": 0.0, "plan_size": list(PLAN_SIZE),
                             "cameras": {str(c): {"pairs": [], "ghost_polygons": [], "weight": 1.0}
                                         for c in CAMERAS},
                             "walkable": [], "obstacles": []}
        self._H = {}
        self._mask_cache = {}
        self._ghost = {}

    # --- io
    @staticmethod
    def load(path=CALIB_FILE):
        if not os.path.exists(path):
            return Calibration()
        with open(path, "r", encoding="utf-8") as f:
            cal = Calibration(json.load(f))
        for c in CAMERAS:
            cal.data["cameras"].setdefault(str(c), {"pairs": [], "ghost_polygons": [], "weight": 1.0})
        return cal

    def save(self, path=CALIB_FILE):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=1)

    @property
    def k1(self):
        return float(self.data.get("k1", 0.0))

    def cam(self, c):
        return self.data["cameras"][str(c)]

    def weight(self, c):
        return float(self.cam(c).get("weight", 1.0))

    # --- гомография камера -> план
    def homography(self, c):
        if c in self._H:
            return self._H[c]
        pairs = self.cam(c)["pairs"]
        H = None
        if len(pairs) >= 4:
            src = np.array([[p["img"][0] * CANON[0], p["img"][1] * CANON[1]] for p in pairs])
            src = undistort_px(src, self.k1)
            dst = np.array([p["plan"] for p in pairs], np.float64)
            H, _ = cv2.findHomography(src, dst, cv2.RANSAC if len(pairs) > 4 else 0, 4.0)
        self._H[c] = H
        return H

    def reproj_error(self, c):
        H = self.homography(c)
        pairs = self.cam(c)["pairs"]
        if H is None or not pairs:
            return None
        src = undistort_px([[p["img"][0] * CANON[0], p["img"][1] * CANON[1]] for p in pairs], self.k1)
        pr = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
        return float(np.mean(np.linalg.norm(pr - np.array([p["plan"] for p in pairs]), axis=1)))

    def is_calibrated(self, c):
        return self.homography(c) is not None

    def to_plan(self, c, pts_px):
        """Точки (канонические пиксели кадра, искажённые) -> координаты плана."""
        H = self.homography(c)
        u = undistort_px(pts_px, self.k1)
        return cv2.perspectiveTransform(u.reshape(-1, 1, 2), H).reshape(-1, 2)

    def plan_to_cam(self, c, plan_pts):
        H = self.homography(c)
        p = cv2.perspectiveTransform(np.asarray(plan_pts, np.float64).reshape(-1, 1, 2),
                                     np.linalg.inv(H)).reshape(-1, 2)
        return distort_px(p, self.k1)

    # --- проходимая область плана
    def walkable_mask(self, margin=0):
        key = int(margin)
        if key in self._mask_cache:
            return self._mask_cache[key]
        w, h = self.data.get("plan_size", PLAN_SIZE)
        m = np.zeros((h, w), np.uint8)
        for poly in self.data.get("walkable", []):
            cv2.fillPoly(m, [np.round(np.array(poly)).astype(np.int32)], 1)
        for poly in self.data.get("obstacles", []):
            cv2.fillPoly(m, [np.round(np.array(poly)).astype(np.int32)], 0)
        if margin > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * key + 1, 2 * key + 1))
            m = cv2.dilate(m, k)
        self._mask_cache[key] = m
        return m

    def is_walkable(self, xy, margin=0):
        if not self.data.get("walkable"):
            return True                      # полигон не размечен -> не фильтруем
        m = self.walkable_mask(margin)
        x, y = int(round(xy[0])), int(round(xy[1]))
        return 0 <= y < m.shape[0] and 0 <= x < m.shape[1] and m[y, x] > 0

    # --- статичные маски "призраков" (ТВ и т.п.) в пикселях камеры
    def ghost_mask(self, c):
        if c not in self._ghost:
            m = np.zeros((CANON[1], CANON[0]), np.uint8)
            for poly in self.cam(c).get("ghost_polygons", []):
                pts = (np.array(poly) * np.array(CANON)).round().astype(np.int32)
                cv2.fillPoly(m, [pts], 1)
            self._ghost[c] = m
        return self._ghost[c]
