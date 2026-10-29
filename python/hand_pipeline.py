"""
hand_pipeline.py — the same classical hand pipeline as public/js/handTracker.js, in Python OpenCV.

Used by evaluate.py to measure accuracy on photo and video datasets.

Steps
  1. Segment the hand into a binary mask
       - "ycrcb": skin colour threshold on Cr/Cb (colour photos, webcam, videos)
       - "otsu":  automatic global threshold (grayscale images with plain backgrounds,
                  e.g. the Kaggle "Fingers" dataset)
  2. Clean the mask: median blur, morphological opening, closing
  3. Largest contour = hand
  4. Moments -> centroid and orientation
  5. Convex hull + convexity defects -> number of fingers
"""
from dataclasses import dataclass, field
import math

import cv2
import numpy as np

DEFAULT_PARAMS = dict(
    cr_min=135, cr_max=180,
    cb_min=85, cb_max=135,
    kernel_size=5,
    min_area_ratio=0.02,
    defect_depth_ratio=0.18,
    max_defect_angle=90,
)


@dataclass
class HandResult:
    found: bool = False
    fingers: int = 0
    cx: float = 0.0
    cy: float = 0.0
    angle: float = 0.0
    contour: np.ndarray | None = None
    hull: np.ndarray | None = None
    tips: list = field(default_factory=list)
    valleys: list = field(default_factory=list)


def segment(image, mode="ycrcb", p=DEFAULT_PARAMS):
    """Return (raw_mask, clean_mask) as uint8 images with values 0/255."""
    if mode == "ycrcb":
        bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        lower = np.array([0, p["cr_min"], p["cb_min"]], np.uint8)
        upper = np.array([255, p["cr_max"], p["cb_max"]], np.uint8)
        raw = cv2.inRange(ycrcb, lower, upper)
    elif mode == "otsu":
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, raw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # If the hand came out dark on a light background, invert so the hand is white.
        if cv2.countNonZero(raw) > raw.size / 2:
            raw = cv2.bitwise_not(raw)
    else:
        raise ValueError(f"unknown mode {mode}")

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p["kernel_size"], p["kernel_size"]))
    clean = cv2.medianBlur(raw, 5)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, k)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k, iterations=2)
    return raw, clean


def _angle_at(a, b, c):
    ab, cb, ac = math.dist(a, b), math.dist(c, b), math.dist(a, c)
    cos = (ab * ab + cb * cb - ac * ac) / (2 * ab * cb + 1e-6)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def analyse_mask(mask, p=DEFAULT_PARAMS):
    r = HandResult()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return r
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < p["min_area_ratio"] * mask.shape[0] * mask.shape[1]:
        return r

    r.found, r.contour = True, cnt
    m = cv2.moments(cnt)
    r.cx, r.cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    theta = 0.5 * math.atan2(2 * m["mu11"], m["mu20"] - m["mu02"])
    deg = math.degrees(theta)
    r.angle = deg - 90 if deg > 0 else deg + 90  # 0 = upright, + = tilted right

    hull_idx = cv2.convexHull(cnt, returnPoints=False)
    r.hull = cnt[hull_idx.reshape(-1)]
    x, y, w, h = cv2.boundingRect(cnt)
    min_depth = p["defect_depth_ratio"] * h

    gaps = 0
    if len(hull_idx) > 3:
        try:
            defects = cv2.convexityDefects(cnt, hull_idx)
        except cv2.error:
            defects = None
        if defects is not None:
            for s_i, e_i, f_i, depth in defects.reshape(-1, 4):  # works on OpenCV 4 and 5
                s, e, f = (tuple(int(v) for v in cnt[i][0]) for i in (s_i, e_i, f_i))
                if (depth / 256 > min_depth and _angle_at(s, f, e) < p["max_defect_angle"]
                        and f[1] < y + h * 0.85):
                    gaps += 1
                    r.valleys.append(f)
                    r.tips += [s, e]

    if gaps:
        r.fingers = min(5, gaps + 1)
    else:
        solidity = area / (cv2.contourArea(r.hull) + 1e-6)
        r.fingers = 1 if (h / w > 1.6 and solidity < 0.85) else 0
    return r


def detect(image, mode="ycrcb", p=DEFAULT_PARAMS):
    raw, clean = segment(image, mode, p)
    return analyse_mask(clean, p), raw, clean


def draw(image, r, label=None):
    out = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if r.found:
        cv2.drawContours(out, [r.contour], -1, (71, 181, 255), 2)
        cv2.drawContours(out, [r.hull], -1, (196, 230, 94), 2)
        for v in r.valleys:
            cv2.circle(out, v, 5, (107, 107, 255), -1)
        for t in r.tips:
            cv2.circle(out, t, 6, (138, 210, 255), -1)
        cv2.circle(out, (int(r.cx), int(r.cy)), 5, (230, 235, 236), -1)
    text = f"fingers: {r.fingers}" if r.found else "no hand"
    if label is not None:
        text += f"  (true {label})"
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return out
