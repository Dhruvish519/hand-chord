"""
make_synthetic_dataset.py — draws simple cartoon hands with 0-5 fingers.

Only a sanity check that the code runs end to end. Report accuracy on REAL photos
(your own captured dataset or the Kaggle "Fingers" dataset), not on these.

    python make_synthetic_dataset.py --out synthetic --per-class 40
    python evaluate.py images --data synthetic --mode ycrcb
"""
import argparse
import math
import random
from pathlib import Path

import cv2
import numpy as np


def draw_hand(n, size=240, gray=False, rng=random):
    bg = tuple(int(v) for v in rng.choice([(110, 100, 90), (60, 70, 80), (150, 140, 130), (40, 40, 40)]))
    skin = tuple(int(v + rng.uniform(-12, 12)) for v in (140, 172, 224))  # BGR
    img = np.full((size, size, 3), bg, np.uint8)
    s = size / 240 * rng.uniform(0.85, 1.1)
    cx, cy = int(size / 2 + rng.uniform(-20, 20)), int(size * 0.62 + rng.uniform(-10, 10))
    tilt = rng.uniform(-20, 20)
    cv2.circle(img, (cx, cy), int(42 * s), skin, -1)
    pts = cv2.boxPoints(((cx - math.sin(math.radians(tilt)) * 60 * s, cy + 60 * s), (66 * s, 120 * s), tilt))
    cv2.fillPoly(img, [pts.astype(np.int32)], skin)
    total = min(128, 40 * (n - 1)) if n > 1 else 0
    for i in range(n):
        a0 = 0 if n == 1 else -total / 2 + total * i / (n - 1)
        a = math.radians(a0 + tilt + rng.uniform(-4, 4))
        length = 100 * s * rng.uniform(0.85, 1.05)
        tip = (int(cx + math.sin(a) * length), int(cy - math.cos(a) * length))
        cv2.line(img, (cx, cy), tip, skin, int(15 * s))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = np.clip(img + np.random.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if gray else img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synthetic")
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--gray", action="store_true", help="grayscale, Kaggle-style file names (..._3L.png)")
    args = ap.parse_args()
    random.seed(0); np.random.seed(0)
    out = Path(args.out)
    for n in range(6):
        for k in range(args.per_class):
            img = draw_hand(n, gray=args.gray)
            if args.gray:
                out.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out / f"img{n}{k:04d}_{n}L.png"), img)
            else:
                (out / str(n)).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out / str(n) / f"{k:04d}.png"), img)
    print(f"Wrote {6 * args.per_class} images to {out}/")


if __name__ == "__main__":
    main()
