"""
prepare_data.py — turn the HaGRID 30k sample into hand crops for finger-count training.

Dataset: HaGRID (HAnd Gesture Recognition Image Dataset), 31,833-image sample at 384p
         https://huggingface.co/datasets/cj-mills/hagrid-sample-30k-384p  (CC-BY-SA 4.0)

Output:  <out>/crops/<id>.jpg (160x160, hand box x2 context) and <out>/labels.csv with columns
         file, label, split, user_id, source_gesture

Classes (what the synth plays):
    0 = fist           1 = one           2 = peace / two_up (+ inverted)
    3 = three / three2 4 = four          5 = palm / stop (+ inverted)
    6 = none  -> other gestures (call, like, ok, rock...), relaxed "no_gesture" hands,
                 and random background patches (faces, bodies, furniture, objects)

Split: 80/20 BY PERSON (user_id). Nobody in the test set appears in training,
so test accuracy measures how well the model works on new people.

    python prepare_data.py --zip hagrid-sample-30k-384p.zip --out data
"""
import argparse
import csv
import hashlib
import json
import random
import time
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

CLASS_NAMES = ["0", "1", "2", "3", "4", "5", "none"]
GESTURE_TO_CLASS = {
    "fist": 0, "one": 1,
    "peace": 2, "peace_inverted": 2, "two_up": 2, "two_up_inverted": 2,
    "three": 3, "three2": 3, "four": 4,
    "palm": 5, "stop": 5, "stop_inverted": 5,
    "call": 6, "dislike": 6, "like": 6, "mute": 6, "ok": 6, "rock": 6, "no_gesture": 6,
}
SIZE = 160
CONTEXT = 2.0  # crop side = 2 x hand box, so training can rotate and zoom without cutting fingers off


def user_split(user_id: str) -> str:
    """Deterministic 80/20 split on the person, not the image."""
    h = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 100
    return "test" if h < 20 else "train"


def square_crop(img, cx, cy, side):
    h, w = img.shape[:2]
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = x0 + int(round(side)), y0 + int(round(side))
    pad = max(0, -x0, -y0, x1 - w, y1 - h)
    if pad:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        x0, y0, x1, y1 = x0 + pad, y0 + pad, x1 + pad, y1 + pad
    crop = img[y0:y1, x0:x1]
    return cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def hand_crops(img, box, n, rng):
    """Wide square crops centred on a hand. train.py later rotates, zooms and shifts them."""
    H, W = img.shape[:2]
    x, y, bw, bh = box[0] * W, box[1] * H, box[2] * W, box[3] * H
    out = []
    for _ in range(n):
        side = max(bw, bh) * CONTEXT * rng.uniform(0.95, 1.05)
        cx = x + bw / 2 + rng.uniform(-0.04, 0.04) * side
        cy = y + bh / 2 + rng.uniform(-0.04, 0.04) * side
        out.append(square_crop(img, cx, cy, side))
    return out


def background_crop(img, boxes, rng, tries=20):
    """A random square that does not overlap any hand: faces, bodies, rooms, objects."""
    H, W = img.shape[:2]
    hb = [(b[0] * W, b[1] * H, (b[0] + b[2]) * W, (b[1] + b[3]) * H) for b in boxes]
    for _ in range(tries):
        side = rng.uniform(0.2, 0.6) * H
        x0, y0 = rng.uniform(0, W - side), rng.uniform(0, H - side)
        x1, y1 = x0 + side, y0 + side
        if all(x1 < a or x0 > c or y1 < b or y0 > d for a, b, c, d in hb):
            return square_crop(img, x0 + side / 2, y0 + side / 2, side)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--bg-prob", type=float, default=0.18, help="chance of also taking a background patch per image")
    ap.add_argument("--time-budget", type=float, default=0, help="stop after N seconds and resume on the next run (0 = no limit)")
    args = ap.parse_args()

    t_start = time.time()
    out = Path(args.out)
    (out / "crops").mkdir(parents=True, exist_ok=True)
    zf = zipfile.ZipFile(args.zip)
    names = zf.namelist()

    # Annotations: one JSON per gesture, {image_id: {bboxes, labels, user_id, ...}}.
    # They cover the full HaGRID set, so keep only the images that are in this sample.
    images = {Path(n).stem: n for n in names if n.lower().endswith((".jpg", ".jpeg", ".png"))}
    ann = {}
    for n in names:
        if n.endswith(".json"):
            ann.update({k: v for k, v in json.loads(zf.read(n)).items() if k in images})
    ids = sorted(ann)
    print(f"{len(ids)} annotated images in the sample")

    # Resume support (long jobs can be split into several shorter runs).
    state_f = out / "progress.json"
    state = json.loads(state_f.read_text()) if state_f.exists() else {"next": 0, "idx": 0}
    if state["next"] >= len(ids):
        print("Already done."); summarize(out); return

    # Rare-but-important classes get 2 crops per hand so the classes are less imbalanced.
    per_hand = {0: 2, 1: 2, 2: 1, 3: 1, 4: 2, 5: 1, 6: 1}
    rows, idx = [], state["idx"]
    k = state["next"]
    while k < len(ids):
        if args.time_budget and time.time() - t_start > args.time_budget:
            break
        image_id = ids[k]; a = ann[image_id]; path = images[image_id]; k += 1
        rng = random.Random(image_id)  # per-image seed: same crops whatever the run boundaries
        img = cv2.imdecode(np.frombuffer(zf.read(path), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        split = user_split(a.get("user_id", image_id))
        for box, gesture in zip(a["bboxes"], a["labels"]):
            cls = GESTURE_TO_CLASS.get(gesture)
            if cls is None:
                continue
            # The dataset has many relaxed "no_gesture" hands; keep a third of them.
            if gesture == "no_gesture" and rng.random() > 0.33:
                continue
            for crop in hand_crops(img, box, per_hand[cls] if split == "train" else 1, rng):
                f = f"{idx:06d}.jpg"; idx += 1
                cv2.imwrite(str(out / "crops" / f), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                rows.append([f, cls, split, a.get("user_id", ""), gesture])
        if rng.random() < args.bg_prob:
            crop = background_crop(img, a["bboxes"], rng)
            if crop is not None:
                f = f"{idx:06d}.jpg"; idx += 1
                cv2.imwrite(str(out / "crops" / f), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                rows.append([f, 6, split, a.get("user_id", ""), "background"])

    new_file = not (out / "labels.csv").exists()
    with open(out / "labels.csv", "a", newline="") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(["file", "label", "split", "user_id", "source_gesture"])
        w.writerows(rows)
    state_f.write_text(json.dumps({"next": k, "idx": idx}))
    print(f"Processed images up to {k}/{len(ids)} ({idx} crops so far)")
    if k >= len(ids):
        summarize(out)


def summarize(out):
    with open(out / "labels.csv") as fh:
        rows = [(r["file"], int(r["label"]), r["split"], r["user_id"]) for r in csv.DictReader(fh)]
    counts = Counter((r[2], r[1]) for r in rows)
    users = {(r[2], r[3]) for r in rows}
    print(f"\nTotal crops: {len(rows)}")
    print(f"People: train {sum(1 for s, _ in users if s == 'train')}, test {sum(1 for s, _ in users if s == 'test')}")
    for split in ("train", "test"):
        print(f"{split:>5}: " + "  ".join(f"{CLASS_NAMES[c]}={counts[(split, c)]}" for c in range(7)))


if __name__ == "__main__":
    main()
