"""
evaluate.py — measure the finger-counting pipeline on photo and video datasets.

Photo dataset (accuracy + confusion matrix):
    python evaluate.py images --data path/to/dataset --mode ycrcb
    python evaluate.py images --data path/to/fingers/test --mode otsu      # Kaggle "Fingers" dataset

  Labels are read either from the folder name (dataset/3/img001.jpg -> 3)
  or from the file name (..._3L.png, 3_hand.jpg, hand-3.png -> 3).

Video (annotated output video + per-frame CSV):
    python evaluate.py video --input clip.mp4 --out results/annotated.mp4

Live webcam window (backup demo for the viva):
    python evaluate.py webcam
"""
import argparse
import csv
import re
import time
from pathlib import Path

import cv2
import numpy as np

from hand_pipeline import detect, draw, DEFAULT_PARAMS

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASSES = list(range(6))  # 0..5 fingers


def label_from_path(path: Path):
    if path.parent.name.isdigit():
        return int(path.parent.name)
    m = re.search(r"(?:^|[_\-\s])([0-5])(?:[LR])?(?=\.|[_\-\s]|$)", path.stem)
    return int(m.group(1)) if m else None


def save_confusion(cm, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.imshow(cm, cmap="Oranges")
        ax.set_xticks(CLASSES); ax.set_yticks(CLASSES)
        ax.set_xlabel("Predicted fingers"); ax.set_ylabel("True fingers")
        ax.set_title("Confusion matrix")
        for i in CLASSES:
            for j in CLASSES:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
    except ImportError:  # draw it with OpenCV instead
        cell = 60
        img = np.full((cell * 7, cell * 7, 3), 255, np.uint8)
        mx = max(cm.max(), 1)
        for i in CLASSES:
            for j in CLASSES:
                shade = int(255 - 200 * cm[i, j] / mx)
                x, y = (j + 1) * cell, (i + 1) * cell
                cv2.rectangle(img, (x, y), (x + cell, y + cell), (shade, shade + 20 if shade < 235 else 255, 255), -1)
                cv2.putText(img, str(int(cm[i, j])), (x + 18, y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            cv2.putText(img, str(i), (24, (i + 1) * cell + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(img, str(i), ((i + 1) * cell + 24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.imwrite(str(out_png), img)


def run_images(args):
    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "samples").mkdir(exist_ok=True)

    files = sorted(p for p in data.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if args.limit:
        files = files[: args.limit]
    cm = np.zeros((6, 6), int)
    rows, times, skipped, missed = [], [], 0, 0

    for idx, path in enumerate(files):
        label = label_from_path(path)
        img = cv2.imread(str(path), cv2.IMREAD_COLOR if args.mode == "ycrcb" else cv2.IMREAD_GRAYSCALE)
        if label is None or img is None:
            skipped += 1
            continue
        t0 = time.perf_counter()
        r, _, _ = detect(img, args.mode, DEFAULT_PARAMS)
        times.append((time.perf_counter() - t0) * 1000)
        pred = r.fingers if r.found else 0
        missed += not r.found
        cm[label, pred] += 1
        rows.append([str(path), label, pred, int(r.found), f"{times[-1]:.2f}"])
        if idx < args.save_samples or (label != pred and len(list((out / "samples").glob("wrong_*"))) < 20):
            prefix = "wrong_" if label != pred else "ok_"
            cv2.imwrite(str(out / "samples" / f"{prefix}{idx:05d}.png"), draw(img, r, label))

    total = cm.sum()
    if total == 0:
        print("No labelled images found. Use folders named 0-5 or put the count in the file name.")
        return
    acc = np.trace(cm) / total
    print(f"\nImages evaluated : {total}   (skipped {skipped} without a label)")
    print(f"No hand detected : {missed}")
    print(f"Overall accuracy : {acc * 100:.1f}%")
    print(f"Mean time/image  : {np.mean(times):.2f} ms  ({1000 / np.mean(times):.0f} images/s)\n")
    print("Per-class accuracy:")
    for c in CLASSES:
        n = cm[c].sum()
        if n:
            print(f"  {c} fingers: {cm[c, c] / n * 100:5.1f}%  ({cm[c, c]}/{n})")
    print("\nConfusion matrix (rows = true, columns = predicted):")
    print("      " + " ".join(f"{c:>5}" for c in CLASSES))
    for c in CLASSES:
        print(f"  {c}:  " + " ".join(f"{v:>5}" for v in cm[c]))

    with open(out / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "true", "predicted", "hand_found", "ms"])
        w.writerows(rows)
    save_confusion(cm, out / "confusion_matrix.png")
    with open(out / "summary.txt", "w") as f:
        f.write(f"images={total}\naccuracy={acc:.4f}\nmean_ms={np.mean(times):.2f}\nmode={args.mode}\n")
    print(f"\nSaved predictions.csv, confusion_matrix.png, summary.txt and sample images to {out}/")


def run_video(args, source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer, rows, frame_no, times = None, [], 0, []
    live = source == 0
    if not live and args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if live:
            frame = cv2.flip(frame, 1)
        t0 = time.perf_counter()
        r, raw, clean = detect(frame, "ycrcb", DEFAULT_PARAMS)
        times.append((time.perf_counter() - t0) * 1000)
        vis = draw(frame, r)
        rows.append([frame_no, round(frame_no / fps, 3), int(r.found), r.fingers if r.found else "",
                     round(r.cx, 1), round(r.cy, 1), round(r.angle, 1)])
        frame_no += 1

        if live:
            mask_small = cv2.cvtColor(cv2.resize(clean, (vis.shape[1] // 3, vis.shape[0] // 3)), cv2.COLOR_GRAY2BGR)
            vis[-mask_small.shape[0]:, -mask_small.shape[1]:] = mask_small
            cv2.imshow("HandChord - OpenCV (press q to quit)", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        elif args.out:
            if writer is None:
                writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (vis.shape[1], vis.shape[0]))
            writer.write(vis)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    if not live:
        csv_path = Path(args.out).with_suffix(".csv") if args.out else Path("video_results.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "time_s", "hand_found", "fingers", "cx", "cy", "angle_deg"])
            w.writerows(rows)
        found = sum(r[2] for r in rows)
        print(f"Frames: {frame_no}   hand found in {found} ({found / max(frame_no, 1) * 100:.1f}%)")
        print(f"Mean processing time: {np.mean(times):.2f} ms/frame")
        print(f"Saved {args.out} and {csv_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("images", help="evaluate on a labelled photo dataset")
    a.add_argument("--data", required=True)
    a.add_argument("--mode", choices=["ycrcb", "otsu"], default="ycrcb")
    a.add_argument("--out", default="results")
    a.add_argument("--limit", type=int, default=0, help="only use the first N images")
    a.add_argument("--save-samples", type=int, default=12)
    v = sub.add_parser("video", help="annotate a video file")
    v.add_argument("--input", required=True)
    v.add_argument("--out", default="results/annotated.mp4")
    sub.add_parser("webcam", help="live demo window")
    args = ap.parse_args()

    if args.cmd == "images":
        run_images(args)
    elif args.cmd == "video":
        run_video(args, args.input)
    else:
        args.out = None
        run_video(args, 0)


if __name__ == "__main__":
    main()
