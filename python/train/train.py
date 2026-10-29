"""
train.py — fine-tune MobileNetV3-Small to (a) count fingers 0-5 or "none" and
(b) estimate how far the hand is tilted, at ANY rotation.

Data:   crops from prepare_data.py (160x160, hand box with 2x context).
Split:  80/20 by person (prepare_data.py), test people never seen in training.

Rotation training: every training crop is rotated by a random angle (mostly within
+-90 degrees, sometimes all the way round), slightly perspective-warped (hand turned
toward/away from the camera), zoomed and shifted. Because we rotate it ourselves we
KNOW the rotation, so the network also learns to output the hand's tilt angle,
as (sin, cos) so that -180 and +180 degrees are the same thing.

Model outputs 9 numbers: 7 class scores (0..5 fingers, none) + sin(tilt) + cos(tilt).
Tilt convention: degrees, 0 = upright, positive = rotated clockwise on screen.

    python train.py --data data --out model --epochs 6 --init model_v1/handchord.pt
"""
import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "mobilenetv3.pytorch"))
from mobilenetv3 import mobilenetv3_small  # noqa: E402

CLASS_NAMES = ["0", "1", "2", "3", "4", "5", "none"]
NONE = 6
N_OUT = 9
SIZE = 112             # network input
CONTEXT = 2.0          # crops on disk are 2x the hand box (see prepare_data.py)
TEST_ZOOM = 1.45       # test view: square of 1.45x the hand box, like the website's crop
EVAL_ANGLES = [-135, -90, -45, 0, 45, 90, 135, 180]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def make_view(img, tilt_deg=0.0, zoom=TEST_ZOOM, shift=(0.0, 0.0), persp=0.0, rng=None):
    """Rotate the wide crop by tilt_deg (clockwise), optional perspective warp, then cut the input square."""
    S = img.shape[0]
    c = S / 2
    if tilt_deg:
        M = cv2.getRotationMatrix2D((c, c), -tilt_deg, 1.0)  # OpenCV: positive = counter-clockwise
        img = cv2.warpAffine(img, M, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if persp:
        src = np.float32([[0, 0], [S, 0], [S, S], [0, S]])
        dst = src + np.float32(rng.uniform(-persp, persp, (4, 2)) * S)
        img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (S, S), borderMode=cv2.BORDER_REPLICATE)
    side = S * zoom / CONTEXT
    x0 = c + shift[0] * side - side / 2
    y0 = c + shift[1] * side - side / 2
    M = np.float32([[SIZE / side, 0, -x0 * SIZE / side], [0, SIZE / side, -y0 * SIZE / side]])
    return cv2.warpAffine(img, M, (SIZE, SIZE), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)


def to_tensor(img_rgb_float):
    x = (img_rgb_float / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


class Crops(Dataset):
    def __init__(self, root, rows, train):
        self.root, self.rows, self.train = Path(root), rows, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        f, label = self.rows[i]
        img = cv2.imread(str(self.root / "crops" / f))[:, :, ::-1]  # BGR -> RGB
        if not self.train:
            return to_tensor(make_view(img).astype(np.float32)), label, 0.0, 1.0, 0.0
        rng = np.random
        if rng.rand() < 0.5:
            img = img[:, ::-1]                                 # left / right hand
        r = rng.rand()                                         # rotation mix
        tilt = rng.uniform(-15, 15) if r < 0.25 else rng.uniform(-90, 90) if r < 0.75 else rng.uniform(-180, 180)
        view = make_view(np.ascontiguousarray(img), tilt, zoom=rng.uniform(1.2, 1.75),
                         shift=rng.uniform(-0.1, 0.1, 2), persp=rng.uniform(0, 0.08), rng=rng)
        v = view.astype(np.float32)
        v = v * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25)              # contrast, brightness
        v = v * rng.uniform(0.9, 1.1, 3).astype(np.float32)               # white balance
        if rng.rand() < 0.25:
            v = cv2.GaussianBlur(v, (3, 3), rng.uniform(0.3, 1.2))        # motion / focus blur
        v = np.clip(v, 0, 255)
        t = math.radians(tilt)
        has_angle = 0.0 if label == NONE else 1.0
        return to_tensor(v), label, math.sin(t), math.cos(t), has_angle


def build_model(init=None, imagenet=None, n_out=N_OUT):
    model = mobilenetv3_small()
    if imagenet and Path(imagenet).exists() and not init:
        model.load_state_dict(torch.load(imagenet, map_location="cpu"))
        print("Loaded ImageNet weights")
    old_last = model.classifier[-1]
    model.classifier[-1] = nn.Linear(old_last.in_features, n_out)
    if init and Path(init).exists():
        sd = torch.load(init, map_location="cpu")
        w, b = sd.pop("classifier.3.weight"), sd.pop("classifier.3.bias")
        model.load_state_dict(sd, strict=False)
        with torch.no_grad():  # keep the old 7 class outputs, new angle outputs start at zero
            model.classifier[-1].weight[: w.shape[0]] = w
            model.classifier[-1].bias[: b.shape[0]] = b
            model.classifier[-1].weight[w.shape[0]:] = 0
            model.classifier[-1].bias[w.shape[0]:] = 0
        print(f"Initialised from {init}")
    return model


def predict(model, x):
    out = model(x)
    cls = out[:, :7].argmax(1)
    ang = torch.rad2deg(torch.atan2(out[:, 7], out[:, 8])) if out.shape[1] >= 9 else None
    return cls, ang


def evaluate(model, loader):
    model.eval()
    cm = np.zeros((7, 7), int)
    with torch.no_grad():
        for x, y, *_ in loader:
            p, _ = predict(model, x)
            for t, q in zip(y.numpy(), p.numpy()):
                cm[t, q] += 1
    return cm


def rotation_eval(model, root, rows, n=2400, seed=0):
    """Accuracy at fixed rotations on test crops; also tilt-angle error for counting hands."""
    rng = random.Random(seed)
    rows = rng.sample(rows, min(n, len(rows)))
    imgs = [cv2.imread(str(Path(root) / "crops" / f))[:, :, ::-1] for f, _ in rows]
    labels = np.array([l for _, l in rows])
    model.eval()
    res = {}
    with torch.no_grad():
        for a in EVAL_ANGLES:
            preds, angs = [], []
            for i in range(0, len(imgs), 128):
                x = torch.stack([to_tensor(make_view(im, a).astype(np.float32)) for im in imgs[i:i + 128]])
                p, ang = predict(model, x)
                preds.append(p.numpy())
                if ang is not None:
                    angs.append(ang.numpy())
            preds = np.concatenate(preds)
            entry = dict(accuracy=round(float((preds == labels).mean()), 4))
            if angs:
                angs = np.concatenate(angs)
                m = labels != NONE
                err = np.abs((angs[m] - a + 180) % 360 - 180)
                entry["tilt_mae_deg"] = round(float(err.mean()), 1)
                entry["tilt_within_15deg"] = round(float((err <= 15).mean()), 4)
            res[str(a)] = entry
            print(f"  rotation {a:>5} deg: {entry}", flush=True)
    return res


def report(cm):
    acc = np.trace(cm) / cm.sum()
    per = {}
    for c, name in enumerate(CLASS_NAMES):
        tp = cm[c, c]
        prec = tp / max(cm[:, c].sum(), 1)
        rec = tp / max(cm[c].sum(), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        per[name] = dict(precision=round(float(prec), 4), recall=round(float(rec), 4),
                         f1=round(float(f1), 4), support=int(cm[c].sum()))
    return float(acc), per


def save_confusion(cm, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6, 5.2))
    ax.imshow(norm, cmap="Oranges", vmin=0, vmax=1)
    ax.set_xticks(range(7), CLASS_NAMES); ax.set_yticks(range(7), CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j] * 100:.0f}%", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def save_rotation_plot(new, old, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    xs = EVAL_ANGLES
    ax.plot(xs, [new[str(a)]["accuracy"] * 100 for a in xs], "o-", label="New model (rotation training)", color="#e07b00")
    if old:
        ax.plot(xs, [old[str(a)]["accuracy"] * 100 for a in xs], "s--", label="Previous model", color="#888")
    ax.set_xticks(xs); ax.set_xlabel("Hand rotation (degrees)"); ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(0, 100); ax.grid(alpha=0.3); ax.legend(loc="lower center")
    ax.set_title("Accuracy vs hand rotation (unseen test people)")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="model")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--angle-weight", type=float, default=1.0)
    ap.add_argument("--init", default="", help="previous handchord.pt to start from (optional)")
    ap.add_argument("--compare", default="", help="previous handchord.pt (7 outputs) to compare on rotated test crops")
    ap.add_argument("--time-budget", type=float, default=0, help="stop after N seconds; run again to continue")
    ap.add_argument("--imagenet", default=str(HERE / "mobilenetv3.pytorch/pretrained/mobilenetv3-small-55df8e1f.pth"))
    args = ap.parse_args()
    t_start = time.time()
    out_of_time = lambda: args.time_budget and time.time() - t_start > args.time_budget
    torch.set_num_threads(4)

    rows = {"train": [], "test": []}
    with open(Path(args.data) / "labels.csv") as fh:
        for r in csv.DictReader(fh):
            rows[r["split"]].append((r["file"], int(r["label"])))
    n_tr, n_te = len(rows["train"]), len(rows["test"])
    counts = np.bincount([l for _, l in rows["train"]], minlength=7)
    weights = torch.tensor(counts.sum() / (7 * np.maximum(counts, 1)), dtype=torch.float32)
    out = Path(args.out); out.mkdir(exist_ok=True)
    steps_per_epoch = n_tr // args.batch

    ckpt_f = out / "checkpoint.pt"
    fresh = not ckpt_f.exists()
    model = build_model(args.init if fresh else None, args.imagenet if fresh else None)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.15)
    if fresh:
        print(f"Train crops: {n_tr}   Test crops: {n_te}   (80/20 split by person)")
        st = dict(epoch=1, step=0, phase="train", loss_sum=0.0, correct=0, seen=0, ang_sum=0.0, ep_seconds=0.0, history=[])
    else:
        ck = torch.load(ckpt_f, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        st = ck["state"]

    def save():
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), state=st), ckpt_f)

    ce = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    te = DataLoader(Crops(args.data, rows["test"], False), batch_size=128, num_workers=args.workers)

    while st["phase"] != "done" and not out_of_time():
        if st["phase"] == "train":
            ep = st["epoch"]
            order = np.random.RandomState(ep).permutation(n_tr)[: steps_per_epoch * args.batch]
            remaining = order[st["step"] * args.batch:]
            seed = ep * 100000 + st["step"]
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            loader = DataLoader(torch.utils.data.Subset(Crops(args.data, rows["train"], True), remaining.tolist()),
                                batch_size=args.batch, shuffle=False, num_workers=args.workers,
                                worker_init_fn=lambda w: np.random.seed(seed + w))
            model.train(); t0 = time.time()
            for x, y, s, c, has in loader:
                o = model(x)
                loss_c = ce(o[:, :7], y)
                has = has.float()
                loss_a = (((o[:, 7] - s.float()) ** 2 + (o[:, 8] - c.float()) ** 2) * has).sum() / has.sum().clamp(min=1)
                loss = loss_c + args.angle_weight * loss_a
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                st["loss_sum"] += loss_c.item() * len(y); st["ang_sum"] += loss_a.item() * len(y); st["seen"] += len(y)
                st["correct"] += (o[:, :7].argmax(1) == y).sum().item(); st["step"] += 1
                if st["step"] % 50 == 0:
                    print(f"  epoch {ep} step {st['step']}/{steps_per_epoch} cls {st['loss_sum'] / st['seen']:.3f} "
                          f"angle {st['ang_sum'] / st['seen']:.3f}", flush=True)
                if out_of_time():
                    break
            st["ep_seconds"] += time.time() - t0
            if st["step"] >= steps_per_epoch:
                st["phase"] = "eval"
            save()
        elif st["phase"] == "eval":
            cm = evaluate(model, te)
            acc, _ = report(cm)
            ep = st["epoch"]
            st["history"].append(dict(epoch=ep, train_loss=round(st["loss_sum"] / st["seen"], 4),
                                      train_angle_loss=round(st["ang_sum"] / st["seen"], 4),
                                      train_acc_rotated=round(st["correct"] / st["seen"], 4),
                                      test_acc_upright=round(acc, 4), train_seconds=round(st["ep_seconds"])))
            print(f"epoch {ep}: train acc (rotated) {st['correct'] / st['seen']:.3f} | test acc upright {acc:.3f}", flush=True)
            st.update(epoch=ep + 1, step=0, loss_sum=0.0, ang_sum=0.0, correct=0, seen=0, ep_seconds=0.0,
                      phase="train" if ep < args.epochs else "final")
            save()
        elif st["phase"] == "final":
            cm = evaluate(model, te)
            acc, per = report(cm)
            print("Rotated test views, new model:")
            rot = rotation_eval(model, args.data, rows["test"])
            sub = cm[:6, :6]
            st["metrics"] = dict(
                dataset="HaGRID sample 30k 384p (cj-mills/hagrid-sample-30k-384p), CC-BY-SA 4.0",
                split="80/20 by person (user_id)", train_crops=n_tr, test_crops=n_te,
                test_accuracy=round(acc, 4),
                finger_count_accuracy_on_counting_hands=round(float(np.trace(sub) / max(sub.sum(), 1)), 4),
                none_detected_as_chord_rate=round(float(1 - cm[NONE, NONE] / max(cm[NONE].sum(), 1)), 4),
                rotation=rot,
                mean_accuracy_over_rotations=round(float(np.mean([v["accuracy"] for v in rot.values()])), 4),
                mean_tilt_error_deg=round(float(np.mean([v["tilt_mae_deg"] for v in rot.values()])), 1),
                per_class=per, confusion_matrix=cm.tolist(), classes=CLASS_NAMES, history=st["history"])
            save_confusion(cm, out / "confusion_matrix.png", "Test set (20% of people, unseen in training)")
            torch.save(model.state_dict(), out / "handchord.pt")
            st["phase"] = "compare" if args.compare else "write"
            save()
        elif st["phase"] == "compare":
            print("Rotated test views, previous model:")
            old = build_model(None, None, n_out=7)
            old.load_state_dict(torch.load(args.compare, map_location="cpu"))
            st["metrics"]["previous_model_rotation"] = rotation_eval(old, args.data, rows["test"])
            st["metrics"]["previous_model_mean_accuracy_over_rotations"] = round(float(np.mean(
                [v["accuracy"] for v in st["metrics"]["previous_model_rotation"].values()])), 4)
            st["phase"] = "write"
            save()
        elif st["phase"] == "write":
            m = st["metrics"]
            json.dump(m, open(out / "metrics.json", "w"), indent=1)
            save_rotation_plot(m["rotation"], m.get("previous_model_rotation"), out / "rotation_accuracy.png")
            print(json.dumps({k: v for k, v in m.items() if k not in ("per_class", "history", "confusion_matrix", "rotation", "previous_model_rotation")}, indent=1))
            st["phase"] = "done"
            save()
    print("DONE" if st["phase"] == "done" else f"PAUSED at epoch {st['epoch']} step {st['step']} ({st['phase']}) - run again to continue")


if __name__ == "__main__":
    main()
