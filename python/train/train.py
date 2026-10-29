"""
train.py — fine-tune MobileNetV3-Small to classify hand crops into 0-5 fingers or "none".

Uses the crops from prepare_data.py. Trains on the 80% "train" people and reports results
on the 20% "test" people (who were never seen in training).

Model:  MobileNetV3-Small (Howard et al., 2019), ImageNet-pretrained weights from
        github.com/d-li14/mobilenetv3.pytorch (MIT), last layer replaced with 7 outputs.
Input:  112x112 RGB, scaled to 0..1 then normalised with ImageNet mean/std.
Output: model/handchord.onnx (runs in OpenCV DNN, including OpenCV.js) + metrics.

    python train.py --data data --epochs 10
"""
import argparse
import csv
import json
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
SIZE = 112
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class Crops(Dataset):
    def __init__(self, root, rows, train):
        self.root, self.rows, self.train = Path(root), rows, train

    def __len__(self):
        return len(self.rows)

    def augment(self, img):
        # Webcam-like variation: mirror (left/right hand), rotation, zoom, brightness/colour.
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
        angle, scale = random.uniform(-20, 20), random.uniform(0.9, 1.15)
        M = cv2.getRotationMatrix2D((SIZE / 2, SIZE / 2), angle, scale)
        M[:, 2] += np.random.uniform(-6, 6, 2)
        img = cv2.warpAffine(img, M, (SIZE, SIZE), borderMode=cv2.BORDER_REPLICATE)
        img = img.astype(np.float32)
        img = img * random.uniform(0.7, 1.3) + random.uniform(-25, 25)          # contrast, brightness
        img = img * np.random.uniform(0.9, 1.1, 3).astype(np.float32)            # white balance
        if random.random() < 0.2:
            img = cv2.GaussianBlur(img, (3, 3), 0)
        return np.clip(img, 0, 255)

    def __getitem__(self, i):
        f, label = self.rows[i]
        img = cv2.imread(str(self.root / "crops" / f))[:, :, ::-1]  # BGR -> RGB
        img = self.augment(img) if self.train else img.astype(np.float32)
        x = (img / 255.0 - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), label


def build_model(pretrained):
    model = mobilenetv3_small()
    if pretrained and Path(pretrained).exists():
        model.load_state_dict(torch.load(pretrained, map_location="cpu"))
        print("Loaded ImageNet weights")
    in_f = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_f, len(CLASS_NAMES))
    return model


def evaluate(model, loader):
    model.eval()
    cm = np.zeros((7, 7), int)
    with torch.no_grad():
        for x, y in loader:
            p = model(x).argmax(1)
            for t, q in zip(y.numpy(), p.numpy()):
                cm[t, q] += 1
    return cm


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


def save_confusion(cm, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6, 5.2))
    ax.imshow(norm, cmap="Oranges", vmin=0, vmax=1)
    ax.set_xticks(range(7), CLASS_NAMES); ax.set_yticks(range(7), CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Test set (20% of people, unseen in training)")
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j] * 100:.0f}%", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="model")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--time-budget", type=float, default=0,
                    help="stop after N seconds and continue on the next run (0 = run to the end)")
    ap.add_argument("--pretrained", default=str(HERE / "mobilenetv3.pytorch/pretrained/mobilenetv3-small-55df8e1f.pth"))
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
    model = build_model(None)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = n_tr // args.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.15)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    ckpt_f = out / "checkpoint.pt"
    if ckpt_f.exists():
        ck = torch.load(ckpt_f, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        st = ck["state"]
    else:
        print(f"Train crops: {n_tr}   Test crops: {n_te}   ({n_tr / (n_tr + n_te) * 100:.0f}/{n_te / (n_tr + n_te) * 100:.0f} split by person)")
        print("Train class counts:", dict(zip(CLASS_NAMES, counts.tolist())))
        model = build_model(args.pretrained)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.15)
        st = dict(epoch=1, step=0, phase="train", loss_sum=0.0, correct=0, seen=0, ep_seconds=0.0, history=[])

    def save():
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), state=st), ckpt_f)

    te = DataLoader(Crops(args.data, rows["test"], False), batch_size=128, num_workers=args.workers)
    while st["phase"] != "done" and not out_of_time():
        if st["phase"] == "train":
            ep = st["epoch"]
            # Same shuffled order for an epoch whatever the run boundaries, so training can resume mid-epoch.
            order = np.random.RandomState(ep).permutation(n_tr)[: steps_per_epoch * args.batch]
            remaining = order[st["step"] * args.batch:]
            random.seed(ep * 100000 + st["step"]); np.random.seed(ep * 100000 + st["step"]); torch.manual_seed(ep * 100000 + st["step"])
            loader = DataLoader(torch.utils.data.Subset(Crops(args.data, rows["train"], True), remaining.tolist()),
                                batch_size=args.batch, shuffle=False, num_workers=args.workers)
            model.train(); t0 = time.time()
            for x, y in loader:
                logits = model(x)
                loss = loss_fn(logits, y)
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                st["loss_sum"] += loss.item() * len(y); st["seen"] += len(y)
                st["correct"] += (logits.argmax(1) == y).sum().item(); st["step"] += 1
                if st["step"] % 50 == 0:
                    print(f"  epoch {ep} step {st['step']}/{steps_per_epoch} loss {st['loss_sum'] / st['seen']:.3f}", flush=True)
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
                                      train_acc=round(st["correct"] / st["seen"], 4), test_acc=round(acc, 4),
                                      train_seconds=round(st["ep_seconds"])))
            print(f"epoch {ep}: train loss {st['loss_sum'] / st['seen']:.3f} acc {st['correct'] / st['seen']:.3f} | test acc {acc:.3f}", flush=True)
            st.update(epoch=ep + 1, step=0, loss_sum=0.0, correct=0, seen=0, ep_seconds=0.0,
                      phase="train" if ep < args.epochs else "final")
            save()
        elif st["phase"] == "final":
            finish(model, te, st, n_tr, n_te, out)
            st["phase"] = "done"; save()
    print("DONE" if st["phase"] == "done" else f"PAUSED at epoch {st['epoch']} step {st['step']} ({st['phase']}) - run again to continue")


def finish(model, te, st, n_tr, n_te, out):
    cm = evaluate(model, te)
    acc, per = report(cm)
    count_mask = np.arange(7) < 6
    sub = cm[np.ix_(count_mask, count_mask)]
    metrics = dict(
        dataset="HaGRID sample 30k 384p (cj-mills/hagrid-sample-30k-384p), CC-BY-SA 4.0",
        split="80/20 by person (user_id)", train_crops=n_tr, test_crops=n_te,
        test_accuracy=round(acc, 4),
        finger_count_accuracy_on_counting_hands=round(float(np.trace(sub) / max(sub.sum(), 1)), 4),
        none_detected_as_chord_rate=round(float(1 - cm[6, 6] / max(cm[6].sum(), 1)), 4),
        per_class=per, confusion_matrix=cm.tolist(), classes=CLASS_NAMES, history=st["history"],
    )
    json.dump(metrics, open(out / "metrics.json", "w"), indent=1)
    save_confusion(cm, out / "confusion_matrix.png")
    torch.save(model.state_dict(), out / "handchord.pt")
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("per_class", "history", "confusion_matrix")}, indent=1))

    # Export for OpenCV DNN / OpenCV.js
    model.eval()
    dummy = torch.zeros(1, 3, SIZE, SIZE)
    kw = dict(input_names=["input"], output_names=["logits"], opset_version=11, do_constant_folding=True)
    try:
        torch.onnx.export(model, dummy, str(out / "handchord.onnx"), dynamo=False, **kw)
    except TypeError:  # older PyTorch has no dynamo argument
        torch.onnx.export(model, dummy, str(out / "handchord.onnx"), **kw)
    net = cv2.dnn.readNetFromONNX(str(out / "handchord.onnx"))
    x = np.random.rand(1, 3, SIZE, SIZE).astype(np.float32)
    net.setInput(x)
    with torch.no_grad():
        diff = np.abs(net.forward() - model(torch.from_numpy(x)).numpy()).max()
    print(f"ONNX exported; OpenCV DNN vs PyTorch max difference: {diff:.2e}")


if __name__ == "__main__":
    main()
