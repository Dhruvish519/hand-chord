"""
export_web.py — export the trained model for the website (OpenCV.js DNN).

The web version takes raw RGB pixels (0-255); the ImageNet normalisation
(x/255 - mean) / std is built into the ONNX graph so the browser code stays simple.
Also checks that OpenCV DNN gives the same answers as PyTorch on real test crops.

    python export_web.py --weights model/handchord.pt --data data --out ../../public/model/handchord.onnx
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "mobilenetv3.pytorch"))
from mobilenetv3 import mobilenetv3_small  # noqa: E402
from train import make_view, N_OUT  # noqa: E402

SIZE = 112


class WebModel(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255)

    def forward(self, x):  # x: 1x3x112x112 RGB in 0..255
        return self.net((x - self.mean) / self.std)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="model/handchord.pt")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="model/handchord_web.onnx")
    args = ap.parse_args()

    net = mobilenetv3_small()
    net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, N_OUT)  # 7 classes + sin, cos of tilt
    net.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model = WebModel(net).eval()

    kw = dict(input_names=["input"], output_names=["logits"], opset_version=11, do_constant_folding=True)
    try:
        torch.onnx.export(model, torch.zeros(1, 3, SIZE, SIZE), args.out, dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(model, torch.zeros(1, 3, SIZE, SIZE), args.out, **kw)
    print(f"Saved {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB)")

    # Agreement check on 300 real test crops: OpenCV DNN (what the website uses) vs PyTorch.
    cvnet = cv2.dnn.readNetFromONNX(args.out)
    rows = [r for r in csv.DictReader(open(Path(args.data) / "labels.csv")) if r["split"] == "test"][:300]
    agree, correct = 0, 0
    for r in rows:
        rgb = make_view(cv2.imread(str(Path(args.data) / "crops" / r["file"]))[:, :, ::-1].copy())
        blob = cv2.dnn.blobFromImage(rgb, 1.0, (SIZE, SIZE), (0, 0, 0), swapRB=False, crop=False)
        cvnet.setInput(blob)
        p_cv = int(cvnet.forward()[0, :7].argmax())
        with torch.no_grad():
            p_pt = int(model(torch.from_numpy(blob))[0, :7].argmax())
        agree += p_cv == p_pt
        correct += p_cv == int(r["label"])
    print(f"OpenCV DNN agrees with PyTorch on {agree}/{len(rows)} test crops; accuracy {correct / len(rows) * 100:.1f}%")


if __name__ == "__main__":
    main()
