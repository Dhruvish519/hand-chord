# HandChord — an OpenCV gesture synthesizer

Play chords with your hand in the browser. OpenCV.js finds the hand, a **MobileNetV3 CNN trained on 44,098 real hand images (HaGRID dataset)** counts the fingers, running inside OpenCV's DNN module, and the result drives a synthesizer built with Tone.js.

**Live:** https://hand-chord.vercel.app

| Gesture | Sound |
|---|---|
| 1 / 2 / 3 / 4 fingers | C, G, Am, F — the I–V–vi–IV progression |
| 5 fingers | Dm (ii) |
| Fist (0) | silence |
| Hand height | volume |
| Hand left ↔ right | filter brightness |
| Hand tilt | vibrato |

Inspired by [Gesture Synth](https://gesture-synth-weld.vercel.app) by Eric Wei, which uses MediaPipe. This project uses its own OpenCV pipeline plus a finger-count model trained and evaluated here (80/20 split by person).

---

## The trained model

| | |
|---|---|
| **Dataset** | [HaGRID](https://github.com/hukenovs/hagrid) (HAnd Gesture Recognition Image Dataset), 31,833-image sample at 384p: [cj-mills/hagrid-sample-30k-384p](https://huggingface.co/datasets/cj-mills/hagrid-sample-30k-384p), CC-BY-SA 4.0 |
| **Crops** | 44,098 hand crops (112×112) from 7,973 people |
| **Classes** | 0 = fist, 1 = one, 2 = peace / two_up (+inverted), 3 = three / three2, 4 = four, 5 = palm / stop (+inverted), **none** = other gestures, relaxed hands and random background patches (faces, bodies, objects) |
| **Split** | 80/20 **by person** (`user_id`): 36,165 train crops, 7,933 test crops, no person in both |
| **Model** | MobileNetV3-Small, ImageNet-pretrained ([d-li14/mobilenetv3.pytorch](https://github.com/d-li14/mobilenetv3.pytorch)), last layer → 7 classes |
| **Training** | 8 epochs, AdamW + one-cycle LR, class-weighted loss, augmentation (flip, ±20° rotation, zoom, brightness, white balance, blur) |
| **Runs in** | OpenCV.js `cv.dnn` (ONNX, 6.2 MB). OpenCV DNN and PyTorch agree on 300/300 test crops |

**Results on the held-out 20% of people:**

| Metric | Value |
|---|---|
| Test accuracy (7 classes) | **98.4%** |
| Finger count correct, when the hand is a counting pose | 98.3% |
| "none" images mistaken for a chord | 1.4% |

| Class | Precision | Recall | F1 | Test crops |
|---|---|---|---|---|
| 0 | 0.980 | 1.000 | 0.990 | 334 |
| 1 | 0.930 | 0.964 | 0.947 | 387 |
| 2 | 0.970 | 0.985 | 0.977 | 1,427 |
| 3 | 0.970 | 0.966 | 0.968 | 726 |
| 4 | 0.965 | 0.970 | 0.967 | 365 |
| 5 | 0.989 | 0.993 | 0.991 | 1,092 |
| none | 0.999 | 0.986 | 0.992 | 3,602 |

Test accuracy by epoch: 91.1 → 95.7 → 94.8 → 96.7 → 97.5 → 98.1 → 98.3 → 98.4%. The confusion matrix is in `python/train/results/confusion_matrix.png` (and on the website).

### How the website uses it
1. OpenCV skin segmentation + largest contour gives a candidate box (and the hand position/tilt from moments).
2. Two square crops go to the CNN: one around the contour, one covering the hand zone (backup when skin detection fails, e.g. skin-coloured walls).
3. The more confident crop wins. Probabilities are averaged over frames and must pass 55% confidence and a 7-frame vote before the chord changes. "none" = silence, so faces and objects don't play.

The original classical method (convexity defects) is still selectable in the panel, which makes a good live comparison.

### Re-train it yourself
```bash
cd python/train
git clone --depth 1 https://github.com/d-li14/mobilenetv3.pytorch   # model code + ImageNet weights
pip install torch onnx opencv-python-headless numpy matplotlib
# download hagrid-sample-30k-384p.zip (1 GB) from the Hugging Face link above, then:
python prepare_data.py --zip hagrid-sample-30k-384p.zip --out data      # ~1 min
python train.py --data data --out model --epochs 8                     # ~25 min on a laptop CPU
python export_web.py --weights model/handchord.pt --data data --out ../../public/model/handchord.onnx
```

---

## The computer-vision pipeline

Same steps in `public/js/handTracker.js` (browser) and `python/hand_pipeline.py` (evaluation):

1. **Colour space: RGB → YCrCb.** Y is brightness, Cr/Cb are colour. Skin tones cluster tightly in Cr/Cb regardless of brightness, so we ignore Y.
2. **Skin segmentation:** `inRange` keeps pixels with 135 ≤ Cr ≤ 180 and 85 ≤ Cb ≤ 135 (based on Chai & Ngan, 1999). An *auto-calibrate* button learns the range from your palm as mean ± 2.5 standard deviations.
3. **Noise removal:** median blur, then morphological **opening** (erode→dilate, removes specks) and **closing** (dilate→erode, fills holes) with an elliptical 5×5 kernel.
4. **Hand = largest contour** (`findContours`, external only), rejected if smaller than 2% of the box.
5. **Image moments:** centroid `cx = m10/m00`, `cy = m01/m00`; orientation `θ = ½·atan2(2μ11, μ20 − μ02)`.
6. **Finger counting with convexity defects:** compute the convex hull, then the defects (points where the contour dips inside the hull). A defect counts as a gap between two fingers if it is
   - deeper than 18% of the hand's height,
   - narrower than 90° (cosine rule on start–far–end points),
   - above the bottom 15% of the hand (rules out the wrist).

   `fingers = gaps + 1`. With no gaps, the shape is a fist (0) or one finger (1): one finger is tall (height/width > 1.6) and not solid (area/hull area < 0.85).
7. **Finger count from the CNN** (default), or from convexity defects in classical mode.
8. **Temporal smoothing (website):** the finger count must win a majority vote over the last 7 frames before the chord changes; position and tilt use an exponential moving average.

The website has tabs to show each stage live: camera, raw skin mask, cleaned mask, contour + hull.

---

## Project structure

```
public/                 ← the website (deployed to Vercel as-is, no build step)
  index.html
  css/style.css
  js/handTracker.js     ← OpenCV pipeline (skin, contour, moments, convexity defects)
  js/fingerModel.js     ← loads the ONNX CNN into cv.dnn and classifies hand crops
  model/handchord.onnx  ← trained model (6.2 MB)
  js/synth.js           ← Tone.js chords, presets, effects
  js/app.js             ← camera loop, smoothing, drawing
  lib/opencv.js         ← OpenCV.js 4.10 (bundled, so the demo works offline)
  lib/Tone.js
python/
  train/prepare_data.py ← HaGRID → hand crops, 80/20 split by person
  train/train.py        ← fine-tunes MobileNetV3-Small, reports test metrics
  train/export_web.py   ← ONNX export for OpenCV.js + agreement check
  train/results/        ← metrics.json, confusion_matrix.png
  hand_pipeline.py      ← same pipeline in Python OpenCV
  evaluate.py           ← accuracy on photo datasets, annotated video output, webcam demo
  make_synthetic_dataset.py
test/synthetic.test.mjs ← automated test of the JS pipeline
vercel.json
```

---

## Run locally

Website (any static server works; the camera needs `localhost` or HTTPS):

```bash
cd public
python -m http.server 8000
# open http://localhost:8000
```

Python evaluation:

```bash
cd python
pip install -r requirements.txt
python evaluate.py webcam                                   # live window, press q to quit
python evaluate.py images --data path/to/dataset            # colour photos
python evaluate.py images --data path/to/fingers/test --mode otsu   # Kaggle Fingers dataset
python evaluate.py video --input my_clip.mp4 --out results/annotated.mp4
```

Each `images` run prints overall and per-class accuracy, a confusion matrix and the time per image, and saves `predictions.csv`, `confusion_matrix.png` and example images (correct and wrong) to `results/`.

### Datasets

- **Your own photos (recommended):** photograph 30–50 images per class (0–5 fingers) with your phone, a few people, different backgrounds and lighting. Put them in folders `0/ … 5/`.
- **Kaggle "Fingers" dataset** (koryakinp/fingers): 21,600 grayscale 128×128 images, labels in the file name (`…_3L.png`). Use `--mode otsu`, since the images have no colour.
- **Video:** record a 30–60 s clip counting 0→5 and run the `video` command. The CSV gives the per-frame count for your report.

---

## Deploy to Vercel

1. Create a new GitHub repository and upload everything in this folder (drag-and-drop on github.com works; `opencv.js` is 10 MB, which is fine).
2. Go to vercel.com → **Add New → Project** → import the repository.
3. Leave the settings alone (`vercel.json` already says: no build, serve `public/`). Click **Deploy**.
4. Open the `…vercel.app` link and allow the camera.

---

## Tests

```bash
npm install
npm test          # synthetic hands with 0–5 fingers at three tilts, through the real JS pipeline
```

## Known limitations

- The CNN knows the HaGRID poses. Unusual ways of showing a number (e.g. 3 = thumb + pinky + ring) may read as "none" or another count.
- "1" is the weakest class (F1 0.947): it's sometimes confused with "none" gestures such as *like* (thumb up) or *mute* (finger on lips).
- Accuracy is measured on still photos; live video adds motion blur. The frame averaging and vote help with that.
- In classical mode: skin-coloured backgrounds break segmentation, and fingers held together undercount.

## Credits

- Concept: Gesture Synth by Eric Wei.
- Dataset: HaGRID, A. Kapitanov, K. Kvanchiani, A. Nagaev, R. Kraynov, A. Makhliarchuk, "HaGRID – HAnd Gesture Recognition Image Dataset," WACV 2024 (CC-BY-SA 4.0). Sample by cj-mills.
- Model: MobileNetV3, A. Howard et al., ICCV 2019. PyTorch implementation and weights: d-li14/mobilenetv3.pytorch (MIT).
- OpenCV.js (Apache 2.0), Tone.js (MIT).
- Skin range: D. Chai and K. N. Ngan, "Face segmentation using skin-color map in videophone applications," IEEE TCSVT, 1999.
