# HandChord — an OpenCV gesture synthesizer

Play chords with your hand in the browser. OpenCV.js finds the hand, a **MobileNetV3 CNN trained on 44,098 real hand images (HaGRID dataset)** counts the fingers, running inside OpenCV's DNN module, and the result drives a synthesizer built with Tone.js.

**Live:** https://hand-chord.vercel.app

| Gesture | Sound |
|---|---|
| Fist (0) | Em (iii) |
| 1 / 2 / 3 / 4 fingers | C, G, Am, F — the I–V–vi–IV progression |
| 5 fingers | Dm (ii) |
| Hand out of the box, or not a counting gesture | silence |
| *(all of the above can be changed in the panel)* | |
| Hand height | volume |
| Hand left ↔ right | filter brightness |
| Hand tilt | vibrato |

Inspired by [Gesture Synth](https://gesture-synth-weld.vercel.app) by Eric Wei, which uses MediaPipe. This project uses its own OpenCV pipeline plus a finger-count model trained and evaluated here (80/20 split by person).

---

## The trained model

| | |
|---|---|
| **Dataset** | [HaGRID](https://github.com/hukenovs/hagrid) (HAnd Gesture Recognition Image Dataset), 31,833-image sample at 384p: [cj-mills/hagrid-sample-30k-384p](https://huggingface.co/datasets/cj-mills/hagrid-sample-30k-384p), CC-BY-SA 4.0 |
| **Crops** | 44,053 hand crops from 7,973 people (stored at 160×160 with 2× context so they can be rotated) |
| **Classes** | 0 = fist, 1 = one, 2 = peace / two_up (+inverted), 3 = three / three2, 4 = four, 5 = palm / stop (+inverted), **none** = other gestures, relaxed hands and random background patches (faces, bodies, objects) |
| **Split** | 80/20 **by person** (`user_id`): 36,128 train crops, 7,925 test crops, no person in both |
| **Model** | MobileNetV3-Small, ImageNet-pretrained ([d-li14/mobilenetv3.pytorch](https://github.com/d-li14/mobilenetv3.pytorch)), last layer → **9 outputs**: 7 class scores + sin, cos of the hand's tilt |
| **Rotation training** | Each training image is rotated by a random angle (25% within ±15°, 50% within ±90°, 25% anywhere in ±180°), perspective-warped up to 8%, zoomed 1.2–1.75×, shifted, flipped, and colour/brightness/blur-augmented. The known rotation is the tilt label |
| **Loss** | class-weighted cross-entropy (label smoothing 0.05) + squared error on (sin, cos) for counting hands |
| **Training** | started from the upright-only model, 6 more epochs, AdamW + one-cycle LR |
| **Runs in** | OpenCV.js `cv.dnn` (ONNX, 6.2 MB). OpenCV DNN and PyTorch agree on 300/300 test crops |

**Results on the held-out 20% of people:**

| Metric | Value |
|---|---|
| Test accuracy, upright hands (7 classes) | **98.0%** |
| Finger count correct when the hand is a counting pose | 98.4% |
| "none" images mistaken for a sound | 2.3% |
| Average accuracy over 8 rotations (−135° … 180°) | **95.6%** (previous model: 60.5%) |
| Average tilt-angle error | **5.7°** |

| Rotation | Previous model | New model | Tilt error |
|---|---|---|---|
| -135° | 44.1% | 93.7% | 5.5° |
| -90° | 43.3% | 96.3% | 5.6° |
| -45° | 82.4% | 97.1% | 5.4° |
| 0° | 99.0% | 98.1% | 4.0° |
| 45° | 84.1% | 96.9% | 6.1° |
| 90° | 44.1% | 95.8% | 6.1° |
| 135° | 43.8% | 93.9% | 6.9° |
| 180° | 43.2% | 93.2% | 5.7° |

| Class (upright test) | Precision | Recall | F1 | Test crops |
|---|---|---|---|---|
| 0 | 0.971 | 0.997 | 0.984 | 334 |
| 1 | 0.913 | 0.954 | 0.933 | 387 |
| 2 | 0.973 | 0.977 | 0.975 | 1,427 |
| 3 | 0.963 | 0.975 | 0.969 | 726 |
| 4 | 0.955 | 0.992 | 0.973 | 365 |
| 5 | 0.980 | 0.998 | 0.989 | 1,092 |
| none | 0.998 | 0.977 | 0.988 | 3,594 |

Charts: `python/train/results/rotation_accuracy.png` and `confusion_matrix.png` (also on the website).

### How the website uses it
1. OpenCV skin segmentation + largest contour gives a candidate box (and the hand position/tilt from moments).
2. Two square crops go to the CNN: one around the contour, one covering the hand zone (backup when skin detection fails, e.g. skin-coloured walls).
3. The more confident crop wins. Probabilities are averaged over frames and must pass 55% confidence and a 7-frame vote before the sound changes. "none" = silence, so faces and objects don't play.
4. **Tilt** comes from the CNN's (sin, cos) output, smoothed over frames. Vibrato starts at 8° and is full at 35° either way. (In classical mode, tilt comes from image moments.)

### Customising the sounds
Open **Customise sounds for each finger** in the panel. Each count, including the fist (0), can play any chord of the current key, any single note from C3 to B5, or silence. The choice is saved in the browser. Default: fist = iii, 1–4 fingers = I–V–vi–IV, 5 = ii.

The original classical method (convexity defects) is still selectable in the panel, which makes a good live comparison.

### Re-train it yourself
```bash
cd python/train
git clone --depth 1 https://github.com/d-li14/mobilenetv3.pytorch   # model code + ImageNet weights
pip install torch onnx opencv-python-headless numpy matplotlib
# download hagrid-sample-30k-384p.zip (1 GB) from the Hugging Face link above, then:
python prepare_data.py --zip hagrid-sample-30k-384p.zip --out data      # ~1 min
python train.py --data data --out model --epochs 6 --init <upright model .pt, optional> --compare <old .pt, optional>  # ~20 min on a laptop CPU
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

- Rotated-hand accuracy is measured by rotating test photos in 2D. Turning the hand towards or away from the camera (3D) is only covered by the mild perspective augmentation.
- Tilt is learned relative to how people naturally hold each gesture in HaGRID (roughly upright), so it's a relative angle, not an exact measurement.
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
