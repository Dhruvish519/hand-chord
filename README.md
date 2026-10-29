# HandChord — an OpenCV gesture synthesizer

Play chords with your hand in the browser. A webcam frame goes through a **classical computer-vision pipeline in OpenCV.js** (no deep learning), and the result drives a synthesizer built with Tone.js.

| Gesture | Sound |
|---|---|
| 1 / 2 / 3 / 4 fingers | C, G, Am, F — the I–V–vi–IV progression |
| 5 fingers | Dm (ii) |
| Fist (0) | silence |
| Hand height | volume |
| Hand left ↔ right | filter brightness |
| Hand tilt | vibrato |

Inspired by [Gesture Synth](https://gesture-synth-weld.vercel.app) by Eric Wei, which uses MediaPipe. This project replaces the pretrained model with a hand-built OpenCV pipeline and adds a Python evaluation on photo and video datasets.

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
7. **Temporal smoothing (website):** the finger count must win a majority vote over the last 7 frames before the chord changes; position and tilt use an exponential moving average.

The website has tabs to show each stage live: camera, raw skin mask, cleaned mask, contour + hull.

---

## Project structure

```
public/                 ← the website (deployed to Vercel as-is, no build step)
  index.html
  css/style.css
  js/handTracker.js     ← OpenCV pipeline
  js/synth.js           ← Tone.js chords, presets, effects
  js/app.js             ← camera loop, smoothing, drawing
  lib/opencv.js         ← OpenCV.js 4.10 (bundled, so the demo works offline)
  lib/Tone.js
python/
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

- Skin segmentation fails if the background or clothing is skin-coloured, or under strongly coloured light. The hand zone box keeps the face out of the frame.
- 0 vs 1 finger is the weakest case, because it can't rely on convexity defects.
- Fingers held tightly together merge into one blob and are undercounted.

## Credits

- Concept: Gesture Synth by Eric Wei.
- OpenCV.js (Apache 2.0), Tone.js (MIT).
- Skin range: D. Chai and K. N. Ngan, "Face segmentation using skin-color map in videophone applications," IEEE TCSVT, 1999.
