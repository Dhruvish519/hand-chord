// fingerModel.js — runs the trained finger-count CNN with OpenCV's DNN module (cv.dnn).
//
// Model: MobileNetV3-Small fine-tuned on hand crops from the HaGRID dataset
//        (see python/train/). Input: 112x112 RGB, pixel values 0-255
//        (the ImageNet normalisation is built into the ONNX file).
// Output: 9 numbers: 7 scores -> softmax -> probabilities for [0, 1, 2, 3, 4, 5 fingers, none],
//         then sin and cos of the hand's tilt -> tilt angle in degrees (0 = upright, + = clockwise).

export const CLASSES = ['0', '1', '2', '3', '4', '5', 'none'];
export const NONE = 6;
const SIZE = 112;

export async function loadFingerModel(cv, url = 'model/handchord.onnx') {
  // Retry a few times: a 6 MB download can be cut off by a network blip.
  let bytes = null, lastErr = null;
  for (let attempt = 0; attempt < 4 && !bytes; attempt++) {
    try {
      if (attempt) await new Promise((r) => setTimeout(r, 1500 * attempt));
      const res = await fetch(url);
      if (!res.ok) throw new Error(`model download failed (${res.status})`);
      bytes = new Uint8Array(await res.arrayBuffer());
    } catch (err) { lastErr = err; }
  }
  if (!bytes) throw lastErr;
  // OpenCV.js reads files from its in-memory file system, so write the model there first.
  const path = 'handchord.onnx';
  try { cv.FS_unlink(path); } catch (e) { /* not there yet */ }
  cv.FS_createDataFile('/', path, bytes, true, false, false);
  const net = cv.readNetFromONNX(path);

  const rgb = new cv.Mat();
  const blobSize = new cv.Size(SIZE, SIZE);
  const zero = new cv.Scalar(0, 0, 0, 0);

  /** crop: CV_8UC4 (RGBA) Mat of any size, ideally square. Returns { probs (7), tilt (degrees or null) }. */
  function classify(crop) {
    cv.cvtColor(crop, rgb, cv.COLOR_RGBA2RGB);
    // blobFromImage resizes to 112x112 and reorders HWC -> NCHW float32.
    const blob = cv.blobFromImage(rgb, 1.0, blobSize, zero, false, false);
    net.setInput(blob);
    const out = net.forward();
    const raw = Array.from(out.data32F);
    blob.delete(); out.delete();
    const logits = raw.slice(0, 7);
    const max = Math.max(...logits);
    const exps = logits.map((v) => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    const tilt = raw.length >= 9 ? (Math.atan2(raw[7], raw[8]) * 180) / Math.PI : null;
    return { probs: exps.map((e) => e / sum), tilt, sin: raw[7], cos: raw[8] };
  }

  return { classify, size: SIZE };
}
