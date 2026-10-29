// Sanity test for handTracker.js using synthetic hand drawings.
// Run: node test/synthetic.test.mjs
import cvModule from '@techstark/opencv-js';
import { createTracker } from '../public/js/handTracker.js';

// Note: the cv object is "thenable", so never resolve a Promise with it directly
// (that loops forever). Wrap it in an object instead.
const { cv } = await new Promise((resolve) => {
  if (cvModule.Mat) resolve({ cv: cvModule });
  else cvModule.onRuntimeInitialized = () => resolve({ cv: cvModule });
});

const SKIN = new cv.Scalar(224, 172, 140, 255);  // RGBA
const BG = new cv.Scalar(40, 60, 120, 255);

function drawHand(n, tilt = 0) {
  const img = new cv.Mat(240, 320, cv.CV_8UC4, BG);
  const c = new cv.Point(160, 165);
  cv.circle(img, c, 48, SKIN, -1);
  cv.rectangle(img, new cv.Point(125, 165), new cv.Point(195, 240), SKIN, -1); // wrist
  const total = Math.min(128, 40 * (n - 1));  // realistic spread: V-sign ~40 deg, open hand ~128 deg
  const spread = n === 1 ? [0] : Array.from({ length: n }, (_, i) => -total / 2 + (total * i) / (n - 1));
  for (const a0 of spread) {
    const a = ((a0 + tilt) * Math.PI) / 180;
    const tip = new cv.Point(c.x + Math.sin(a) * 120, c.y - Math.cos(a) * 120);
    cv.line(img, c, tip, SKIN, 17);
  }
  return img;
}

const tracker = createTracker(cv);
let pass = 0, total = 0;
for (const tilt of [0, -15, 15]) {
  for (let n = 0; n <= 5; n++) {
    const img = drawHand(n, tilt);
    const r = tracker.process(img);
    total++;
    const ok = r.found && r.fingers === n;
    if (ok) pass++;
    console.log(`tilt ${String(tilt).padStart(3)}  expected ${n}  got ${r.fingers}  ` +
      `centroid (${r.cx.toFixed(0)},${r.cy.toFixed(0)}) angle ${r.angle.toFixed(1)}  ${ok ? 'OK' : 'FAIL'}`);
    img.delete();
  }
}
console.log(`\n${pass}/${total} correct`);
process.exit(pass === total ? 0 : 1);
