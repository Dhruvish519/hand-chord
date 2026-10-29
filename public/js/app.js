// app.js — wires camera -> OpenCV tracker -> synth, and draws the UI.
import { createTracker, DEFAULT_PARAMS } from './handTracker.js';
import { ChordSynth, PRESETS, FINGER_TO_DEGREE, NOTE_NAMES } from './synth.js';

const $ = (id) => document.getElementById(id);
const video = $('video'), view = $('view'), proc = $('proc'), maskCanvas = $('maskCanvas'), scope = $('scope');
const vctx = view.getContext('2d');
const pctx = proc.getContext('2d', { willReadFrequently: true });
const sctx = scope.getContext('2d');

const PROC_W = proc.width, PROC_H = proc.height;   // 320 x 240: processing resolution
const SCALE = view.width / PROC_W;                  // display is 2x the processing size
const STABLE_FRAMES = 7;                            // finger count must win a vote over the last 7 frames
const SMOOTH = 0.35;                                // exponential smoothing for position/tilt

let cv, tracker, synth;
let params = { ...DEFAULT_PARAMS };
let stage = 'camera';
let side = 'right';
let running = false;
let history = [];
let stableFingers = -1;
let smooth = null;
let last = null;
let fpsFrames = 0, fpsT0 = performance.now();
let calibrating = false;

// ---------- OpenCV loading ----------
// The opencv.js build exposes a "thenable" object; never resolve a Promise with it directly.
function waitForOpenCV() {
  return new Promise((resolve) => {
    const check = () => {
      if (window.cv && window.cv.Mat) { resolve({ cv: window.cv }); return true; }
      return false;
    };
    if (check()) return;
    const t = setInterval(() => { if (check()) clearInterval(t); }, 100);
  });
}

function roiRect() {
  const w = 144, h = 208, y = 16;
  return { x: side === 'right' ? PROC_W - w - 12 : 12, y, w, h };
}

// ---------- UI setup ----------
function setupControls() {
  const presetSel = $('preset');
  for (const [k, v] of Object.entries(PRESETS)) presetSel.add(new Option(v.label, k));
  const keySel = $('key');
  NOTE_NAMES.forEach((n, i) => keySel.add(new Option(`${n} major`, i)));

  presetSel.onchange = () => synth && synth.setPreset(presetSel.value);
  keySel.onchange = () => { synth && synth.setKey(+keySel.value); renderChordMap(); };
  $('side').onchange = (e) => { side = e.target.value; };

  for (const key of ['crMin', 'crMax', 'cbMin', 'cbMax', 'defectDepthRatio']) {
    const input = $(key);
    input.oninput = () => { params[key] = +input.value; syncSliders(); };
  }
  $('resetBtn').onclick = () => { params = { ...DEFAULT_PARAMS }; syncSliders(); };
  $('calibBtn').onclick = calibrate;
  document.querySelectorAll('.tab').forEach((tab) => {
    tab.onclick = () => {
      document.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
      tab.classList.add('active');
      stage = tab.dataset.stage;
    };
  });
  syncSliders();
}

function syncSliders() {
  for (const key of ['crMin', 'crMax', 'cbMin', 'cbMax', 'defectDepthRatio']) {
    const input = $(key);
    input.value = params[key];
    input.nextElementSibling.textContent = params[key];
  }
}

function renderChordMap(active = -1) {
  const ol = $('chordMap');
  ol.innerHTML = '';
  for (let f = 1; f <= 5; f++) {
    const deg = FINGER_TO_DEGREE[f];
    const c = synth ? synth.chordName(deg) : { name: '', roman: '' };
    const li = document.createElement('li');
    if (f === active) li.className = 'on';
    li.innerHTML = `<b>${f} ✋</b>${c.name}`;
    ol.appendChild(li);
  }
}

// ---------- Start ----------
async function start() {
  const btn = $('startBtn');
  btn.disabled = true;
  btn.textContent = 'Starting camera…';
  try {
    synth = new ChordSynth();
    await synth.start();
    synth.setKey(+$('key').value);
    synth.setPreset($('preset').value);
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }, audio: false,
    });
    video.srcObject = stream;
    await video.play();
  } catch (err) {
    btn.disabled = false;
    btn.textContent = 'Try again';
    $('overlay').querySelector('p').textContent = `Couldn't start: ${err.message}. Allow camera access and reload.`;
    return;
  }
  $('overlay').hidden = true;
  $('calibBtn').disabled = false;
  renderChordMap();
  running = true;
  requestAnimationFrame(loop);
}

// ---------- Main loop ----------
function loop() {
  if (!running) return;
  // Draw the mirrored camera frame at processing resolution.
  pctx.save();
  pctx.translate(PROC_W, 0);
  pctx.scale(-1, 1);
  pctx.drawImage(video, 0, 0, PROC_W, PROC_H);
  pctx.restore();

  const roi = roiRect();
  let result = null;
  if (!calibrating) {
    const t0 = performance.now();
    const src = cv.matFromImageData(pctx.getImageData(roi.x, roi.y, roi.w, roi.h));
    result = tracker.process(src, params);
    src.delete();
    $('procMs').textContent = `${(performance.now() - t0).toFixed(1)} ms`;
    update(result, roi);
  }
  draw(result, roi);
  drawScope();

  fpsFrames++;
  const now = performance.now();
  if (now - fpsT0 > 500) {
    $('fps').textContent = `${Math.round((fpsFrames * 1000) / (now - fpsT0))} fps`;
    fpsFrames = 0; fpsT0 = now;
  }
  requestAnimationFrame(loop);
}

/** Temporal smoothing + music control. */
function update(r, roi) {
  // Majority vote over recent frames removes one-frame flicker in the finger count.
  history.push(r.found ? r.fingers : -1);
  if (history.length > STABLE_FRAMES) history.shift();
  const votes = {};
  for (const h of history) votes[h] = (votes[h] || 0) + 1;
  const [winner, count] = Object.entries(votes).sort((a, b) => b[1] - a[1])[0];
  if (count >= Math.ceil(STABLE_FRAMES / 2) + 1) stableFingers = +winner;

  if (!r.found) {
    smooth = null;
    if (stableFingers === -1) {
      synth.stop();
      setReadout('–', '—', 'no hand in box', -1);
    }
    return;
  }

  const target = { x: r.cx / roi.w, y: r.cy / roi.h, angle: r.angle };
  smooth = smooth
    ? { x: smooth.x + SMOOTH * (target.x - smooth.x), y: smooth.y + SMOOTH * (target.y - smooth.y),
        angle: smooth.angle + SMOOTH * (target.angle - smooth.angle) }
    : target;

  const clamp01 = (v) => Math.max(0, Math.min(1, v));
  const controls = {
    height: clamp01((0.85 - smooth.y) / 0.6),    // centroid near the top of the box = loud
    x: clamp01((smooth.x - 0.15) / 0.7),
    tilt: clamp01(Math.abs(smooth.angle) / 40),
  };
  synth.setControls(controls);
  $('mVol').style.width = `${controls.height * 100}%`;
  $('mBright').style.width = `${controls.x * 100}%`;
  $('mTilt').style.width = `${controls.tilt * 100}%`;
  $('tiltDeg').textContent = `${smooth.angle.toFixed(0)}°`;

  if (stableFingers >= 0) {
    synth.play(stableFingers);
    if (stableFingers === 0) setReadout('0', '—', 'fist · silent', -1);
    else {
      const c = synth.chordName(FINGER_TO_DEGREE[stableFingers]);
      setReadout(String(stableFingers), c.name, `chord ${c.roman}`, stableFingers);
    }
  }
}

function setReadout(fingers, chord, roman, active) {
  const key = fingers + chord + roman;
  if (last === key) return;
  last = key;
  $('fingerCount').textContent = fingers;
  $('chordName').textContent = chord;
  $('chordRoman').textContent = roman;
  renderChordMap(active);
}

// ---------- Drawing ----------
function draw(r, roi) {
  const S = SCALE;
  const R = { x: roi.x * S, y: roi.y * S, w: roi.w * S, h: roi.h * S };

  // Background: mirrored camera, dimmed outside the box.
  vctx.save();
  vctx.translate(view.width, 0);
  vctx.scale(-1, 1);
  vctx.drawImage(video, 0, 0, view.width, view.height);
  vctx.restore();
  vctx.fillStyle = 'rgba(8,8,10,0.55)';
  vctx.fillRect(0, 0, view.width, R.y);
  vctx.fillRect(0, R.y + R.h, view.width, view.height - R.y - R.h);
  vctx.fillRect(0, R.y, R.x, R.h);
  vctx.fillRect(R.x + R.w, R.y, view.width - R.x - R.w, R.h);

  // Intermediate pipeline stages, drawn inside the box.
  if (!calibrating && (stage === 'raw' || stage === 'clean')) {
    cv.imshow(maskCanvas, stage === 'raw' ? tracker.mats.rawMask : tracker.mats.mask);
    vctx.imageSmoothingEnabled = false;
    vctx.drawImage(maskCanvas, R.x, R.y, R.w, R.h);
    vctx.imageSmoothingEnabled = true;
  } else if (!calibrating && stage === 'shape') {
    vctx.fillStyle = '#000';
    vctx.fillRect(R.x, R.y, R.w, R.h);
  }

  // Box outline + label.
  vctx.strokeStyle = r && r.found ? '#ffb547' : 'rgba(236,235,230,0.6)';
  vctx.lineWidth = 2;
  vctx.setLineDash(r && r.found ? [] : [8, 6]);
  vctx.strokeRect(R.x, R.y, R.w, R.h);
  vctx.setLineDash([]);
  vctx.font = '600 13px "JetBrains Mono", monospace';
  vctx.fillStyle = 'rgba(236,235,230,0.85)';
  vctx.fillText(calibrating ? 'COVER THE SQUARE WITH YOUR PALM' : 'HAND ZONE', R.x + 4, R.y - 8);

  if (calibrating) {
    const p = calibPatch(roi);
    vctx.strokeStyle = '#5ee6c4';
    vctx.lineWidth = 3;
    vctx.strokeRect(p.x * S, p.y * S, p.w * S, p.h * S);
    return;
  }
  if (!r || !r.found) return;

  const P = (q) => [R.x + q.x * S, R.y + q.y * S];
  const poly = (pts, close = true) => {
    vctx.beginPath();
    pts.forEach((q, i) => (i ? vctx.lineTo(...P(q)) : vctx.moveTo(...P(q))));
    if (close) vctx.closePath();
  };

  if (stage === 'shape' || stage === 'camera') {
    poly(r.contourPts);
    if (stage === 'shape') { vctx.fillStyle = 'rgba(255,181,71,0.25)'; vctx.fill(); }
    vctx.strokeStyle = '#ffb547'; vctx.lineWidth = 2; vctx.stroke();
  }
  poly(r.hullPts);
  vctx.strokeStyle = '#5ee6c4'; vctx.lineWidth = 2; vctx.stroke();

  for (const v of r.valleys) { vctx.fillStyle = '#ff6b6b'; dot(...P(v), 6); }
  for (const t of r.tips) { vctx.fillStyle = '#ffd28a'; dot(...P(t), 7); }

  // Centroid + orientation axis.
  const [cx, cy] = P({ x: r.cx, y: r.cy });
  const a = (r.angle * Math.PI) / 180, L = 60;
  vctx.strokeStyle = '#ecebe6'; vctx.lineWidth = 2;
  vctx.beginPath();
  vctx.moveTo(cx - Math.sin(a) * L, cy + Math.cos(a) * L);
  vctx.lineTo(cx + Math.sin(a) * L, cy - Math.cos(a) * L);
  vctx.stroke();
  vctx.fillStyle = '#ecebe6'; dot(cx, cy, 7);
}

function dot(x, y, rad) { vctx.beginPath(); vctx.arc(x, y, rad, 0, Math.PI * 2); vctx.fill(); }

function drawScope() {
  const w = scope.width, h = scope.height;
  sctx.clearRect(0, 0, w, h);
  if (!synth) return;
  const data = synth.waveform();
  sctx.strokeStyle = '#ffb547';
  sctx.lineWidth = 3;
  sctx.beginPath();
  for (let i = 0; i < data.length; i++) {
    const x = (i / (data.length - 1)) * w;
    const y = h / 2 + data[i] * h * 0.45 * 4;
    i ? sctx.lineTo(x, y) : sctx.moveTo(x, y);
  }
  sctx.stroke();
}

// ---------- Skin auto-calibration ----------
function calibPatch(roi) {
  const s = 36;
  return { x: roi.x + roi.w / 2 - s / 2, y: roi.y + roi.h * 0.55 - s / 2, w: s, h: s };
}

async function calibrate() {
  calibrating = true;
  synth.stop();
  const cd = $('countdown');
  cd.hidden = false;
  for (let i = 3; i > 0; i--) { cd.textContent = i; await new Promise((r) => setTimeout(r, 800)); }
  cd.hidden = true;
  const p = calibPatch(roiRect());
  const src = cv.matFromImageData(pctx.getImageData(p.x, p.y, p.w, p.h));
  Object.assign(params, tracker.calibrate(src));
  src.delete();
  syncSliders();
  history = [];
  calibrating = false;
}

// ---------- Boot ----------
setupControls();
renderChordMap();
waitForOpenCV().then((res) => {
  cv = res.cv;
  tracker = createTracker(cv);
  $('cvStatus').textContent = `OpenCV ${cv.getBuildInformation ? cv.getBuildInformation().match(/Version control:\s*(\S+)/)?.[1] || '4.x' : '4.x'} ready`;
  $('cvStatus').classList.add('ok');
  const btn = $('startBtn');
  btn.disabled = false;
  btn.textContent = 'Start camera & sound';
  btn.onclick = start;
});
