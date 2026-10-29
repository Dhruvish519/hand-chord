// app.js — wires camera -> OpenCV tracker -> synth, and draws the UI.
import { createTracker, DEFAULT_PARAMS } from './handTracker.js';
import { ChordSynth, PRESETS, NOTE_NAMES, DEFAULT_ASSIGNMENTS, describeAssignment } from './synth.js';
import { loadFingerModel, CLASSES, NONE } from './fingerModel.js';

const $ = (id) => document.getElementById(id);
const video = $('video'), view = $('view'), proc = $('proc'), maskCanvas = $('maskCanvas'), scope = $('scope');
const vctx = view.getContext('2d');
const pctx = proc.getContext('2d', { willReadFrequently: true });
const sctx = scope.getContext('2d');

const PROC_W = proc.width, PROC_H = proc.height;   // 320 x 240: processing resolution
const SCALE = view.width / PROC_W;                  // display is 2x the processing size
const STABLE_FRAMES = 7;                            // finger count must win a vote over the last 7 frames
const SMOOTH = 0.35;                                // exponential smoothing for position/tilt
const MIN_CONF = 0.55;                              // CNN must be at least this sure before a chord plays

// A mirrored full-resolution copy of the frame, and a 112x112 canvas for the CNN input crop.
const hires = document.createElement('canvas'); hires.width = 640; hires.height = 480;
const hctx = hires.getContext('2d');
const cropCanvas = document.createElement('canvas'); cropCanvas.width = cropCanvas.height = 112;
const cctx = cropCanvas.getContext('2d', { willReadFrequently: true });

let cv, tracker, synth, model = null;
let mode = 'cnn';          // 'cnn' = trained network, 'classic' = convexity defects
let probsEma = null;
let tiltEma = null;        // smoothed (sin, cos) of the CNN's tilt estimate
const STORE_KEY = 'handchord.assignments.v1';
let assignments = loadAssignments();
let lastDet = null;
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
  keySel.onchange = () => { synth && synth.setKey(+keySel.value); renderAssignEditor(); renderChordMap(); };
  $('assignReset').onclick = () => {
    assignments = structuredClone(DEFAULT_ASSIGNMENTS); saveAssignments();
    if (synth) synth.setAssignments(assignments);
    renderAssignEditor(); renderChordMap();
  };
  $('side').onchange = (e) => { side = e.target.value; };
  $('mode').onchange = (e) => { mode = e.target.value; history = []; probsEma = null; };

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

// ---------- Finger -> note assignments ----------
function loadAssignments() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORE_KEY));
    if (saved && [0, 1, 2, 3, 4, 5].every((f) => saved[f] && saved[f].type)) return saved;
  } catch (e) { /* storage blocked or empty: use defaults */ }
  return structuredClone(DEFAULT_ASSIGNMENTS);
}

function saveAssignments() {
  try { localStorage.setItem(STORE_KEY, JSON.stringify(assignments)); } catch (e) { /* not persisted */ }
}

const encode = (a) => (a.type === 'chord' ? `c${a.degree}` : a.type === 'note' ? `n${a.pc}-${a.octave}` : 's');
function decode(v) {
  if (v === 's') return { type: 'silence' };
  if (v[0] === 'c') return { type: 'chord', degree: +v.slice(1) };
  const [pc, oct] = v.slice(1).split('-').map(Number);
  return { type: 'note', pc, octave: oct };
}

/** One dropdown per finger count: chords of the current key, single notes, or silence. */
function renderAssignEditor() {
  const root = +$('key').value;
  const box = $('assignEditor');
  box.innerHTML = '';
  for (let f = 0; f <= 5; f++) {
    const label = document.createElement('label');
    label.innerHTML = `<span>${f === 0 ? '0 · fist' : `${f} finger${f > 1 ? 's' : ''}`}</span>`;
    const sel = document.createElement('select');
    const chords = document.createElement('optgroup'); chords.label = 'Chords in key';
    for (let d = 0; d < 7; d++) {
      const a = { type: 'chord', degree: d }, t = describeAssignment(a, root);
      chords.appendChild(new Option(`${t.name} (${t.sub.replace('chord ', '')})`, encode(a)));
    }
    sel.appendChild(chords);
    for (const oct of [3, 4, 5]) {
      const g = document.createElement('optgroup'); g.label = `Single notes · octave ${oct}`;
      NOTE_NAMES.forEach((n, pc) => g.appendChild(new Option(`${n}${oct}`, encode({ type: 'note', pc, octave: oct }))));
      sel.appendChild(g);
    }
    sel.appendChild(new Option('Silence', 's'));
    sel.value = encode(assignments[f]);
    sel.onchange = () => {
      assignments[f] = decode(sel.value);
      saveAssignments();
      if (synth) synth.setAssignments(assignments);
      last = null; renderChordMap();
    };
    label.appendChild(sel);
    box.appendChild(label);
  }
}

function renderChordMap(active = -1) {
  const root = +$('key').value;
  const ol = $('chordMap');
  ol.innerHTML = '';
  for (let f = 0; f <= 5; f++) {
    const t = describeAssignment(assignments[f], root);
    const li = document.createElement('li');
    if (f === active) li.className = 'on';
    li.innerHTML = `<b>${f === 0 ? '0 ✊' : `${f} ✋`}</b>${t.name}`;
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
    synth.setAssignments(assignments);
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
    lastDet = mode === 'cnn' && model ? detectWithCNN(result, roi) : null;
    $('procMs').textContent = `${(performance.now() - t0).toFixed(1)} ms`;
    update(result, roi, lastDet);
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

// ---------- Trained CNN ----------
/** Square crop (processing-pixel coords) around a box, kept inside the frame. */
function squareAround(cx, cy, side) {
  side = Math.min(side, PROC_H);
  const x = Math.max(0, Math.min(PROC_W - side, cx - side / 2));
  const y = Math.max(0, Math.min(PROC_H - side, cy - side / 2));
  return { x, y, side };
}

function classifyBox(box) {
  const k = hires.width / PROC_W;
  cctx.drawImage(hires, box.x * k, box.y * k, box.side * k, box.side * k, 0, 0, 112, 112);
  const mat = cv.matFromImageData(cctx.getImageData(0, 0, 112, 112));
  const res = model.classify(mat);
  mat.delete();
  return res;
}

/**
 * Two candidate crops for the CNN:
 *   A) around the skin blob that OpenCV found (tight, follows the hand), and
 *   B) the whole hand zone (backup when skin detection misses the hand).
 * The CNN decides the finger count, or "none" for faces, objects and other gestures.
 */
function detectWithCNN(r, roi) {
  hctx.save(); hctx.translate(hires.width, 0); hctx.scale(-1, 1);
  hctx.drawImage(video, 0, 0, hires.width, hires.height); hctx.restore();

  const candidates = [];
  if (r.found) {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const q of r.contourPts) { x0 = Math.min(x0, q.x); y0 = Math.min(y0, q.y); x1 = Math.max(x1, q.x); y1 = Math.max(y1, q.y); }
    candidates.push(squareAround(roi.x + (x0 + x1) / 2, roi.y + (y0 + y1) / 2, Math.max(x1 - x0, y1 - y0) * 1.35));
  }
  candidates.push(squareAround(roi.x + roi.w / 2, roi.y + roi.h * 0.45, roi.w * 1.05));

  let best = null;
  for (const box of candidates) {
    const res = classifyBox(box);
    const probs = res.probs;
    const top = probs.indexOf(Math.max(...probs));
    const score = top === NONE ? 0 : probs[top];
    if (!best || score > best.score) best = { box, probs, score, res };
    if (score >= 0.8) break; // confident enough, skip the backup crop
  }

  // Smooth the probabilities over time so one odd frame can't change the chord.
  probsEma = probsEma ? probsEma.map((v, i) => 0.5 * v + 0.5 * best.probs[i]) : best.probs.slice();
  const top = probsEma.indexOf(Math.max(...probsEma));
  const fingers = top !== NONE && probsEma[top] >= MIN_CONF ? top : -1;
  // Tilt: smooth sin/cos (not the angle itself, so 179 deg and -179 deg average correctly).
  let tilt = null;
  if (best.res.tilt !== null) {
    const s = best.res.sin, c = best.res.cos;
    tiltEma = tiltEma ? [0.6 * tiltEma[0] + 0.4 * s, 0.6 * tiltEma[1] + 0.4 * c] : [s, c];
    tilt = (Math.atan2(tiltEma[0], tiltEma[1]) * 180) / Math.PI;
  }
  return { fingers, conf: probsEma[top], top, probs: probsEma, box: best.box, tilt };
}

function renderProbs(det) {
  const bars = document.querySelectorAll('#probs i');
  bars.forEach((b, i) => { b.style.setProperty('--p', det ? `${Math.round(det.probs[i] * 100)}%` : '0%'); b.parentElement.classList.toggle('hi', !!det && det.top === i); });
  $('confidence').textContent = det ? `${CLASSES[det.top] === 'none' ? 'none' : CLASSES[det.top] + ' fingers'} · ${Math.round(det.conf * 100)}%` : '—';
}

/** Temporal smoothing + music control. */
function update(r, roi, det) {
  if (det) renderProbs(det);
  const frameFingers = det ? det.fingers : (r.found ? r.fingers : -1);
  // Majority vote over recent frames removes one-frame flicker in the finger count.
  history.push(frameFingers);
  if (history.length > STABLE_FRAMES) history.shift();
  const votes = {};
  for (const h of history) votes[h] = (votes[h] || 0) + 1;
  const [winner, count] = Object.entries(votes).sort((a, b) => b[1] - a[1])[0];
  if (count >= Math.ceil(STABLE_FRAMES / 2) + 1) stableFingers = +winner;

  if (stableFingers === -1) {
    synth.stop();
    setReadout('–', '—', det && r.found ? 'not a counting hand' : 'no hand in box', -1);
  }
  // Position comes from the OpenCV contour (moments); without it, the CNN crop centre.
  // Tilt comes from the CNN (trained on rotated hands) when available, else from moments.
  let target = null;
  if (r.found) target = { x: r.cx / roi.w, y: r.cy / roi.h, angle: r.angle };
  else if (det && det.fingers >= 0) target = { x: (det.box.x + det.box.side / 2 - roi.x) / roi.w, y: (det.box.y + det.box.side / 2 - roi.y) / roi.h, angle: 0 };
  if (target && det && det.tilt !== null && det.fingers >= 0) target.angle = det.tilt;
  if (!target) { smooth = null; return; }

  smooth = smooth
    ? { x: smooth.x + SMOOTH * (target.x - smooth.x), y: smooth.y + SMOOTH * (target.y - smooth.y),
        angle: smooth.angle + SMOOTH * (target.angle - smooth.angle) }
    : target;

  const clamp01 = (v) => Math.max(0, Math.min(1, v));
  const controls = {
    height: clamp01((0.85 - smooth.y) / 0.6),    // centroid near the top of the box = loud
    x: clamp01((smooth.x - 0.15) / 0.7),
    tilt: clamp01((Math.abs(smooth.angle) - 8) / 27), // 8 deg dead zone, full vibrato at 35 deg
  };
  synth.setControls(controls);
  $('mVol').style.width = `${controls.height * 100}%`;
  $('mBright').style.width = `${controls.x * 100}%`;
  $('mTilt').style.width = `${controls.tilt * 100}%`;
  $('tiltDeg').textContent = `${smooth.angle.toFixed(0)}°`;

  if (stableFingers >= 0) {
    synth.play(stableFingers);
    const t = synth.labelFor(stableFingers);
    setReadout(String(stableFingers), t.name, t.sub, stableFingers);
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
  if (lastDet && mode === 'cnn') {
    const b = lastDet.box, ok = lastDet.fingers >= 0;
    vctx.strokeStyle = ok ? '#5ee6c4' : 'rgba(255,107,107,0.8)';
    vctx.lineWidth = 2; vctx.setLineDash([4, 4]);
    vctx.strokeRect(b.x * S, b.y * S, b.side * S, b.side * S); vctx.setLineDash([]);
    const label = `CNN: ${ok ? lastDet.fingers : 'none'} ${Math.round(lastDet.conf * 100)}%`;
    vctx.font = '600 14px "JetBrains Mono", monospace';
    const tw = vctx.measureText(label).width;
    vctx.fillStyle = ok ? '#5ee6c4' : '#ff6b6b';
    vctx.fillRect(b.x * S, b.y * S + b.side * S, tw + 12, 22);
    vctx.fillStyle = '#0f0f12'; vctx.fillText(label, b.x * S + 6, b.y * S + b.side * S + 16);
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
renderAssignEditor();
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
  loadFingerModel(cv)
    .then((m) => { model = m; $('modelStatus').textContent = 'CNN model ready'; $('modelStatus').classList.add('ok'); })
    .catch((err) => {
      console.error(err);
      mode = 'classic'; $('mode').value = 'classic';
      $('modelStatus').textContent = 'CNN unavailable – using classical';
    });
});
