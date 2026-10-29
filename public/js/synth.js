// synth.js — the music layer (Tone.js). Knows nothing about computer vision:
// app.js tells it how many fingers are up and passes hand position/tilt as 0..1 values.
/* global Tone */

export const NOTE_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'];

// Chord types: name suffix, friendly label, and intervals in semitones from the root.
export const QUALITIES = {
  maj:  { suffix: '',     label: 'Major',      iv: [0, 4, 7] },
  min:  { suffix: 'm',    label: 'Minor',      iv: [0, 3, 7] },
  '7':  { suffix: '7',    label: '7',          iv: [0, 4, 7, 10] },
  maj7: { suffix: 'maj7', label: 'maj7',       iv: [0, 4, 7, 11] },
  m7:   { suffix: 'm7',   label: 'm7',         iv: [0, 3, 7, 10] },
  sus2: { suffix: 'sus2', label: 'sus2',       iv: [0, 2, 7] },
  sus4: { suffix: 'sus4', label: 'sus4',       iv: [0, 5, 7] },
  add9: { suffix: 'add9', label: 'add9',       iv: [0, 4, 7, 14] },
  '6':  { suffix: '6',    label: '6',          iv: [0, 4, 7, 9] },
  m6:   { suffix: 'm6',   label: 'm6',         iv: [0, 3, 7, 9] },
  '9':  { suffix: '9',    label: '9',          iv: [0, 4, 7, 10, 14] },
  m9:   { suffix: 'm9',   label: 'm9',         iv: [0, 3, 7, 10, 14] },
  dim:  { suffix: 'dim',  label: 'dim',        iv: [0, 3, 6] },
  aug:  { suffix: 'aug',  label: 'aug',        iv: [0, 4, 8] },
  '5':  { suffix: '5',    label: '5 (power)',  iv: [0, 7, 12] },
};

// Display order for the editor (JS would otherwise list number-like keys such as '7' first).
export const QUALITY_ORDER = ['maj', 'min', '7', 'maj7', 'm7', 'sus2', 'sus4', 'add9', '6', 'm6', '9', 'm9', 'dim', 'aug', '5'];

// What each finger count plays. Every entry is one of:
//   { type: 'chord', root: 0-11, quality: key of QUALITIES }   e.g. D maj7 = { root: 2, quality: 'maj7' }
//   { type: 'note', pc: 0-11, octave }                          one single note (pc 0 = C)
//   { type: 'silence' }
// Default: fist = Em, then 1-2-3-4 fingers = C G Am F (the pop I–V–vi–IV progression), 5 = Dm.
export const DEFAULT_ASSIGNMENTS = {
  0: { type: 'chord', root: 4, quality: 'min' },
  1: { type: 'chord', root: 0, quality: 'maj' },
  2: { type: 'chord', root: 7, quality: 'maj' },
  3: { type: 'chord', root: 9, quality: 'min' },
  4: { type: 'chord', root: 5, quality: 'maj' },
  5: { type: 'chord', root: 2, quality: 'min' },
};

// Ready-made progressions for "quick fill": [semitones above the key, chord type] for fist, 1, 2, 3, 4, 5.
export const PROGRESSIONS = {
  pop:   { label: 'Pop (I–V–vi–IV)',       chords: [[4, 'min'], [0, 'maj'], [7, 'maj'], [9, 'min'], [5, 'maj'], [2, 'min']] },
  sad:   { label: 'Emotional (vi–IV–I–V)', chords: [[4, 'm7'], [9, 'min'], [5, 'maj'], [0, 'maj'], [7, 'maj'], [2, 'm7']] },
  fifties: { label: '50s (I–vi–IV–V)',     chords: [[2, 'min'], [0, 'maj'], [9, 'min'], [5, 'maj'], [7, 'maj'], [0, 'add9']] },
  jazz:  { label: 'Jazz (ii7–V7–Imaj7)',   chords: [[9, 'm7'], [2, 'm7'], [7, '7'], [0, 'maj7'], [5, 'maj7'], [4, 'm7']] },
  lofi:  { label: 'Lo-fi (maj7 & m9)',     chords: [[4, 'm7'], [0, 'maj7'], [9, 'm9'], [5, 'maj7'], [2, 'm9'], [7, 'sus4']] },
  blues: { label: 'Blues (I7–IV7–V7)',     chords: [[0, '5'], [0, '7'], [5, '7'], [7, '7'], [9, 'm7'], [0, '9']] },
};

export function progressionAssignments(key, name) {
  const out = {};
  PROGRESSIONS[name].chords.forEach(([st, q], f) => { out[f] = { type: 'chord', root: (key + st) % 12, quality: q }; });
  return out;
}

/** UI text for an assignment: { name: 'Dmaj7', sub: 'D · maj7' } */
export function describeAssignment(a) {
  if (!a || a.type === 'silence') return { name: '—', sub: 'silence' };
  if (a.type === 'note') return { name: `${NOTE_NAMES[a.pc]}${a.octave}`, sub: 'single note' };
  const q = QUALITIES[a.quality] || QUALITIES.maj;
  return { name: NOTE_NAMES[a.root] + q.suffix, sub: `${NOTE_NAMES[a.root]} ${q.label.toLowerCase()}` };
}

/** MIDI notes for an assignment (empty = silence). Bass note + chord voiced around middle C. */
export function notesFor(a) {
  if (!a || a.type === 'silence') return [];
  if (a.type === 'note') return [12 * (a.octave + 1) + a.pc]; // C4 = MIDI 60
  const q = QUALITIES[a.quality] || QUALITIES.maj;
  const base = 60 + a.root - (a.root > 6 ? 12 : 0);          // keep chords between F#3 and F#4
  return [base - 12, ...q.iv.map((i) => base + i)];
}

export const PRESETS = {
  pad: {
    label: 'Warm Pad',
    synth: { oscillator: { type: 'fatsawtooth', count: 3, spread: 24 },
             envelope: { attack: 0.35, decay: 0.3, sustain: 0.8, release: 1.4 } },
  },
  pluck: {
    label: 'Pluck',
    synth: { oscillator: { type: 'triangle' },
             envelope: { attack: 0.005, decay: 0.5, sustain: 0.15, release: 0.8 } },
  },
  organ: {
    label: 'Organ',
    synth: { oscillator: { type: 'fatsine', count: 2, spread: 8 },
             envelope: { attack: 0.03, decay: 0.1, sustain: 1, release: 0.3 } },
  },
};

export class ChordSynth {
  constructor() {
    this.current = null;    // currently held MIDI notes
    this.assign = structuredClone(DEFAULT_ASSIGNMENTS);
    this.analyser = new Tone.Analyser('waveform', 1024);
    this.volume = new Tone.Volume(-12);
    this.filter = new Tone.Filter(1200, 'lowpass', -24);
    this.vibrato = new Tone.Vibrato(5.5, 0);
    this.reverb = new Tone.Freeverb({ roomSize: 0.8, dampening: 3000, wet: 0.3 });
    this.synth = new Tone.PolySynth(Tone.Synth);
    this.synth.maxPolyphony = 12;
    this.synth.chain(this.vibrato, this.filter, this.reverb, this.volume, this.analyser, Tone.getDestination());
    this.setPreset('pad');
  }

  async start() { await Tone.start(); }

  setPreset(name) {
    this.synth.releaseAll();
    this.synth.set(PRESETS[name].synth);
    this.current = null;
  }

  setAssignments(map) { this.assign = structuredClone(map); this.current = null; this.synth.releaseAll(); }

  labelFor(fingers) { return describeAssignment(this.assign[fingers]); }

  /** Play an assignment for a moment (editor preview). */
  preview(a) {
    const notes = notesFor(a);
    if (!notes.length) return;
    this.stop();
    this.synth.triggerAttackRelease(notes.map((n) => Tone.Frequency(n, 'midi').toFrequency()), 0.9);
  }

  /** Play whatever is assigned to this finger count. */
  play(fingers) {
    const notes = notesFor(this.assign[fingers]);
    if (!notes.length) { this.stop(); return; }
    if (this.current && this.current.join() === notes.join()) return;
    const now = Tone.now();
    const hz = (list) => list.map((n) => Tone.Frequency(n, 'midi').toFrequency());
    if (this.current) this.synth.triggerRelease(hz(this.current), now);
    this.synth.triggerAttack(hz(notes), now + 0.01);
    this.current = notes;
  }

  stop() {
    if (!this.current) return;
    this.synth.triggerRelease(this.current.map((n) => Tone.Frequency(n, 'midi').toFrequency()));
    this.current = null;
  }

  /**
   * Continuous controls, all 0..1:
   *   height -> loudness (hand high = loud)
   *   x      -> filter cutoff (left = dark, right = bright), log scale 200 Hz – 6 kHz
   *   tilt   -> vibrato depth
   */
  setControls({ height, x, tilt }) {
    this.volume.volume.rampTo(-36 + 30 * height, 0.08);
    this.filter.frequency.rampTo(200 * Math.pow(30, x), 0.08);
    this.vibrato.depth.rampTo(Math.min(1, tilt) * 0.7, 0.1);
  }

  waveform() { return this.analyser.getValue(); }
}
