// synth.js — the music layer (Tone.js). Knows nothing about computer vision:
// app.js tells it how many fingers are up and passes hand position/tilt as 0..1 values.
/* global Tone */

const MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11];
export const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const ROMAN = ['I', 'ii', 'iii', 'IV', 'V', 'vi', 'vii°'];

// What each finger count plays. Every entry is one of:
//   { type: 'chord', degree: 0-6 }        a chord built on that step of the chosen key
//   { type: 'note', pc: 0-11, octave }    one single note (pc 0 = C)
//   { type: 'silence' }
// Default: fist = iii, then 1-2-3-4 fingers = the I–V–vi–IV progression, 5 = ii.
export const DEFAULT_ASSIGNMENTS = {
  0: { type: 'chord', degree: 2 },
  1: { type: 'chord', degree: 0 },
  2: { type: 'chord', degree: 4 },
  3: { type: 'chord', degree: 5 },
  4: { type: 'chord', degree: 3 },
  5: { type: 'chord', degree: 1 },
};

/** UI text for an assignment in a given key (root 0 = C): { name, sub } */
export function describeAssignment(a, root = 0) {
  if (!a || a.type === 'silence') return { name: '—', sub: 'silence' };
  if (a.type === 'note') return { name: `${NOTE_NAMES[a.pc]}${a.octave}`, sub: 'note' };
  const rootName = NOTE_NAMES[(root + MAJOR_SCALE[a.degree]) % 12];
  const quality = [1, 2, 5].includes(a.degree) ? 'm' : a.degree === 6 ? 'dim' : '';
  return { name: rootName + quality, sub: `chord ${ROMAN[a.degree]}` };
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
    this.root = 0;          // key: 0 = C major
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

  setKey(rootIndex) { this.root = rootIndex; this.current = null; this.synth.releaseAll(); }

  setAssignments(map) { this.assign = structuredClone(map); this.current = null; this.synth.releaseAll(); }

  /** Triad on a scale degree, plus a bass note one octave below the chord root. */
  chordNotes(degree) {
    const base = 60 + this.root; // middle C + key offset
    const note = (d) => base + MAJOR_SCALE[d % 7] + 12 * Math.floor(d / 7);
    const triad = [note(degree), note(degree + 2), note(degree + 4)];
    return [triad[0] - 12, ...triad];
  }

  chordName(degree) {
    const rootName = NOTE_NAMES[(this.root + MAJOR_SCALE[degree]) % 12];
    const quality = [1, 2, 5].includes(degree) ? 'm' : degree === 6 ? 'dim' : '';
    return { name: rootName + quality, roman: ROMAN[degree] };
  }

  /** MIDI notes for an assignment (empty = silence). */
  notesFor(a) {
    if (!a || a.type === 'silence') return [];
    if (a.type === 'note') return [12 * (a.octave + 1) + a.pc]; // C4 = MIDI 60
    return this.chordNotes(a.degree);
  }

  /** Short text for the UI: { name, sub } */
  label(a) { return describeAssignment(a, this.root); }

  labelFor(fingers) { return this.label(this.assign[fingers]); }

  /** Play whatever is assigned to this finger count. */
  play(fingers) {
    const notes = this.notesFor(this.assign[fingers]);
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
