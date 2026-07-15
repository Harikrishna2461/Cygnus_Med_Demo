/**
 * Shunt Classification Accuracy Test Streams
 *
 * Streams 1–3  : Real clinical data from "multiple shunts in 1 sesh" folder.
 *                Each is the most-complete snapshot from its session folder.
 * Streams 4–10 : Synthesised realistic data using CHIVA scanning patterns
 *                with anatomically-grounded posX/posY coordinates.
 *
 * Ground-truth labels are what a CHIVA-trained clinician would classify from
 * the clips listed.  They are used to evaluate the LLM's accuracy.
 */

// ── Real streams ──────────────────────────────────────────────────────────────

// Stream 1 — shunt type 1 folder, last file "2 - shunt type 1.json"
// Only 1 EP clip present (recording was sparse).  GT = Type 1.
const STREAM_1_REAL = [
  {
    sequenceNumber: 0, flow: 'EP', fromType: 'N1', toType: 'N2',
    step: 'SFJ-Knee', legSide: 'right',
    posXRatio: 0.275, posYRatio: 0.05,
    confidence: 0.93, reflux_duration: 0.0,
    description: 'EP N1→N2 at SFJ-Knee — normal saphenous entry, groin level',
    clipPath: 'SFJ-Knee_00_EP_N1-N2.mp4', eliminationTest: '',
  },
];

// Stream 2 — shunt type 2 folder, last file "3 - shunt type 2c.json"
// 1 EP + 2 RP clips.  GT = Type 2C.
const STREAM_2_REAL = [
  {
    sequenceNumber: 0, flow: 'EP', fromType: 'N2', toType: 'N2',
    step: 'SFJ-Knee', legSide: 'right',
    posXRatio: 0.23, posYRatio: 0.05,
    confidence: 0.92, reflux_duration: 0.0,
    description: 'EP N2→N2 — GSV competent at groin, no SFJ incompetence',
    clipPath: 'SFJ-Knee_00_EP_N2-N2.mp4', eliminationTest: '',
  },
  {
    sequenceNumber: 1, flow: 'RP', fromType: 'N3', toType: 'N1',
    step: 'SFJ-Knee', legSide: 'right',
    posXRatio: 0.173, posYRatio: 0.132,
    confidence: 0.87, reflux_duration: 1.1,
    description: 'RP N3→N1 — tributary reflux into deep system, Type 2B/C pattern',
    clipPath: 'SFJ-Knee_01_RP_N3-N1.mp4', eliminationTest: '',
  },
  {
    sequenceNumber: 2, flow: 'RP', fromType: 'N2', toType: 'N1',
    step: 'SFJ-Knee', legSide: 'right',
    posXRatio: 0.21, posYRatio: 0.212,
    confidence: 0.84, reflux_duration: 0.9,
    description: 'RP N2→N1 — additional GSV reflux confirms Type 2C dual-axis pattern',
    clipPath: 'SFJ-Knee_02_RP_N2-N1.mp4', eliminationTest: '',
  },
];

// Stream 3 — shunt type 3 folder, last file "5 - shunt type 1+2 (after making ep23 reflux).json"
// 2 EP + 2 RP clips.  GT = Type 1+2.
const STREAM_3_REAL = [
  {
    sequenceNumber: 0, flow: 'EP', fromType: 'N1', toType: 'N2',
    step: 'SFJ-Knee', legSide: 'right',
    posXRatio: 0.275, posYRatio: 0.05,
    confidence: 0.93, reflux_duration: 0.0,
    description: 'EP N1→N2 at SFJ — saphenofemoral junction entry confirmed',
    clipPath: 'SFJ-Knee_00_EP_N1-N2.mp4', eliminationTest: '',
  },
  {
    sequenceNumber: 1, flow: 'EP', fromType: 'N2', toType: 'N3',
    step: 'SFJ-Knee', legSide: 'right',
    posXRatio: 0.219, posYRatio: 0.132,
    confidence: 0.90, reflux_duration: 0.0,
    description: 'EP N2→N3 — medial tributary entry along thigh (Type 1+2 dual EP)',
    clipPath: 'SFJ-Knee_01_EP_N2-N3.mp4', eliminationTest: '',
  },
  {
    sequenceNumber: 2, flow: 'RP', fromType: 'N3', toType: 'N1',
    step: 'SFJ-Knee', legSide: 'right',
    posXRatio: 0.164, posYRatio: 0.212,
    confidence: 0.86, reflux_duration: 1.3,
    description: 'RP N3→N1 — tributary reflux persists after compression (Type 1+2)',
    clipPath: 'SFJ-Knee_02_RP_N3-N1.mp4', eliminationTest: 'Reflux',
  },
  {
    sequenceNumber: 3, flow: 'RP', fromType: 'N2', toType: 'N1',
    step: 'SFJ-Knee', legSide: 'right',
    posXRatio: 0.204, posYRatio: 0.289,
    confidence: 0.83, reflux_duration: 1.0,
    description: 'RP N2→N1 — dual shunt confirmed, Hunterian canal level (Type 1+2)',
    clipPath: 'SFJ-Knee_03_RP_N2-N1.mp4', eliminationTest: '',
  },
];

// ── Synthetic streams 4–10 ────────────────────────────────────────────────────
// Each is a complete realistic multi-clip session with varied shunt patterns.

const STREAM_4_SYNTH = [
  // Left leg, Type 1 pattern (SFJ incompetence → GSV reflux)
  { sequenceNumber: 0,  flow: 'EP', fromType: 'N1', toType: 'N1', step: 'SFJ', legSide: 'left', posXRatio: 0.71, posYRatio: 0.06, confidence: 0.94, reflux_duration: 0.0, description: 'Baseline deep vein — normal forward flow at left groin', clipPath: 'frame-001.png', eliminationTest: '', groundTruth: 'Type 1' },
  { sequenceNumber: 1,  flow: 'EP', fromType: 'N1', toType: 'N2', step: 'SFJ', legSide: 'left', posXRatio: 0.68, posYRatio: 0.07, confidence: 0.92, reflux_duration: 0.0, description: 'SFJ entry EP N1→N2 — left saphenofemoral junction', clipPath: 'frame-002.png', eliminationTest: '', groundTruth: 'Type 1' },
  { sequenceNumber: 2,  flow: 'RP', fromType: 'N2', toType: 'N1', step: 'SFJ-Knee', legSide: 'left', posXRatio: 0.65, posYRatio: 0.22, confidence: 0.88, reflux_duration: 0.9, description: 'RP N2→N1 proximal GSV reflux — left Hunterian region', clipPath: 'frame-003.png', eliminationTest: '', groundTruth: 'Type 1' },
  { sequenceNumber: 3,  flow: 'RP', fromType: 'N2', toType: 'N1', step: 'SFJ-Knee', legSide: 'left', posXRatio: 0.63, posYRatio: 0.35, confidence: 0.85, reflux_duration: 0.7, description: 'RP N2→N1 mid-GSV reflux — medial thigh continuation', clipPath: 'frame-004.png', eliminationTest: '', groundTruth: 'Type 1' },
  { sequenceNumber: 4,  flow: 'EP', fromType: 'N2', toType: 'N2', step: 'Knee', legSide: 'left', posXRatio: 0.72, posYRatio: 0.49, confidence: 0.91, reflux_duration: 0.0, description: 'Normal GSV flow at knee — competent below junction', clipPath: 'frame-005.png', eliminationTest: '', groundTruth: 'Type 1' },
  { sequenceNumber: 5,  flow: 'EP', fromType: 'N1', toType: 'N1', step: 'Ankle', legSide: 'left', posXRatio: 0.79, posYRatio: 0.92, confidence: 0.93, reflux_duration: 0.0, description: 'Normal deep system at left ankle', clipPath: 'frame-006.png', eliminationTest: '', groundTruth: 'Type 1' },
  { sequenceNumber: 6,  flow: 'EP', fromType: 'N1', toType: 'N1', step: 'Knee-Ankle', legSide: 'left', posXRatio: 0.82, posYRatio: 0.71, confidence: 0.92, reflux_duration: 0.0, description: 'Normal calf deep system — no distal extension of reflux', clipPath: 'frame-007.png', eliminationTest: '', groundTruth: 'Type 1' },
];

const STREAM_5_SYNTH = [
  // Right leg, Type 2A isolated tributary pattern
  { sequenceNumber: 0,  flow: 'EP', fromType: 'N1', toType: 'N1', step: 'SFJ', legSide: 'right', posXRatio: 0.28, posYRatio: 0.06, confidence: 0.95, reflux_duration: 0.0, description: 'Normal deep vein at right SFJ — no groin incompetence', clipPath: 'frame-001.png', eliminationTest: '', groundTruth: 'Type 2A' },
  { sequenceNumber: 1,  flow: 'EP', fromType: 'N2', toType: 'N3', step: 'SFJ-Knee', legSide: 'right', posXRatio: 0.22, posYRatio: 0.18, confidence: 0.90, reflux_duration: 0.0, description: 'EP N2→N3 — isolated tributary entry, no SFJ incompetence', clipPath: 'frame-002.png', eliminationTest: '', groundTruth: 'Type 2A' },
  { sequenceNumber: 2,  flow: 'EP', fromType: 'N2', toType: 'N3', step: 'SFJ-Knee', legSide: 'right', posXRatio: 0.21, posYRatio: 0.30, confidence: 0.88, reflux_duration: 0.0, description: 'Second N2→N3 branch along medial thigh', clipPath: 'frame-003.png', eliminationTest: '', groundTruth: 'Type 2A' },
  { sequenceNumber: 3,  flow: 'RP', fromType: 'N3', toType: 'N2', step: 'Knee', legSide: 'right', posXRatio: 0.19, posYRatio: 0.48, confidence: 0.86, reflux_duration: 0.8, description: 'RP N3→N2 — Type 2A pattern at knee tributary junction', clipPath: 'frame-004.png', eliminationTest: '', groundTruth: 'Type 2A' },
  { sequenceNumber: 4,  flow: 'RP', fromType: 'N3', toType: 'N2', step: 'Knee-Ankle', legSide: 'right', posXRatio: 0.17, posYRatio: 0.65, confidence: 0.83, reflux_duration: 0.6, description: 'RP N3→N2 distal tributary — Cockett zone perforator', clipPath: 'frame-005.png', eliminationTest: '', groundTruth: 'Type 2A' },
  { sequenceNumber: 5,  flow: 'EP', fromType: 'N2', toType: 'N2', step: 'SFJ-Knee', legSide: 'right', posXRatio: 0.24, posYRatio: 0.14, confidence: 0.92, reflux_duration: 0.0, description: 'GSV competent at groin — no SFJ reflux confirmed', clipPath: 'frame-006.png', eliminationTest: '', groundTruth: 'Type 2A' },
  { sequenceNumber: 6,  flow: 'EP', fromType: 'N1', toType: 'N1', step: 'Ankle', legSide: 'right', posXRatio: 0.15, posYRatio: 0.91, confidence: 0.94, reflux_duration: 0.0, description: 'Normal deep system at ankle — no distal extension', clipPath: 'frame-007.png', eliminationTest: '', groundTruth: 'Type 2A' },
];

const STREAM_6_SYNTH = [
  // Right leg, Type 3 tributary-to-tributary reflux
  { sequenceNumber: 0,  flow: 'EP', fromType: 'N1', toType: 'N2', step: 'SFJ', legSide: 'right', posXRatio: 0.29, posYRatio: 0.05, confidence: 0.93, reflux_duration: 0.0, description: 'SFJ entry EP N1→N2 at groin level', clipPath: 'frame-001.png', eliminationTest: '', groundTruth: 'Type 3' },
  { sequenceNumber: 1,  flow: 'EP', fromType: 'N2', toType: 'N3', step: 'SFJ-Knee', legSide: 'right', posXRatio: 0.22, posYRatio: 0.20, confidence: 0.90, reflux_duration: 0.0, description: 'EP N2→N3 tributary entry — medial thigh', clipPath: 'frame-002.png', eliminationTest: '', groundTruth: 'Type 3' },
  { sequenceNumber: 2,  flow: 'RP', fromType: 'N3', toType: 'N2', step: 'Knee', legSide: 'right', posXRatio: 0.18, posYRatio: 0.47, confidence: 0.85, reflux_duration: 1.1, description: 'RP N3→N2 reflux at knee level — Type 3 pattern', clipPath: 'frame-003.png', eliminationTest: '', groundTruth: 'Type 3' },
  { sequenceNumber: 3,  flow: 'RP', fromType: 'N3', toType: 'N1', step: 'Knee-Ankle', legSide: 'right', posXRatio: 0.16, posYRatio: 0.63, confidence: 0.83, reflux_duration: 0.9, description: 'RP N3→N1 with elimination test — calf perforator zone', clipPath: 'frame-004.png', eliminationTest: 'No Reflux', groundTruth: 'Type 3' },
  { sequenceNumber: 4,  flow: 'RP', fromType: 'N2', toType: 'N1', step: 'SFJ-Knee', legSide: 'right', posXRatio: 0.21, posYRatio: 0.33, confidence: 0.82, reflux_duration: 1.0, description: 'RP N2→N1 proximal GSV after tributary loop', clipPath: 'frame-005.png', eliminationTest: '', groundTruth: 'Type 3' },
  { sequenceNumber: 5,  flow: 'EP', fromType: 'N3', toType: 'N3', step: 'Knee-Ankle', legSide: 'right', posXRatio: 0.17, posYRatio: 0.74, confidence: 0.91, reflux_duration: 0.0, description: 'Normal distal tributary — no calf extension', clipPath: 'frame-006.png', eliminationTest: '', groundTruth: 'Type 3' },
  { sequenceNumber: 6,  flow: 'EP', fromType: 'N1', toType: 'N1', step: 'Ankle', legSide: 'right', posXRatio: 0.14, posYRatio: 0.92, confidence: 0.95, reflux_duration: 0.0, description: 'Normal deep system at ankle', clipPath: 'frame-007.png', eliminationTest: '', groundTruth: 'Type 3' },
];

const STREAM_7_SYNTH = [
  // Left leg, Type 2B (SPJ-based tributary reflux)
  { sequenceNumber: 0,  flow: 'EP', fromType: 'N1', toType: 'N1', step: 'SPJ', legSide: 'left', posXRatio: 0.63, posYRatio: 0.61, confidence: 0.93, reflux_duration: 0.0, description: 'Normal deep vein at left SPJ — no popliteal incompetence yet', clipPath: 'frame-001.png', eliminationTest: '', groundTruth: 'Type 2B' },
  { sequenceNumber: 1,  flow: 'EP', fromType: 'N2', toType: 'N3', step: 'SPJ-Ankle', legSide: 'left', posXRatio: 0.61, posYRatio: 0.70, confidence: 0.89, reflux_duration: 0.0, description: 'EP N2→N3 — SSV tributary entry at posterior calf', clipPath: 'frame-002.png', eliminationTest: '', groundTruth: 'Type 2B' },
  { sequenceNumber: 2,  flow: 'RP', fromType: 'N3', toType: 'N2', step: 'SPJ-Ankle', legSide: 'left', posXRatio: 0.60, posYRatio: 0.72, confidence: 0.87, reflux_duration: 1.2, description: 'RP N3→N2 — tributary reflux at SPJ level, Type 2B', clipPath: 'frame-003.png', eliminationTest: '', groundTruth: 'Type 2B' },
  { sequenceNumber: 3,  flow: 'RP', fromType: 'N3', toType: 'N1', step: 'Knee-Ankle', legSide: 'left', posXRatio: 0.83, posYRatio: 0.68, confidence: 0.84, reflux_duration: 0.8, description: 'RP N3→N1 distal calf tributary to deep — Cockett zone', clipPath: 'frame-004.png', eliminationTest: '', groundTruth: 'Type 2B' },
  { sequenceNumber: 4,  flow: 'EP', fromType: 'N1', toType: 'N1', step: 'SFJ', legSide: 'left', posXRatio: 0.72, posYRatio: 0.06, confidence: 0.95, reflux_duration: 0.0, description: 'Normal deep vein at left groin — SFJ competent', clipPath: 'frame-005.png', eliminationTest: '', groundTruth: 'Type 2B' },
  { sequenceNumber: 5,  flow: 'EP', fromType: 'N2', toType: 'N2', step: 'SFJ-Knee', legSide: 'left', posXRatio: 0.67, posYRatio: 0.28, confidence: 0.91, reflux_duration: 0.0, description: 'GSV competent along thigh — isolated SPJ incompetence confirmed', clipPath: 'frame-006.png', eliminationTest: '', groundTruth: 'Type 2B' },
  { sequenceNumber: 6,  flow: 'EP', fromType: 'N1', toType: 'N1', step: 'Ankle', legSide: 'left', posXRatio: 0.80, posYRatio: 0.93, confidence: 0.94, reflux_duration: 0.0, description: 'Normal deep system at left ankle', clipPath: 'frame-007.png', eliminationTest: '', groundTruth: 'Type 2B' },
];

const STREAM_8_SYNTH = [
  // Both legs, Type 1 right + Type 2A left (bilateral assessment)
  { sequenceNumber: 0,  flow: 'EP', fromType: 'N1', toType: 'N2', step: 'SFJ', legSide: 'right', posXRatio: 0.29, posYRatio: 0.06, confidence: 0.94, reflux_duration: 0.0, description: 'Right SFJ entry EP N1→N2', clipPath: 'frame-001.png', eliminationTest: '', groundTruth: 'Type 1 (R) + Type 2A (L)' },
  { sequenceNumber: 1,  flow: 'RP', fromType: 'N2', toType: 'N1', step: 'SFJ-Knee', legSide: 'right', posXRatio: 0.24, posYRatio: 0.25, confidence: 0.87, reflux_duration: 1.0, description: 'RP N2→N1 right GSV reflux — Type 1 Hunterian level', clipPath: 'frame-002.png', eliminationTest: '', groundTruth: 'Type 1 (R) + Type 2A (L)' },
  { sequenceNumber: 2,  flow: 'EP', fromType: 'N2', toType: 'N2', step: 'Knee', legSide: 'right', posXRatio: 0.20, posYRatio: 0.47, confidence: 0.91, reflux_duration: 0.0, description: 'Normal right GSV at knee — below junction competent', clipPath: 'frame-003.png', eliminationTest: '', groundTruth: 'Type 1 (R) + Type 2A (L)' },
  { sequenceNumber: 3,  flow: 'EP', fromType: 'N1', toType: 'N1', step: 'SFJ', legSide: 'left', posXRatio: 0.69, posYRatio: 0.06, confidence: 0.95, reflux_duration: 0.0, description: 'Left SFJ deep vein — competent', clipPath: 'frame-004.png', eliminationTest: '', groundTruth: 'Type 1 (R) + Type 2A (L)' },
  { sequenceNumber: 4,  flow: 'EP', fromType: 'N2', toType: 'N3', step: 'SFJ-Knee', legSide: 'left', posXRatio: 0.64, posYRatio: 0.21, confidence: 0.89, reflux_duration: 0.0, description: 'EP N2→N3 left tributary entry — no SFJ incompetence', clipPath: 'frame-005.png', eliminationTest: '', groundTruth: 'Type 1 (R) + Type 2A (L)' },
  { sequenceNumber: 5,  flow: 'RP', fromType: 'N3', toType: 'N2', step: 'Knee', legSide: 'left', posXRatio: 0.78, posYRatio: 0.48, confidence: 0.85, reflux_duration: 0.8, description: 'RP N3→N2 left tributary reflux at knee — Type 2A', clipPath: 'frame-006.png', eliminationTest: '', groundTruth: 'Type 1 (R) + Type 2A (L)' },
  { sequenceNumber: 6,  flow: 'EP', fromType: 'N1', toType: 'N1', step: 'Ankle', legSide: 'left', posXRatio: 0.81, posYRatio: 0.92, confidence: 0.94, reflux_duration: 0.0, description: 'Normal left deep system at ankle', clipPath: 'frame-007.png', eliminationTest: '', groundTruth: 'Type 1 (R) + Type 2A (L)' },
  { sequenceNumber: 7,  flow: 'EP', fromType: 'N1', toType: 'N1', step: 'Ankle', legSide: 'right', posXRatio: 0.15, posYRatio: 0.93, confidence: 0.95, reflux_duration: 0.0, description: 'Normal right deep system at ankle', clipPath: 'frame-008.png', eliminationTest: '', groundTruth: 'Type 1 (R) + Type 2A (L)' },
];

const STREAM_9_SYNTH = [
  // Right leg, No shunt (all normal EP — negative study)
  { sequenceNumber: 0,  flow: 'EP', fromType: 'N1', toType: 'N1', step: 'SFJ', legSide: 'right', posXRatio: 0.30, posYRatio: 0.06, confidence: 0.96, reflux_duration: 0.0, description: 'Normal deep vein at right SFJ — competent', clipPath: 'frame-001.png', eliminationTest: '', groundTruth: 'No Shunt (Normal)' },
  { sequenceNumber: 1,  flow: 'EP', fromType: 'N1', toType: 'N2', step: 'SFJ', legSide: 'right', posXRatio: 0.28, posYRatio: 0.07, confidence: 0.94, reflux_duration: 0.0, description: 'Normal SFJ entry EP N1→N2 — no incompetence', clipPath: 'frame-002.png', eliminationTest: '', groundTruth: 'No Shunt (Normal)' },
  { sequenceNumber: 2,  flow: 'EP', fromType: 'N2', toType: 'N2', step: 'SFJ-Knee', legSide: 'right', posXRatio: 0.23, posYRatio: 0.22, confidence: 0.93, reflux_duration: 0.0, description: 'Normal GSV flow along thigh', clipPath: 'frame-003.png', eliminationTest: '', groundTruth: 'No Shunt (Normal)' },
  { sequenceNumber: 3,  flow: 'EP', fromType: 'N2', toType: 'N2', step: 'Knee', legSide: 'right', posXRatio: 0.20, posYRatio: 0.47, confidence: 0.93, reflux_duration: 0.0, description: 'Normal GSV at knee — fully competent', clipPath: 'frame-004.png', eliminationTest: '', groundTruth: 'No Shunt (Normal)' },
  { sequenceNumber: 4,  flow: 'EP', fromType: 'N1', toType: 'N1', step: 'Knee-Ankle', legSide: 'right', posXRatio: 0.17, posYRatio: 0.67, confidence: 0.95, reflux_duration: 0.0, description: 'Normal deep system mid-calf', clipPath: 'frame-005.png', eliminationTest: '', groundTruth: 'No Shunt (Normal)' },
  { sequenceNumber: 5,  flow: 'EP', fromType: 'N1', toType: 'N1', step: 'SPJ', legSide: 'right', posXRatio: 0.33, posYRatio: 0.62, confidence: 0.94, reflux_duration: 0.0, description: 'Normal popliteal vein at SPJ — no reflux', clipPath: 'frame-006.png', eliminationTest: '', groundTruth: 'No Shunt (Normal)' },
  { sequenceNumber: 6,  flow: 'EP', fromType: 'N1', toType: 'N1', step: 'Ankle', legSide: 'right', posXRatio: 0.16, posYRatio: 0.92, confidence: 0.96, reflux_duration: 0.0, description: 'Normal deep system at ankle — negative study', clipPath: 'frame-007.png', eliminationTest: '', groundTruth: 'No Shunt (Normal)' },
];

const STREAM_10_SYNTH = [
  // Left leg, Type 1+2 complex bilateral entry pattern
  { sequenceNumber: 0,  flow: 'EP', fromType: 'N1', toType: 'N2', step: 'SFJ', legSide: 'left', posXRatio: 0.70, posYRatio: 0.06, confidence: 0.93, reflux_duration: 0.0, description: 'SFJ entry EP N1→N2 left — groin level incompetence', clipPath: 'frame-001.png', eliminationTest: '', groundTruth: 'Type 1+2' },
  { sequenceNumber: 1,  flow: 'EP', fromType: 'N2', toType: 'N3', step: 'SFJ-Knee', legSide: 'left', posXRatio: 0.64, posYRatio: 0.20, confidence: 0.90, reflux_duration: 0.0, description: 'EP N2→N3 double entry along medial left thigh', clipPath: 'frame-002.png', eliminationTest: '', groundTruth: 'Type 1+2' },
  { sequenceNumber: 2,  flow: 'RP', fromType: 'N3', toType: 'N2', step: 'Knee', legSide: 'left', posXRatio: 0.78, posYRatio: 0.48, confidence: 0.86, reflux_duration: 1.3, description: 'RP N3→N2 — reflux persists after compression, knee level', clipPath: 'frame-003.png', eliminationTest: 'Reflux', groundTruth: 'Type 1+2' },
  { sequenceNumber: 3,  flow: 'RP', fromType: 'N2', toType: 'N1', step: 'SFJ-Knee', legSide: 'left', posXRatio: 0.66, posYRatio: 0.31, confidence: 0.84, reflux_duration: 1.1, description: 'RP N2→N1 dual shunt confirmed — Hunterian level left', clipPath: 'frame-004.png', eliminationTest: '', groundTruth: 'Type 1+2' },
  { sequenceNumber: 4,  flow: 'EP', fromType: 'N3', toType: 'N3', step: 'Knee-Ankle', legSide: 'left', posXRatio: 0.83, posYRatio: 0.72, confidence: 0.91, reflux_duration: 0.0, description: 'Normal distal tributary — calf segment unaffected', clipPath: 'frame-005.png', eliminationTest: '', groundTruth: 'Type 1+2' },
  { sequenceNumber: 5,  flow: 'EP', fromType: 'N1', toType: 'N1', step: 'SPJ', legSide: 'left', posXRatio: 0.62, posYRatio: 0.63, confidence: 0.93, reflux_duration: 0.0, description: 'Normal popliteal vein at left SPJ', clipPath: 'frame-006.png', eliminationTest: '', groundTruth: 'Type 1+2' },
  { sequenceNumber: 6,  flow: 'EP', fromType: 'N1', toType: 'N1', step: 'Ankle', legSide: 'left', posXRatio: 0.81, posYRatio: 0.93, confidence: 0.95, reflux_duration: 0.0, description: 'Normal deep system at left ankle', clipPath: 'frame-007.png', eliminationTest: '', groundTruth: 'Type 1+2' },
];

// ── Master stream list ────────────────────────────────────────────────────────
export const ACCURACY_TEST_STREAMS = [
  {
    id: 'stream1',
    name: 'Stream 1',
    source: 'real',
    groundTruth: 'Type 1',
    description: 'Real session — single Type 1 shunt (sparse recording, 1 EP clip)',
    clips: STREAM_1_REAL,
  },
  {
    id: 'stream2',
    name: 'Stream 2',
    source: 'real',
    groundTruth: 'Type 2C',
    description: 'Real session — Type 2B progressing to Type 2C (3 clips: 1 EP + 2 RP)',
    clips: STREAM_2_REAL,
  },
  {
    id: 'stream3',
    name: 'Stream 3',
    source: 'real',
    groundTruth: 'Type 1+2',
    description: 'Real session — Type 3 variants leading to Type 1+2 (4 clips: 2 EP + 2 RP)',
    clips: STREAM_3_REAL,
  },
  {
    id: 'stream4',
    name: 'Stream 4',
    source: 'synthetic',
    groundTruth: 'Type 1',
    description: 'Synthetic — Left leg Type 1, GSV SFJ incompetence (7 clips)',
    clips: STREAM_4_SYNTH,
  },
  {
    id: 'stream5',
    name: 'Stream 5',
    source: 'synthetic',
    groundTruth: 'Type 2A',
    description: 'Synthetic — Right leg isolated Type 2A tributary reflux (7 clips)',
    clips: STREAM_5_SYNTH,
  },
  {
    id: 'stream6',
    name: 'Stream 6',
    source: 'synthetic',
    groundTruth: 'Type 3',
    description: 'Synthetic — Right leg Type 3 tributary-to-tributary reflux (7 clips)',
    clips: STREAM_6_SYNTH,
  },
  {
    id: 'stream7',
    name: 'Stream 7',
    source: 'synthetic',
    groundTruth: 'Type 2B',
    description: 'Synthetic — Left leg SPJ-based Type 2B pattern (7 clips)',
    clips: STREAM_7_SYNTH,
  },
  {
    id: 'stream8',
    name: 'Stream 8',
    source: 'synthetic',
    groundTruth: 'Type 1 (R) + Type 2A (L)',
    description: 'Synthetic — Bilateral: Type 1 right leg + Type 2A left leg (8 clips)',
    clips: STREAM_8_SYNTH,
  },
  {
    id: 'stream9',
    name: 'Stream 9',
    source: 'synthetic',
    groundTruth: 'No Shunt (Normal)',
    description: 'Synthetic — Negative study, all normal EP flow (7 clips)',
    clips: STREAM_9_SYNTH,
  },
  {
    id: 'stream10',
    name: 'Stream 10',
    source: 'synthetic',
    groundTruth: 'Type 1+2',
    description: 'Synthetic — Left leg Type 1+2 complex dual-entry pattern (7 clips)',
    clips: STREAM_10_SYNTH,
  },
];
