import React, { useState } from 'react';
import axios from 'axios';
import { ACCURACY_TEST_STREAMS } from '../data/accuracyTestStreams';

// ── Dynamic mock clip generator ──────────────────────────────────────────────
// Generates 15–20 clips with exactly 2 or 3 shunt patterns on random leg sides.
// Called once per page load so every refresh produces a fresh assessment.

const rnd = (min, max, dp = 2) => parseFloat((Math.random() * (max - min) + min).toFixed(dp));

// Anatomically grounded posX/posY ranges per leg side and scanning zone.
// Coordinate system: (0,0) = top-left corner, (1,1) = bottom-right corner.
// Right leg occupies the LEFT side of the image (lower X values).
// Left  leg occupies the RIGHT side of the image (higher X values).
const POS = {
  right: {
    SFJ:         { x: [0.0931, 0.475],  y: [0.03,   0.098]  },
    'SFJ-Knee':  { x: [0.0931, 0.475],  y: [0.10,   0.50]   },
    Knee:        { x: [0.0931, 0.475],  y: [0.45,   0.5497] },
    'Knee-Ankle':{ x: [0.105,  0.2947], y: [0.5497, 0.90]   },
    'SPJ-Ankle': { x: [0.2827, 0.4386], y: [0.5497, 0.80]   },
    SPJ:         { x: [0.2827, 0.4386], y: [0.55,   0.72]   },
    Ankle:       { x: [0.105,  0.2947], y: [0.86,   0.98]   },
  },
  left: {
    SFJ:         { x: [0.4985, 0.909],  y: [0.03,   0.098]  },
    'SFJ-Knee':  { x: [0.4985, 0.909],  y: [0.10,   0.50]   },
    Knee:        { x: [0.4985, 0.909],  y: [0.45,   0.5497] },
    'Knee-Ankle':{ x: [0.7081, 0.91],   y: [0.5497, 0.90]   },
    'SPJ-Ankle': { x: [0.588,  0.714],  y: [0.5497, 0.80]   },
    SPJ:         { x: [0.588,  0.714],  y: [0.55,   0.72]   },
    Ankle:       { x: [0.7081, 0.91],   y: [0.86,   0.98]   },
  },
};
const px = (leg, step) => { const r = (POS[leg] || POS.right)[step] || POS.right['SFJ-Knee']; return rnd(r.x[0], r.x[1]); };
const py = (leg, step) => { const r = (POS[leg] || POS.right)[step] || POS.right['SFJ-Knee']; return rnd(r.y[0], r.y[1]); };

// Each template returns an array of clips for one shunt on one leg side.
const SHUNT_TEMPLATES = {
  type1: (leg, seqStart) => {
    return [
      { sequenceNumber: seqStart,     flow: 'EP', fromType: 'N1', toType: 'N1', posXRatio: px(leg,'SFJ'),       posYRatio: py(leg,'SFJ'),       step: 'SFJ',       legSide: leg, confidence: rnd(0.90, 0.97), reflux_duration: 0.0,           description: 'Baseline deep vein — normal forward flow at groin', clipPath: `frame-${String(seqStart).padStart(3,'0')}.png`, eliminationTest: '' },
      { sequenceNumber: seqStart + 1, flow: 'EP', fromType: 'N1', toType: 'N2', posXRatio: px(leg,'SFJ'),       posYRatio: py(leg,'SFJ'),       step: 'SFJ',       legSide: leg, confidence: rnd(0.90, 0.96), reflux_duration: 0.0,           description: 'SFJ entry point — EP N1→N2 at saphenofemoral junction', clipPath: `frame-${String(seqStart+1).padStart(3,'0')}.png`, eliminationTest: '' },
      { sequenceNumber: seqStart + 2, flow: 'RP', fromType: 'N2', toType: 'N1', posXRatio: px(leg,'SFJ-Knee'),  posYRatio: rnd(...(POS[leg]||POS.right)['SFJ-Knee'].y.map((v,i)=>i===0?v:v*0.45)), step: 'SFJ-Knee', legSide: leg, confidence: rnd(0.85, 0.93), reflux_duration: rnd(0.6, 1.4), description: 'RP N2→N1 proximal GSV reflux — Hunterian region', clipPath: `frame-${String(seqStart+2).padStart(3,'0')}.png`, eliminationTest: '',
        ligation: { procedure_name: 'SFJ Ligation (Crossectomy)', technique: 'Open groin incision, 4-0 Vicryl at SFJ', location: 'Saphenofemoral junction', vessels_ligated: ['GSV at SFJ', 'Superficial epigastric vein'], compression_post_op: 'Class III 40-50mmHg wk 1-2, Class II wk 3-6' } },
      { sequenceNumber: seqStart + 3, flow: 'RP', fromType: 'N2', toType: 'N1', posXRatio: px(leg,'SFJ-Knee'),  posYRatio: py(leg,'SFJ-Knee'),  step: 'SFJ-Knee',  legSide: leg, confidence: rnd(0.82, 0.90), reflux_duration: rnd(0.5, 1.2), description: 'RP N2→N1 mid-GSV reflux along medial thigh', clipPath: `frame-${String(seqStart+3).padStart(3,'0')}.png`, eliminationTest: '' },
      { sequenceNumber: seqStart + 4, flow: 'EP', fromType: 'N2', toType: 'N2', posXRatio: px(leg,'Knee'),      posYRatio: py(leg,'Knee'),      step: 'Knee',      legSide: leg, confidence: rnd(0.90, 0.96), reflux_duration: 0.0,           description: 'Normal GSV flow at knee — competent below junction', clipPath: `frame-${String(seqStart+4).padStart(3,'0')}.png`, eliminationTest: '' },
      { sequenceNumber: seqStart + 5, flow: 'EP', fromType: 'N1', toType: 'N1', posXRatio: px(leg,'Ankle'),     posYRatio: py(leg,'Ankle'),     step: 'Ankle',     legSide: leg, confidence: rnd(0.91, 0.97), reflux_duration: 0.0,           description: 'Normal deep system at ankle — no distal extension', clipPath: `frame-${String(seqStart+5).padStart(3,'0')}.png`, eliminationTest: '' },
    ];
  },
  type3: (leg, seqStart) => {
    return [
      { sequenceNumber: seqStart,     flow: 'EP', fromType: 'N1', toType: 'N2', posXRatio: px(leg,'SFJ'),       posYRatio: py(leg,'SFJ'),       step: 'SFJ',       legSide: leg, confidence: rnd(0.90, 0.96), reflux_duration: 0.0,           description: 'SFJ entry — EP N1→N2 at groin level', clipPath: `frame-${String(seqStart).padStart(3,'0')}.png`, eliminationTest: '' },
      { sequenceNumber: seqStart + 1, flow: 'EP', fromType: 'N2', toType: 'N3', posXRatio: px(leg,'SFJ-Knee'),  posYRatio: py(leg,'SFJ-Knee'),  step: 'SFJ-Knee',  legSide: leg, confidence: rnd(0.89, 0.95), reflux_duration: 0.0,           description: 'EP N2→N3 — tributary entry along medial thigh', clipPath: `frame-${String(seqStart+1).padStart(3,'0')}.png`, eliminationTest: '' },
      { sequenceNumber: seqStart + 2, flow: 'RP', fromType: 'N3', toType: 'N2', posXRatio: px(leg,'Knee'),      posYRatio: py(leg,'Knee'),      step: 'Knee',      legSide: leg, confidence: rnd(0.84, 0.92), reflux_duration: rnd(0.8, 1.6), description: 'RP N3→N2 tributary reflux at knee level', clipPath: `frame-${String(seqStart+2).padStart(3,'0')}.png`, eliminationTest: '',
        ligation: { procedure_name: 'Tributary Ligation at N2→N3', technique: 'Small 2cm incision, 3-0 Vicryl at tributary junction', location: 'Medial thigh tributary', vessels_ligated: ['N3 tributary at N2 junction'], compression_post_op: 'Class III wk 1-2, Class II wk 3-8' } },
      { sequenceNumber: seqStart + 3, flow: 'RP', fromType: 'N3', toType: 'N1', posXRatio: px(leg,'Knee-Ankle'), posYRatio: py(leg,'Knee-Ankle'), step: 'Knee-Ankle', legSide: leg, confidence: rnd(0.82, 0.90), reflux_duration: rnd(0.7, 1.3), description: 'RP N3→N1 with elimination test — calf perforator zone', clipPath: `frame-${String(seqStart+3).padStart(3,'0')}.png`, eliminationTest: 'No Reflux' },
      { sequenceNumber: seqStart + 4, flow: 'RP', fromType: 'N2', toType: 'N1', posXRatio: px(leg,'SFJ-Knee'),  posYRatio: py(leg,'SFJ-Knee'),  step: 'SFJ-Knee',  legSide: leg, confidence: rnd(0.81, 0.89), reflux_duration: rnd(0.9, 1.5), description: 'RP N2→N1 proximal GSV after tributary loop — Hunterian level', clipPath: `frame-${String(seqStart+4).padStart(3,'0')}.png`, eliminationTest: '' },
      { sequenceNumber: seqStart + 5, flow: 'EP', fromType: 'N3', toType: 'N3', posXRatio: px(leg,'Knee-Ankle'), posYRatio: py(leg,'Knee-Ankle'), step: 'Knee-Ankle', legSide: leg, confidence: rnd(0.90, 0.96), reflux_duration: 0.0,           description: 'Normal distal tributary — no calf extension of reflux', clipPath: `frame-${String(seqStart+5).padStart(3,'0')}.png`, eliminationTest: '' },
    ];
  },
  type2a: (leg, seqStart) => {
    return [
      { sequenceNumber: seqStart,     flow: 'EP', fromType: 'N2', toType: 'N3', posXRatio: px(leg,'SFJ-Knee'),  posYRatio: py(leg,'SFJ-Knee'),  step: 'SFJ-Knee',  legSide: leg, confidence: rnd(0.88, 0.95), reflux_duration: 0.0,           description: 'EP N2→N3 — no SFJ incompetence, isolated tributary entry', clipPath: `frame-${String(seqStart).padStart(3,'0')}.png`, eliminationTest: '' },
      { sequenceNumber: seqStart + 1, flow: 'EP', fromType: 'N2', toType: 'N3', posXRatio: px(leg,'SFJ-Knee'),  posYRatio: py(leg,'SFJ-Knee'),  step: 'SFJ-Knee',  legSide: leg, confidence: rnd(0.87, 0.94), reflux_duration: 0.0,           description: 'Second N2→N3 branch entry along medial thigh', clipPath: `frame-${String(seqStart+1).padStart(3,'0')}.png`, eliminationTest: '' },
      { sequenceNumber: seqStart + 2, flow: 'RP', fromType: 'N3', toType: 'N2', posXRatio: px(leg,'Knee'),      posYRatio: py(leg,'Knee'),      step: 'Knee',      legSide: leg, confidence: rnd(0.83, 0.91), reflux_duration: rnd(0.7, 1.3), description: 'RP N3→N2 — Type 2A pattern at knee junction', clipPath: `frame-${String(seqStart+2).padStart(3,'0')}.png`, eliminationTest: '',
        ligation: { procedure_name: 'Ligate highest EP at N2→N3', technique: 'Small incision at highest EP entry, 3-0 absorbable sutures', location: 'Medial knee tributary entry', vessels_ligated: ['Highest N2→N3 entry point'], compression_post_op: 'Class II 23-32mmHg wk 1-4' } },
      { sequenceNumber: seqStart + 3, flow: 'RP', fromType: 'N3', toType: 'N2', posXRatio: px(leg,'Knee-Ankle'), posYRatio: py(leg,'Knee-Ankle'), step: 'Knee-Ankle', legSide: leg, confidence: rnd(0.81, 0.89), reflux_duration: rnd(0.5, 1.1), description: 'RP N3→N2 distal tributary — Cockett zone', clipPath: `frame-${String(seqStart+3).padStart(3,'0')}.png`, eliminationTest: '' },
      { sequenceNumber: seqStart + 4, flow: 'EP', fromType: 'N2', toType: 'N2', posXRatio: px(leg,'SFJ-Knee'),  posYRatio: py(leg,'SFJ-Knee'),  step: 'SFJ-Knee',  legSide: leg, confidence: rnd(0.90, 0.96), reflux_duration: 0.0,           description: 'GSV competent — no SFJ reflux confirmed', clipPath: `frame-${String(seqStart+4).padStart(3,'0')}.png`, eliminationTest: '' },
      { sequenceNumber: seqStart + 5, flow: 'EP', fromType: 'N1', toType: 'N1', posXRatio: px(leg,'Ankle'),     posYRatio: py(leg,'Ankle'),     step: 'Ankle',     legSide: leg, confidence: rnd(0.91, 0.97), reflux_duration: 0.0,           description: 'Normal deep system at ankle', clipPath: `frame-${String(seqStart+5).padStart(3,'0')}.png`, eliminationTest: '' },
    ];
  },
  type2c: (leg, seqStart) => {
    return [
      { sequenceNumber: seqStart,     flow: 'EP', fromType: 'N2', toType: 'N3', posXRatio: px(leg,'SFJ-Knee'),  posYRatio: py(leg,'SFJ-Knee'),  step: 'SFJ-Knee',  legSide: leg, confidence: rnd(0.88, 0.94), reflux_duration: 0.0,           description: 'EP N2→N3 entry — Type 2C pattern, medial thigh', clipPath: `frame-${String(seqStart).padStart(3,'0')}.png`, eliminationTest: '' },
      { sequenceNumber: seqStart + 1, flow: 'RP', fromType: 'N3', toType: 'N2', posXRatio: px(leg,'Knee'),      posYRatio: py(leg,'Knee'),      step: 'Knee',      legSide: leg, confidence: rnd(0.83, 0.91), reflux_duration: rnd(0.8, 1.4), description: 'RP N3→N2 tributary reflux at knee', clipPath: `frame-${String(seqStart+1).padStart(3,'0')}.png`, eliminationTest: '',
        ligation: { procedure_name: 'Ligate N2→N3 entry + SFJ', technique: 'Combined groin incision and tributary ligation', location: 'SFJ and medial knee', vessels_ligated: ['GSV at SFJ', 'N2→N3 entry point'], compression_post_op: 'Class III 40-50mmHg wk 1-2, Class II wk 3-8' } },
      { sequenceNumber: seqStart + 2, flow: 'RP', fromType: 'N2', toType: 'N1', posXRatio: px(leg,'SFJ-Knee'),  posYRatio: py(leg,'SFJ-Knee'),  step: 'SFJ-Knee',  legSide: leg, confidence: rnd(0.82, 0.90), reflux_duration: rnd(0.7, 1.3), description: 'RP N2→N1 — additional GSV reflux (Type 2C) Hunterian level', clipPath: `frame-${String(seqStart+2).padStart(3,'0')}.png`, eliminationTest: '' },
      { sequenceNumber: seqStart + 3, flow: 'EP', fromType: 'N2', toType: 'N2', posXRatio: px(leg,'Knee-Ankle'), posYRatio: py(leg,'Knee-Ankle'), step: 'Knee-Ankle', legSide: leg, confidence: rnd(0.89, 0.95), reflux_duration: 0.0,           description: 'Normal below-knee GSV — calf segment competent', clipPath: `frame-${String(seqStart+3).padStart(3,'0')}.png`, eliminationTest: '' },
      { sequenceNumber: seqStart + 4, flow: 'EP', fromType: 'N1', toType: 'N1', posXRatio: px(leg,'Ankle'),     posYRatio: py(leg,'Ankle'),     step: 'Ankle',     legSide: leg, confidence: rnd(0.92, 0.97), reflux_duration: 0.0,           description: 'Normal deep system at ankle', clipPath: `frame-${String(seqStart+4).padStart(3,'0')}.png`, eliminationTest: '' },
    ];
  },
  type1plus2: (leg, seqStart) => {
    return [
      { sequenceNumber: seqStart,     flow: 'EP', fromType: 'N1', toType: 'N2', posXRatio: px(leg,'SFJ'),       posYRatio: py(leg,'SFJ'),       step: 'SFJ',       legSide: leg, confidence: rnd(0.90, 0.96), reflux_duration: 0.0,           description: 'SFJ entry EP N1→N2 — groin level incompetence', clipPath: `frame-${String(seqStart).padStart(3,'0')}.png`, eliminationTest: '' },
      { sequenceNumber: seqStart + 1, flow: 'EP', fromType: 'N2', toType: 'N3', posXRatio: px(leg,'SFJ-Knee'),  posYRatio: py(leg,'SFJ-Knee'),  step: 'SFJ-Knee',  legSide: leg, confidence: rnd(0.88, 0.94), reflux_duration: 0.0,           description: 'EP N2→N3 — double entry pattern along medial thigh', clipPath: `frame-${String(seqStart+1).padStart(3,'0')}.png`, eliminationTest: '' },
      { sequenceNumber: seqStart + 2, flow: 'RP', fromType: 'N3', toType: 'N2', posXRatio: px(leg,'Knee'),      posYRatio: py(leg,'Knee'),      step: 'Knee',      legSide: leg, confidence: rnd(0.84, 0.92), reflux_duration: rnd(0.9, 1.6), description: 'RP N3→N2 — reflux persists after compression at knee', clipPath: `frame-${String(seqStart+2).padStart(3,'0')}.png`, eliminationTest: 'Reflux',
        ligation: { procedure_name: 'SFJ + Tributary Ligation (Type 1+2)', technique: 'Groin crossectomy + medial tributary ligation', location: 'SFJ and medial thigh tributary', vessels_ligated: ['GSV at SFJ', 'N2→N3 entry point'], compression_post_op: 'Class III 40-50mmHg wk 1-3, Class II wk 4-8' } },
      { sequenceNumber: seqStart + 3, flow: 'RP', fromType: 'N2', toType: 'N1', posXRatio: px(leg,'SFJ-Knee'),  posYRatio: py(leg,'SFJ-Knee'),  step: 'SFJ-Knee',  legSide: leg, confidence: rnd(0.82, 0.90), reflux_duration: rnd(0.8, 1.4), description: 'RP N2→N1 — dual shunt confirmed, Hunterian canal level', clipPath: `frame-${String(seqStart+3).padStart(3,'0')}.png`, eliminationTest: '' },
      { sequenceNumber: seqStart + 4, flow: 'EP', fromType: 'N3', toType: 'N3', posXRatio: px(leg,'Knee-Ankle'), posYRatio: py(leg,'Knee-Ankle'), step: 'Knee-Ankle', legSide: leg, confidence: rnd(0.90, 0.96), reflux_duration: 0.0,           description: 'Normal distal tributary — calf segment unaffected', clipPath: `frame-${String(seqStart+4).padStart(3,'0')}.png`, eliminationTest: '' },
      { sequenceNumber: seqStart + 5, flow: 'EP', fromType: 'N1', toType: 'N1', posXRatio: px(leg,'SPJ'),       posYRatio: py(leg,'SPJ'),       step: 'SPJ',       legSide: leg, confidence: rnd(0.91, 0.97), reflux_duration: 0.0,           description: 'Normal deep system at SPJ — popliteal vein competent', clipPath: `frame-${String(seqStart+5).padStart(3,'0')}.png`, eliminationTest: '' },
    ];
  },
};

const TEMPLATE_KEYS = Object.keys(SHUNT_TEMPLATES);

/**
 * Generates a stream of 15–20 clips with 0, 1, 2 or 3 shunt patterns
 * distributed across left/right legs. New random composition on every call
 * (called once per page load and on every refresh).
 *
 * Shunt count distribution:
 *   0 shunts (all normal): ~15 %
 *   1 shunt:               ~25 %
 *   2 shunts:              ~35 %
 *   3 shunts:              ~25 %
 *
 * Total clip count is a random integer in [15, 20].
 */
function generateMockClips() {
  // --- 1. Decide how many shunts this run contains ---
  const roll = Math.random();
  const numShunts = roll < 0.15 ? 0 : roll < 0.40 ? 1 : roll < 0.75 ? 2 : 3;

  // --- 2. Target total clip count (15–20 inclusive) ---
  const targetCount = 15 + Math.floor(Math.random() * 6); // 15, 16, 17, 18, 19 or 20

  // --- 3. Build shunt clips ---
  const legs = ['left', 'right'];
  const allClips = [];
  let seq = 1;

  const TEMPLATE_LABELS = { type1: 'Type 1', type3: 'Type 3', type2a: 'Type 2A', type2c: 'Type 2C', type1plus2: 'Type 1+2' };

  for (let i = 0; i < numShunts; i++) {
    const template = TEMPLATE_KEYS[Math.floor(Math.random() * TEMPLATE_KEYS.length)];
    const leg = legs[i % 2]; // alternate legs
    const clips = SHUNT_TEMPLATES[template](leg, seq);
    const gtLabel = TEMPLATE_LABELS[template] || template;
    clips.forEach(c => { c.groundTruth = gtLabel; });
    allClips.push(...clips);
    seq += clips.length;
    // Stop early if we've already hit the target
    if (allClips.length >= targetCount) break;
  }

  // --- 4. Pad with normal EP clips up to targetCount ---
  const padLegs = ['left', 'right'];
  const padSteps = ['SFJ-Knee', 'Knee-Ankle', 'SPJ-Ankle', 'Knee', 'Ankle'];
  while (allClips.length < targetCount) {
    const leg = padLegs[allClips.length % 2];
    const step = padSteps[Math.floor(Math.random() * padSteps.length)];
    const normalFromTypes = ['N1', 'N2', 'N1', 'N1'];
    const ft = normalFromTypes[Math.floor(Math.random() * normalFromTypes.length)];
    allClips.push({
      sequenceNumber: seq++,
      flow: 'EP', fromType: ft, toType: ft,
      posXRatio: px(leg, step), posYRatio: py(leg, step),
      step,
      legSide: leg,
      confidence: rnd(0.90, 0.97),
      reflux_duration: 0.0,
      description: `Normal ${ft === 'N1' ? 'deep' : ft === 'N2' ? 'saphenous' : 'tributary'} system — no reflux`,
      eliminationTest: '',
      clipPath: `frame-${String(seq).padStart(3,'0')}.png`,
      groundTruth: 'No Shunt (Normal)',
    });
  }

  // --- 5. Trim to targetCount if shunt templates overshot ---
  if (allClips.length > targetCount) allClips.splice(targetCount);

  // --- 6. Renumber sequenceNumber in order ---
  allClips.forEach((c, i) => { c.sequenceNumber = i + 1; });

  return JSON.stringify(allClips, null, 2);
}

/**
 * Normalise a shunt label for fuzzy comparison.
 * Strips "Type", whitespace, and handles "No Shunt" variants.
 */
function normLabel(s) {
  return (s || '').toLowerCase()
    .replace(/\btype\b/g, '').replace(/\s+/g, '').replace(/[()]/g, '');
}
const NO_SHUNT_RE = /no\s*shunt|normal|negative|no\s*reflux\s*found|no\s*patholog/i;

/**
 * Returns true if llmText matches the ground-truth label.
 * Handles:
 *   - "No Shunt (Normal)" / "No shunt detected" / "Normal study"
 *   - "Type 2A" vs "2a", "Type 1+2" vs "1+2", bilateral "Type 1 (R) + Type 2A (L)"
 */
function shuntLabelMatch(groundTruth, llmText) {
  const gt = (groundTruth || '').trim();
  const llm = (llmText || '').trim();
  if (!gt || gt === '—') return !NO_SHUNT_RE.test(llm) && llm.length > 0; // no GT → any non-empty answer

  const gtIsNormal = /no\s*shunt|normal/i.test(gt);
  const llmIsNormal = NO_SHUNT_RE.test(llm);
  if (gtIsNormal && llmIsNormal) return true;   // both say "no shunt" — match
  if (gtIsNormal !== llmIsNormal) return false;  // one says shunt, other doesn't — mismatch

  const n = normLabel;
  const gtN = n(gt);
  const llmN = n(llm);

  // Exact normalised match
  if (gtN === llmN) return true;

  // Substring containment (handles "Type 1 - Simple..." containing "1")
  if (llmN.includes(gtN) || gtN.includes(llmN)) return true;

  // For bilateral labels like "Type 1 (R) + Type 2A (L)", check each part
  const gtParts = gt.split(/[+&,]/).map(p => n(p.replace(/[()rl]/gi, '').trim())).filter(Boolean);
  if (gtParts.length > 1) {
    return gtParts.some(part => llmN.includes(part));
  }

  return false;
}

const normalizeLigationSteps = (finding = {}) => {
  const steps = Array.isArray(finding.ligation_steps) && finding.ligation_steps.length > 0
    ? finding.ligation_steps
    : Array.isArray(finding.ligation)
      ? finding.ligation
      : [];

  return steps
    .map((step) => String(step).replace(/^[•\-\s]+/, '').trim())
    .filter(Boolean);
};

const getPointOfLigation = (finding = {}) => {
  const explicitPoint = String(finding.point_of_ligation || '').trim();
  if (explicitPoint) return explicitPoint;

  const steps = normalizeLigationSteps(finding);
  if (steps.length > 0) return steps[0];

  return String(finding.primary_target || finding.ligation_target || finding.location || '').trim();
};

const getLigationApproach = (finding = {}) => String(finding.chiva_approach || finding.approach || '').trim();


const ClinicalReasoning = () => {
  const [mode, setMode] = useState('single'); // 'single', 'stream', 'report', or 'accuracy'

  // Accuracy-testing mode state
  const [accStreamResults, setAccStreamResults] = useState(() => {
    try { return JSON.parse(localStorage.getItem('acc_stream_results') || '{}'); } catch { return {}; }
  });
  const [accStreamLoading, setAccStreamLoading] = useState({}); // { streamId: bool }
  const [accExpandedStream, setAccExpandedStream] = useState(null);

  // Post-assessment report state — new random clips on every page load
  const [reportClips, setReportClips] = useState(() => generateMockClips());
  const [reportPatientInfo, setReportPatientInfo] = useState(JSON.stringify({
    "patient_id": "PAT-001",
    "assessor": "Dr. Smith",
    "leg_side": "Left",
    "notes": ""
  }, null, 2));
  const [reportResult, setReportResult] = useState(null);
  const [reportLoading, setReportLoading] = useState(false);
  const [reportError, setReportError] = useState(null);
  
  // Single mode state - Task-1: Temporal Flow Analysis with Ligation
  const [inputData, setInputData] = useState(JSON.stringify({
    sequenceNumber: 1,
    fromType: "N1",
    toType: "N2",
    step: "SFJ-Knee",
    flow: "RP",
    posXRatio: 0.65,
    posYRatio: 0.08,
    clipPath: "video-frame-001.png",
    legSide: "left",
    confidence: 0.92,
    reflux_duration: 2.1,
    description: "GSV reflux with SFJ incompetence",
    ligation: {
      shunt_type: "Type 1 - Simple N1-N2-N1",
      primary_target: "Saphenofemoral junction (SFJ)",
      technique: "Endovenous Laser Ablation (EVLA)",
      wavelength: "1470nm",
      power: "12W",
      pullback_speed: "1cm/min",
      total_duration: "90-120 seconds",
      compression_protocol: {
        phase1: "Class III 40-50mmHg, weeks 1-2",
        phase2: "Class II 23-32mmHg, weeks 3-6"
      },
      follow_up: "Week 2 and Week 6 ultrasound",
      contraindications: "Active thrombosis, acute cellulitis, uncontrolled hypercoagulability",
      saphenous_nerve_protection: "Tumescent anesthesia with 0.1% lidocaine in 500mL NS",
      complications_to_monitor: "Paresthesia (2%), DVT risk <1%, skin burns <0.5%",
      clinical_rationale: "Type 1 (simple GSV reflux) responds excellently to SFJ ablation alone. EVLA preferred for saphenous nerve preservation compared to stripping. Success rate 95% at 1 year with saphenous-sparing approach. Avoid unnecessary tributary ablation per book guidelines."
    }
  }, null, 2));

  // Stream mode state — anatomically grounded posX/posY per the CHIVA scanning coordinate system:
  // (0,0)=top-left, (1,1)=bottom-right. Right leg: X 0.09–0.48 (left side of image).
  // Left leg: X 0.50–0.91 (right side). Y ≤ 0.55 = SFJ-Knee zone; Y > 0.55 = Knee-Ankle/SPJ zone.
  const [streamData, setStreamData] = useState(JSON.stringify([
    // ── Cluster 1: LEFT leg — Type 1 (EP N1→N2 at SFJ, RP N2→N1) ────────────
    // Left leg SFJ zone: X 0.4985–0.909, Y 0.03–0.098
    { sequenceNumber:1,  flow:"EP", fromType:"N1", toType:"N1", posXRatio:0.72, posYRatio:0.06, step:"SFJ",       legSide:"left",  confidence:0.95, reflux_duration:0.0, description:"Left leg baseline deep vein — normal forward flow at groin", clipPath:"frame-001.png" },
    { sequenceNumber:2,  flow:"EP", fromType:"N1", toType:"N2", posXRatio:0.68, posYRatio:0.05, step:"SFJ",       legSide:"left",  confidence:0.94, reflux_duration:0.0, description:"Left SFJ entry point — EP N1→N2 at saphenofemoral junction", clipPath:"frame-002.png" },
    { sequenceNumber:3,  flow:"RP", fromType:"N1", toType:"N2", posXRatio:0.65, posYRatio:0.05, step:"SFJ",       legSide:"left",  confidence:0.90, reflux_duration:0.8, description:"Left SFJ reflux — N1→N2 at groin level (Type 1)", clipPath:"frame-003.png", ligation:{ procedure_name:"SFJ Ligation (Crossectomy)", technique:"Open groin incision, 4-0 Vicryl at SFJ", location:"Saphenofemoral junction", vessels_ligated:["GSV at SFJ","Superficial epigastric vein"], compression_post_op:"Class III 40-50mmHg wk 1-2, Class II wk 3-6" } },
    // Left leg SFJ-Knee zone: X 0.4985–0.909, Y 0.10–0.50
    { sequenceNumber:4,  flow:"RP", fromType:"N2", toType:"N1", posXRatio:0.62, posYRatio:0.18, step:"SFJ-Knee",  legSide:"left",  confidence:0.88, reflux_duration:1.1, description:"Left RP N2→N1 proximal GSV — Hunterian canal region", clipPath:"frame-004.png" },
    { sequenceNumber:5,  flow:"RP", fromType:"N2", toType:"N1", posXRatio:0.59, posYRatio:0.35, step:"SFJ-Knee",  legSide:"left",  confidence:0.86, reflux_duration:0.9, description:"Left RP N2→N1 mid-GSV — medial thigh", clipPath:"frame-005.png" },
    // Left leg Knee: X 0.4985–0.909, Y 0.45–0.5497
    { sequenceNumber:6,  flow:"EP", fromType:"N2", toType:"N2", posXRatio:0.71, posYRatio:0.50, step:"Knee",      legSide:"left",  confidence:0.93, reflux_duration:0.0, description:"Left knee — normal GSV flow below junction", clipPath:"frame-006.png" },
    // ── Cluster 2: LEFT leg — Type 3 (EP N2→N3, RP N3→N2, RP N2→N1) ─────────
    { sequenceNumber:7,  flow:"EP", fromType:"N2", toType:"N3", posXRatio:0.66, posYRatio:0.22, step:"SFJ-Knee",  legSide:"left",  confidence:0.92, reflux_duration:0.0, description:"Left EP N2→N3 — tributary entry along medial thigh", clipPath:"frame-007.png" },
    { sequenceNumber:8,  flow:"RP", fromType:"N3", toType:"N2", posXRatio:0.70, posYRatio:0.49, step:"Knee",      legSide:"left",  confidence:0.87, reflux_duration:1.2, description:"Left RP N3→N2 — tributary reflux at knee level", clipPath:"frame-008.png", ligation:{ procedure_name:"Tributary Ligation at N2→N3", technique:"Small 2cm incision, 3-0 Vicryl at tributary junction", location:"Left medial thigh tributary", vessels_ligated:["N3 tributary at N2 junction"], compression_post_op:"Class III wk 1-2, Class II wk 3-8" } },
    // Left leg Knee-Ankle: X 0.7081–0.91, Y 0.5497–0.90
    { sequenceNumber:9,  flow:"RP", fromType:"N3", toType:"N2", posXRatio:0.78, posYRatio:0.65, step:"Knee-Ankle", legSide:"left",  confidence:0.85, reflux_duration:0.9, description:"Left RP N3→N2 distal tributary — Boyd perforator zone", clipPath:"frame-009.png" },
    { sequenceNumber:10, flow:"RP", fromType:"N2", toType:"N1", posXRatio:0.63, posYRatio:0.28, step:"SFJ-Knee",  legSide:"left",  confidence:0.84, reflux_duration:1.3, description:"Left RP N2→N1 via GSV after tributary loop — Hunterian level", clipPath:"frame-010.png" },
    { sequenceNumber:11, flow:"EP", fromType:"N3", toType:"N3", posXRatio:0.80, posYRatio:0.73, step:"Knee-Ankle", legSide:"left",  confidence:0.93, reflux_duration:0.0, description:"Left distal tributary — normal calf segment", clipPath:"frame-011.png" },
    // Left leg SPJ: X 0.588–0.714, Y 0.55–0.72
    { sequenceNumber:12, flow:"EP", fromType:"N1", toType:"N1", posXRatio:0.63, posYRatio:0.63, step:"SPJ",        legSide:"left",  confidence:0.94, reflux_duration:0.0, description:"Left SPJ — popliteal vein competent", clipPath:"frame-012.png" },
    // ── Cluster 3: RIGHT leg — Type 2A (EP N2→N3 only, no SFJ entry, RP N3) ──
    // Right leg SFJ-Knee zone: X 0.0931–0.475, Y 0.10–0.50
    { sequenceNumber:13, flow:"EP", fromType:"N2", toType:"N3", posXRatio:0.26, posYRatio:0.28, step:"SFJ-Knee",  legSide:"right", confidence:0.91, reflux_duration:0.0, description:"Right EP N2→N3 — isolated tributary entry, no SFJ incompetence", clipPath:"frame-013.png" },
    { sequenceNumber:14, flow:"EP", fromType:"N2", toType:"N3", posXRatio:0.22, posYRatio:0.38, step:"SFJ-Knee",  legSide:"right", confidence:0.90, reflux_duration:0.0, description:"Right second N2→N3 branch — medial thigh", clipPath:"frame-014.png" },
    // Right leg Knee: X 0.0931–0.475, Y 0.45–0.5497
    { sequenceNumber:15, flow:"RP", fromType:"N3", toType:"N2", posXRatio:0.30, posYRatio:0.49, step:"Knee",      legSide:"right", confidence:0.86, reflux_duration:1.0, description:"Right RP N3→N2 — Type 2A at knee junction", clipPath:"frame-015.png", ligation:{ procedure_name:"Ligate highest EP at N2→N3", technique:"Small incision at highest EP entry, 3-0 absorbable sutures", location:"Right medial knee", vessels_ligated:["Highest N2→N3 entry point"], compression_post_op:"Class II 23-32mmHg wk 1-4" } },
    // Right leg Knee-Ankle: X 0.105–0.2947, Y 0.5497–0.90
    { sequenceNumber:16, flow:"RP", fromType:"N3", toType:"N2", posXRatio:0.19, posYRatio:0.68, step:"Knee-Ankle", legSide:"right", confidence:0.84, reflux_duration:0.8, description:"Right RP N3→N2 distal tributary — Cockett zone", clipPath:"frame-016.png" },
    { sequenceNumber:17, flow:"EP", fromType:"N2", toType:"N2", posXRatio:0.28, posYRatio:0.32, step:"SFJ-Knee",  legSide:"right", confidence:0.93, reflux_duration:0.0, description:"Right GSV competent — no SFJ reflux confirmed", clipPath:"frame-017.png" },
    // Right leg Ankle: X 0.105–0.2947, Y 0.86–0.98
    { sequenceNumber:18, flow:"EP", fromType:"N1", toType:"N1", posXRatio:0.17, posYRatio:0.92, step:"Ankle",     legSide:"right", confidence:0.92, reflux_duration:0.0, description:"Right ankle — normal deep system, distal Cockett perforators competent", clipPath:"frame-018.png" }
  ], null, 2));
  
  const [bufferInterval, setBufferInterval] = useState(0.5);

  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState(null);
  const [error, setError] = useState(null);
  const [analysisTime, setAnalysisTime] = useState(null);

  // ── Run history (persisted to localStorage) ─────────────────────────────────
  const [runs, setRuns] = useState(() => {
    try { return JSON.parse(localStorage.getItem('task1_runs') || '[]'); } catch { return []; }
  });
  const saveRun = (run) => {
    setRuns(prev => {
      const updated = [run, ...prev].slice(0, 50); // keep last 50 runs
      try { localStorage.setItem('task1_runs', JSON.stringify(updated)); } catch {}
      return updated;
    });
  };

  // ── Metrics panel state ──────────────────────────────────────────────────────
  const [metricsOpen, setMetricsOpen] = useState(false);
  const [selectedRunId, setSelectedRunId] = useState(null);
  const [metricsTab, setMetricsTab] = useState('stream'); // 'stream' | 'single' | 'report'

  // Single mode analysis
  const handleAnalyze = async () => {
    setError(null);
    setResponse(null);
    setLoading(true);

    try {
      const startTime = performance.now();

      // Parse input JSON
      let ultrasoundData;
      try {
        ultrasoundData = JSON.parse(inputData);
      } catch (e) {
        throw new Error('Invalid JSON input: ' + e.message);
      }

      // Make API call
      const res = await axios.post('/api/analyze', { ultrasound_data: ultrasoundData });

      const elapsedSec = ((performance.now() - startTime) / 1000).toFixed(2);
      setAnalysisTime(elapsedSec);
      setResponse(res.data);

      // Save single-mode run
      const isRP = ultrasoundData.flow === 'RP';
      const assessment = res.data.shunt_type_assessment || '';
      const tu = res.data.token_usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 };
      const noShunt = /no shunt|insufficient/i.test(assessment);
      saveRun({
        id: `run-${Date.now()}`,
        source: 'single',
        timestamp: new Date().toISOString(),
        total_clips: 1,
        rp_clips: isRP ? 1 : 0,
        ep_clips: isRP ? 0 : 1,
        duration_sec: parseFloat(elapsedSec),
        accuracy_pct: isRP ? (noShunt ? '0.0' : '100.0') : null,
        rp_correct: isRP && !noShunt ? 1 : 0,
        total_prompt_tokens: tu.prompt_tokens || 0,
        total_completion_tokens: tu.completion_tokens || 0,
        total_tokens: tu.total_tokens || 0,
        avg_tokens_per_rp: isRP ? (tu.total_tokens || 0) : 0,
        predictions: isRP ? [{
          seq: ultrasoundData.sequenceNumber || 1,
          flow_type: 'RP',
          step: ultrasoundData.step || '—',
          leg_side: ultrasoundData.legSide || '—',
          vein_path: `${ultrasoundData.fromType || '?'}→${ultrasoundData.toType || '?'}`,
          reflux_duration: ultrasoundData.reflux_duration || 0,
          description: ultrasoundData.description || '',
          ground_truth: ultrasoundData.groundTruth || '—',
          assessment,
          reasoning: res.data.reasoning || '',
          treatment: res.data.treatment_plan || '',
          prompt_tokens: tu.prompt_tokens || 0,
          completion_tokens: tu.completion_tokens || 0,
          total_tokens: tu.total_tokens || 0,
        }] : [],
      });
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Analysis failed');
    } finally {
      setLoading(false);
    }
  };

  // Stream mode analysis - PROCESS ONE POINT AT A TIME WITH UPDATES + RUN RECORDING
  const handleStreamAnalysis = async () => {
    setError(null);
    setResponse(null);
    setLoading(true);

    const runId = `run-${Date.now()}`;
    const runStartTime = performance.now();

    try {
      let dataStream;
      try { dataStream = JSON.parse(streamData); }
      catch (e) { throw new Error('Invalid JSON input: ' + e.message); }
      if (!Array.isArray(dataStream)) throw new Error('Stream data must be an array');

      const results = {
        total_points: dataStream.length,
        processed_points: [],
        current_reasoning: '',
        current_assessment: '',
        current_treatment: '',
        shunt_classifications: []
      };

      // Per-point run record (for metrics dashboard)
      const runPredictions = [];   // one entry per RP clip
      let runTotalPromptTokens = 0;
      let runTotalCompletionTokens = 0;
      let runTotalTokens = 0;
      let rpCorrect = 0;   // RP clips where LLM detected a shunt (not "No shunt" / "Insufficient")
      let rpCount = 0;

      for (let i = 0; i < dataStream.length; i++) {
        const dataPoint = dataStream[i];
        if (i > 0) await new Promise(r => setTimeout(r, bufferInterval * 1000));

        try {
          const isRP = dataPoint.flow === 'RP';

          if (isRP) {
            rpCount++;
            const res = await axios.post('/api/analyze', { ultrasound_data: dataPoint });
            const assessment = res.data.shunt_type_assessment || '';
            const reasoning  = res.data.reasoning || '';
            const treatment  = res.data.treatment_plan || '';
            const tu = res.data.token_usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 };

            // Accumulate tokens
            runTotalPromptTokens     += tu.prompt_tokens     || 0;
            runTotalCompletionTokens += tu.completion_tokens || 0;
            runTotalTokens           += tu.total_tokens      || 0;

            // Accuracy: compare LLM assessment against ground truth
            const gt = dataPoint.groundTruth || '';
            const matchesGt = shuntLabelMatch(gt, assessment);
            if (matchesGt) rpCorrect++;

            results.processed_points.push({
              point_number: i + 1,
              sequence_number: dataPoint.sequenceNumber,
              flow_type: dataPoint.flow,
              location: dataPoint.step,
              leg_side: dataPoint.legSide,
              vein_path: `${dataPoint.fromType}→${dataPoint.toType}`,
              reflux_duration: dataPoint.reflux_duration,
              description: dataPoint.description || 'Reflux detected',
              ligation: dataPoint.ligation || null,
            });
            results.current_assessment = assessment;
            results.current_reasoning  = reasoning;
            results.current_treatment  = treatment;
            results.shunt_classifications.push({
              point: i + 1,
              sequence: dataPoint.sequenceNumber,
              flow: `${dataPoint.fromType}→${dataPoint.toType}`,
              step: dataPoint.step,
              leg_side: dataPoint.legSide,
              assessment,
            });

            // Record for metrics panel
            runPredictions.push({
              seq: dataPoint.sequenceNumber,
              flow_type: 'RP',
              step: dataPoint.step,
              leg_side: dataPoint.legSide,
              vein_path: `${dataPoint.fromType}→${dataPoint.toType}`,
              reflux_duration: dataPoint.reflux_duration,
              description: dataPoint.description || '',
              ground_truth: dataPoint.groundTruth || '—',
              assessment,
              reasoning,
              treatment,
              prompt_tokens:     tu.prompt_tokens,
              completion_tokens: tu.completion_tokens,
              total_tokens:      tu.total_tokens,
            });

          } else {
            results.processed_points.push({
              point_number: i + 1,
              sequence_number: dataPoint.sequenceNumber,
              flow_type: dataPoint.flow,
              location: dataPoint.step,
              leg_side: dataPoint.legSide,
              vein_path: `${dataPoint.fromType}→${dataPoint.toType}`,
              reflux_duration: dataPoint.reflux_duration,
              description: dataPoint.description || 'Normal forward flow',
              ligation: null,
            });
          }

          setResponse({ ...results });

        } catch (pointErr) {
          console.error(`Error processing point ${i + 1}:`, pointErr);
          throw pointErr;
        }
      }

      const elapsedSec = ((performance.now() - runStartTime) / 1000).toFixed(2);
      setAnalysisTime(elapsedSec);
      setResponse({ ...results });

      // ── Save completed run to history ──────────────────────────────────────
      const accuracy = rpCount > 0 ? ((rpCorrect / rpCount) * 100).toFixed(1) : null;
      saveRun({
        id: runId,
        source: 'stream',
        timestamp: new Date().toISOString(),
        total_clips: dataStream.length,
        rp_clips: rpCount,
        ep_clips: dataStream.length - rpCount,
        duration_sec: parseFloat(elapsedSec),
        accuracy_pct: accuracy,          // % of RP clips where shunt was detected
        rp_correct: rpCorrect,
        total_prompt_tokens: runTotalPromptTokens,
        total_completion_tokens: runTotalCompletionTokens,
        total_tokens: runTotalTokens,
        avg_tokens_per_rp: rpCount > 0 ? Math.round(runTotalTokens / rpCount) : 0,
        predictions: runPredictions,     // full per-RP detail
      });

    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Stream analysis failed');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setResponse(null);
    setError(null);
    setAnalysisTime(null);
  };

  // Post-assessment report handlers
  const handleGenerateReport = async () => {
    setReportError(null);
    setReportResult(null);
    setReportLoading(true);
    const t0 = Date.now();
    try {
      let clips, patientInfo;
      try { clips = JSON.parse(reportClips); } catch(e) { throw new Error('Invalid clip_list JSON: ' + e.message); }
      try { patientInfo = JSON.parse(reportPatientInfo); } catch(e) { patientInfo = {}; }
      const { data: { classification } } = await axios.post('/api/shunt/classify-report', { clip_list: clips, patient_info: patientInfo });
      setReportResult(classification);

      // Save run — report mode (single batch LLM call, no per-clip token breakdown)
      const rpClips = clips.filter(c => c.flow === 'RP' || c.clipType === 'RP');
      const rpCount = rpClips.length;
      const findings = classification.findings || (classification.shunt_type ? [classification] : []);
      const predictions = findings.map(f => ({
        leg: f.leg || 'Assessment',
        shunt_type: f.shunt_type || '—',
        confidence: f.confidence != null ? ((f.confidence * 100).toFixed(0) + '%') : '—',
        reasoning: (f.reasoning || []).join(' | '),
        point_of_ligation: getPointOfLigation(f),
        approach: getLigationApproach(f),
        clinical_rationale: String(f.clinical_rationale || '').trim(),
        ligation_steps: normalizeLigationSteps(f).join(' | '),
      }));
      // accuracy: did LLM detect a concrete shunt type (report mode has no per-finding GT)
      const noShuntRe = /no\s*shunt|normal|insufficient/i;
      const correctFindings = findings.filter(f => !noShuntRe.test(f.shunt_type || '')).length;
      const accuracyPct = findings.length > 0 ? ((correctFindings / findings.length) * 100).toFixed(1) : null;

      saveRun({
        id: `run-${Date.now()}`,
        source: 'report',
        timestamp: new Date().toISOString(),
        total_clips: clips.length,
        rp_clips: rpCount,
        ep_clips: clips.length - rpCount,
        duration_sec: ((Date.now() - t0) / 1000).toFixed(1),
        accuracy_pct: accuracyPct,
        rp_correct: correctFindings,
        total_prompt_tokens: 0,
        total_completion_tokens: 0,
        total_tokens: 0,
        avg_tokens_per_rp: 0,
        predictions,
      });
    } catch(err) {
      setReportError(err.response?.data?.error || err.message || 'Report generation failed');
    } finally {
      setReportLoading(false);
    }
  };

  const handleDownloadPDF = async () => {
    try {
      let clips, patientInfo;
      try { clips = JSON.parse(reportClips); } catch(e) { throw new Error('Invalid clip_list JSON'); }
      try { patientInfo = JSON.parse(reportPatientInfo); } catch(e) { patientInfo = {}; }
      const res = await axios.post(
        '/api/shunt/classify-report?format=pdf',
        { clip_list: clips, patient_info: patientInfo },
        { responseType: 'blob' }
      );
      const url = window.URL.createObjectURL(new Blob([res.data], { type: 'application/pdf' }));
      const a = document.createElement('a');
      a.href = url;
      a.download = `shunt_report_${new Date().toISOString().slice(0,10)}.pdf`;
      a.click();
      window.URL.revokeObjectURL(url);
    } catch(err) {
      setReportError('PDF download failed: ' + (err.message || 'Unknown error'));
    }
  };

  // ── Accuracy Testing mode: classify a single stream ──────────────────────
  const handleAccuracyClassify = async (stream) => {
    setAccStreamLoading(prev => ({ ...prev, [stream.id]: true }));
    const t0 = Date.now();
    try {
      const response = await axios.post('/api/shunt/classify-report', {
        clip_list: stream.clips,
        patient_info: { stream_id: stream.id, ground_truth: stream.groundTruth },
      });
      const { classification, total_prompt_tokens = 0, total_completion_tokens = 0, total_tokens = 0 } = response.data;

      const findings = classification.findings || (classification.shunt_type ? [classification] : []);
      const llmLabel = findings.map(f => f.shunt_type).filter(Boolean).join(' + ') || 'No Shunt';
      const gt = stream.groundTruth;
      const isCorrect = shuntLabelMatch(gt, llmLabel);

      const result = {
        streamId: stream.id,
        groundTruth: gt,
        llmLabel,
        isCorrect,
        confidence: findings[0]?.confidence ?? null,
        reasoning: findings.map(f => (f.reasoning || []).join(' ')).join(' | '),
        ligation: findings.flatMap(f => normalizeLigationSteps(f)),
        ligation_steps: findings.flatMap(f => normalizeLigationSteps(f)),
        point_of_ligation: findings.map(f => getPointOfLigation(f)).filter(Boolean).join(' | '),
        chiva_approach: findings.map(f => getLigationApproach(f)).filter(Boolean).join(' | '),
        clinical_rationale: findings.map(f => String(f.clinical_rationale || '').trim()).filter(Boolean).join(' | '),
        findings,
        total_tokens,
        total_prompt_tokens,
        total_completion_tokens,
        duration_sec: ((Date.now() - t0) / 1000).toFixed(1),
        timestamp: new Date().toISOString(),
      };

      setAccStreamResults(prev => {
        const updated = { ...prev, [stream.id]: result };
        try { localStorage.setItem('acc_stream_results', JSON.stringify(updated)); } catch {}
        return updated;
      });

      // Also save to metrics dashboard as an 'accuracy' source run
      const avgTokensPerRp = stream.clips.filter(c => c.flow === 'RP').length > 0
        ? Math.round(total_tokens / stream.clips.filter(c => c.flow === 'RP').length)
        : 0;

      saveRun({
        id: `run-${Date.now()}`,
        source: 'accuracy',
        timestamp: result.timestamp,
        stream_id: stream.id,
        stream_name: stream.name,
        total_clips: stream.clips.length,
        rp_clips: stream.clips.filter(c => c.flow === 'RP').length,
        ep_clips: stream.clips.filter(c => c.flow === 'EP').length,
        duration_sec: parseFloat(result.duration_sec),
        accuracy_pct: isCorrect ? '100.0' : '0.0',
        rp_correct: isCorrect ? 1 : 0,
        total_prompt_tokens, total_completion_tokens, total_tokens,
        avg_tokens_per_rp: avgTokensPerRp,
        predictions: findings.map(f => ({
          seq: 0,
          flow_type: 'batch',
          step: 'all clips',
          leg_side: f.leg || 'all',
          vein_path: '—',
          reflux_duration: 0,
          ground_truth: gt,
          assessment: f.shunt_type || '—',
          reasoning: (f.reasoning || []).join(' '),
          treatment: (f.ligation || []).join(', '),
          prompt_tokens: Math.round(total_prompt_tokens / findings.length),
          completion_tokens: Math.round(total_completion_tokens / findings.length),
          total_tokens: Math.round(total_tokens / findings.length),
        })),
      });
    } catch (err) {
      setAccStreamResults(prev => ({
        ...prev,
        [stream.id]: { streamId: stream.id, error: err.response?.data?.error || err.message },
      }));
    } finally {
      setAccStreamLoading(prev => ({ ...prev, [stream.id]: false }));
    }
  };

  return (
  <><div className="page-container">
      {/* Mode Selection */}
      <div className="section">
        <h2 className="section-title">🔄 Analysis Mode</h2>
        <div className="section-content">
          <div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
              <input
                type="radio"
                value="single"
                checked={mode === 'single'}
                onChange={(e) => setMode(e.target.value)} />
              Single Data Point
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
              <input
                type="radio"
                value="stream"
                checked={mode === 'stream'}
                onChange={(e) => setMode(e.target.value)} />
              Continuous Stream (with buffer)
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
              <input
                type="radio"
                value="report"
                checked={mode === 'report'}
                onChange={(e) => setMode(e.target.value)} />
              Assess Captured Clips (Post-Assessment)
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
              <input
                type="radio"
                value="accuracy"
                checked={mode === 'accuracy'}
                onChange={(e) => setMode(e.target.value)} />
              🎯 Shunt Classification Accuracy Testing
            </label>
          </div>
        </div>
      </div>

      {/* Single Mode Input */}
      {mode === 'single' && (
        <div className="section">
          <h2 className="section-title">📊 Single Ultrasound Input</h2>
          <div className="section-content">
            <div className="form-group">
              <label className="form-label">
                Ultrasound Data (JSON)
                <span className="text-muted"> - Include reflux_type and description for shunt classification</span>
              </label>
              <textarea
                className="form-textarea"
                value={inputData}
                onChange={(e) => setInputData(e.target.value)}
                placeholder="Enter ultrasound JSON data" />
            </div>

            <p className="text-muted" style={{ fontSize: '0.85rem', marginBottom: '1rem' }}>
              <strong>Required fields for shunt classification:</strong> reflux_type, location, reflux_duration, description
            </p>

            <div style={{ display: 'flex', gap: '1rem' }}>
              <button
                className="btn btn-primary"
                onClick={handleAnalyze}
                disabled={loading}
              >
                {loading ? '🔄 Analyzing...' : '🔍 Analyze'}
              </button>
              {response && (
                <button
                  className="btn btn-secondary"
                  onClick={handleClear}
                >
                  Clear Results
                </button>
              )}
            </div>

            {analysisTime && (
              <p className="text-muted mt-2">
                Analysis completed in {analysisTime}s
              </p>
            )}
          </div>
        </div>
      )}

      {/* Stream Mode Input */}
      {mode === 'stream' && (
        <div className="section">
          <h2 className="section-title" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.5rem' }}>
            <span>🌊 Continuous Data Stream</span>
            <button
              className="btn btn-secondary"
              onClick={() => setMetricsOpen(true)}
              style={{ fontSize: '0.85rem', padding: '0.35rem 0.9rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}
              title={`${runs.length} run(s) recorded`}
            >
              📊 View Model Metrics
            </button>
          </h2>
          <div className="section-content">
            <div className="form-group">
              <label className="form-label">
                Data Stream (JSON Array)
                <span className="text-muted"> — auto-generated on every refresh (0–3 shunts, 15–20 clips)</span>
              </label>
              <textarea
                className="form-textarea"
                value={streamData}
                onChange={(e) => setStreamData(e.target.value)}
                placeholder="Enter array of ultrasound data points"
                style={{ minHeight: '300px' }} />
            </div>

            <div className="form-group">
              <label className="form-label">
                Buffer Interval (seconds)
                <span className="text-muted"> — delay between processing each clip</span>
              </label>
              <input
                type="number"
                min="0.2"
                max="5"
                step="0.1"
                value={bufferInterval}
                onChange={(e) => setBufferInterval(parseFloat(e.target.value))}
                style={{ padding: '0.5rem', borderRadius: '0.25rem', border: '1px solid #ddd', width: '100px' }} />
            </div>

            <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', alignItems: 'center' }}>
              <button className="btn btn-primary" onClick={handleStreamAnalysis} disabled={loading}>
                {loading ? '🌊 Processing Stream...' : '🌊 Classify Shunts'}
              </button>
              <button
                className="btn btn-secondary"
                onClick={() => { setStreamData(generateMockClips()); setResponse(null); setError(null); setAnalysisTime(null); } }
                disabled={loading}
                title="Generate a fresh random stream (0–3 shunts, 15–20 clips)"
              >
                🔀 Regenerate Stream
              </button>
              {response && (
                <button className="btn btn-secondary" onClick={handleClear}>Clear Results</button>
              )}
            </div>

            {analysisTime && (
              <p className="text-muted mt-2">Stream processed in {analysisTime}s</p>
            )}
          </div>
        </div>
      )}

      {/* ── Model Metrics Modal ──────────────────────────────────────────────── */}
      {metricsOpen && (
        <div style={{
          position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.55)',
          zIndex: 9999, display: 'flex', alignItems: 'flex-start', justifyContent: 'center',
          padding: '2rem 1rem', overflowY: 'auto'
        }}>
          <div style={{
            background: '#fff', borderRadius: '12px', width: '100%', maxWidth: '960px',
            boxShadow: '0 20px 60px rgba(0,0,0,0.3)', overflow: 'hidden'
          }}>
            {/* Header */}
            <div style={{ background: '#C01C1C', padding: '1rem 1.5rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <h2 style={{ color: '#fff', margin: 0, fontSize: '1.1rem' }}>📊 Task-1 Model Metrics Dashboard</h2>
              <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
                {runs.length > 0 && (
                  <button
                    onClick={() => { if (window.confirm('Clear all run history?')) { setRuns([]); localStorage.removeItem('task1_runs'); setSelectedRunId(null); } } }
                    style={{ background: 'rgba(255,255,255,0.15)', border: '1px solid rgba(255,255,255,0.4)', color: '#fff', borderRadius: '6px', padding: '0.3rem 0.7rem', cursor: 'pointer', fontSize: '0.78rem' }}
                  >
                    🗑 Clear History
                  </button>
                )}
                <button
                  onClick={() => { setMetricsOpen(false); setSelectedRunId(null); } }
                  style={{ background: 'rgba(255,255,255,0.2)', border: 'none', color: '#fff', borderRadius: '6px', padding: '0.3rem 0.8rem', cursor: 'pointer', fontSize: '1.1rem', fontWeight: 700 }}
                >✕</button>
              </div>
            </div>

            {/* Mode Tabs */}
            {(() => {
              const tabDefs = [
                { key: 'stream', label: '🌊 Stream', desc: 'Continuous Stream' },
                { key: 'single', label: '📍 Single', desc: 'Single Data Point' },
                { key: 'report', label: '📋 Report', desc: 'Assess Captured Clips' },
                { key: 'accuracy', label: '🎯 Accuracy', desc: 'Accuracy Testing' },
              ];
              return (
                <div style={{ display: 'flex', borderBottom: '2px solid #f0d0d0', background: '#fdf9f9' }}>
                  {tabDefs.map(t => {
                    const active = metricsTab === t.key;
                    return (
                      <button
                        key={t.key}
                        onClick={() => { setMetricsTab(t.key); setSelectedRunId(null); } }
                        style={{
                          flex: 1, padding: '0.65rem 0.5rem', border: 'none', cursor: 'pointer',
                          background: active ? '#fff' : 'transparent',
                          borderBottom: active ? '2px solid #C01C1C' : '2px solid transparent',
                          marginBottom: '-2px',
                          fontWeight: active ? 700 : 400,
                          fontSize: '0.82rem', color: active ? '#C01C1C' : '#666',
                          display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.35rem',
                        }}
                      >
                        {t.label}
                      </button>
                    );
                  })}
                </div>
              );
            })()}

            {(() => {
              const tabRuns = runs.filter(r => r.source === metricsTab);
              return tabRuns.length === 0 ? (
                <div style={{ padding: '3rem', textAlign: 'center', color: '#888' }}>
                  <div style={{ fontSize: '2.5rem', marginBottom: '0.5rem' }}>📭</div>
                  <p>No runs recorded for this mode yet.</p>
                </div>
              ) : (
                <div style={{ display: 'flex', minHeight: '520px' }}>
                  {/* Run list sidebar */}
                  <div style={{ width: '280px', borderRight: '1px solid #eee', overflowY: 'auto', flexShrink: 0 }}>
                    {(() => {
                      // Summary stats for this tab's runs
                      const totalRuns = tabRuns.length;
                      const avgAcc = (() => {
                        const accRuns = tabRuns.filter(r => r.accuracy_pct !== null);
                        return accRuns.length ? (accRuns.reduce((s, r) => s + parseFloat(r.accuracy_pct), 0) / accRuns.length).toFixed(1) : 'N/A';
                      })();
                      const totalTok = tabRuns.reduce((s, r) => s + (r.total_tokens || 0), 0);
                      const avgTokPerRun = totalRuns ? Math.round(totalTok / totalRuns) : 0;
                      return (
                        <div style={{ padding: '0.75rem 1rem', background: '#fdf2f2', borderBottom: '1px solid #eee' }}>
                          <div style={{ fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#888', marginBottom: '0.4rem' }}>Summary — {metricsTab} mode</div>
                          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.4rem' }}>
                            {[['Total runs', totalRuns], ['Avg accuracy', avgAcc !== 'N/A' ? `${avgAcc}%` : 'N/A'], ['Total tokens', totalTok.toLocaleString()], ['Avg tok/run', avgTokPerRun.toLocaleString()]].map(([k, v]) => (
                              <div key={k} style={{ background: '#fff', borderRadius: '6px', padding: '0.35rem 0.5rem', border: '1px solid #f0d0d0' }}>
                                <div style={{ fontSize: '0.65rem', color: '#999' }}>{k}</div>
                                <div style={{ fontWeight: 700, fontSize: '0.9rem', color: '#C01C1C' }}>{v}</div>
                              </div>
                            ))}
                          </div>
                        </div>
                      );
                    })()}
                    {tabRuns.map((run, idx) => {
                      const isSelected = selectedRunId === run.id;
                      const ts = new Date(run.timestamp);
                      return (
                        <div
                          key={run.id}
                          onClick={() => setSelectedRunId(isSelected ? null : run.id)}
                          style={{
                            padding: '0.75rem 1rem', cursor: 'pointer', borderBottom: '1px solid #f5f5f5',
                            background: isSelected ? '#fdf2f2' : '#fff',
                            borderLeft: isSelected ? '3px solid #C01C1C' : '3px solid transparent',
                            transition: 'background 0.15s'
                          }}
                        >
                          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.2rem' }}>
                            <span style={{ fontWeight: 600, fontSize: '0.82rem', color: isSelected ? '#C01C1C' : '#333' }}>
                              Run #{tabRuns.length - idx}
                            </span>
                            <span style={{ fontSize: '0.7rem', color: '#999' }}>{ts.toLocaleTimeString()}</span>
                          </div>
                          <div style={{ fontSize: '0.72rem', color: '#555' }}>
                            {ts.toLocaleDateString()} · {run.total_clips} clips · {run.rp_clips} RP
                            {run.source === 'accuracy' && run.stream_name && (
                              <span style={{ marginLeft: '0.3rem', color: '#7c3aed' }}>· {run.stream_name}</span>
                            )}
                          </div>
                          <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.3rem', flexWrap: 'wrap' }}>
                            {run.accuracy_pct !== null && (
                              <span style={{ background: parseFloat(run.accuracy_pct) >= 80 ? '#dcfce7' : parseFloat(run.accuracy_pct) >= 50 ? '#fef9c3' : '#fee2e2', color: parseFloat(run.accuracy_pct) >= 80 ? '#166534' : parseFloat(run.accuracy_pct) >= 50 ? '#854d0e' : '#991b1b', borderRadius: '999px', padding: '1px 7px', fontSize: '0.7rem', fontWeight: 600 }}>
                                {run.accuracy_pct}% acc
                              </span>
                            )}
                            <span style={{ background: '#f3f4f6', color: '#374151', borderRadius: '999px', padding: '1px 7px', fontSize: '0.7rem' }}>
                              {(run.total_tokens || 0).toLocaleString()} tok
                            </span>
                            <span style={{ background: '#f3f4f6', color: '#374151', borderRadius: '999px', padding: '1px 7px', fontSize: '0.7rem' }}>
                              {run.duration_sec}s
                            </span>
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  {/* Run detail panel */}
                  <div style={{ flex: 1, overflowY: 'auto', padding: '1.25rem 1.5rem' }}>
                    {selectedRunId ? (() => {
                      const run = runs.find(r => r.id === selectedRunId);
                      if (!run) return null;
                      const runNo = runs.length - runs.findIndex(r => r.id === selectedRunId);
                      const accVal = run.accuracy_pct !== null ? parseFloat(run.accuracy_pct) : null;
                      const accColor = accVal === null ? '#888' : accVal >= 80 ? '#166534' : accVal >= 50 ? '#854d0e' : '#991b1b';
                      const accBg = accVal === null ? '#f3f4f6' : accVal >= 80 ? '#dcfce7' : accVal >= 50 ? '#fef9c3' : '#fee2e2';

                      return (
                        <>
                          {/* KPI cards */}
                          <h3 style={{ margin: '0 0 0.75rem', fontSize: '1rem', color: '#C01C1C' }}>
                            Run #{runNo} — {new Date(run.timestamp).toLocaleString()}
                          </h3>
                          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: '0.6rem', marginBottom: '1.25rem' }}>
                            {[
                              ['Total clips', run.total_clips, '#e0e7ff', '#3730a3'],
                              ['RP clips', run.rp_clips, '#fee2e2', '#991b1b'],
                              ['EP clips', run.ep_clips, '#d1fae5', '#065f46'],
                              ['Duration', `${run.duration_sec}s`, '#f3f4f6', '#374151'],
                            ].map(([label, val, bg, fg]) => (
                              <div key={label} style={{ background: bg, borderRadius: '8px', padding: '0.6rem 0.8rem' }}>
                                <div style={{ fontSize: '0.65rem', color: fg, opacity: 0.7, textTransform: 'uppercase', letterSpacing: '0.05em' }}>{label}</div>
                                <div style={{ fontWeight: 700, fontSize: '1.1rem', color: fg }}>{val}</div>
                              </div>
                            ))}
                          </div>

                          {/* Accuracy + token summary */}
                          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', marginBottom: '1.25rem' }}>
                            {/* Accuracy */}
                            <div style={{ background: accBg, borderRadius: '10px', padding: '1rem 1.2rem', border: `1px solid ${accColor}30` }}>
                              <div style={{ fontSize: '0.7rem', color: accColor, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.25rem' }}>Shunt Detection Accuracy</div>
                              <div style={{ fontSize: '2.2rem', fontWeight: 800, color: accColor, lineHeight: 1 }}>
                                {run.accuracy_pct !== null ? `${run.accuracy_pct}%` : 'N/A'}
                              </div>
                              <div style={{ fontSize: '0.75rem', color: accColor, marginTop: '0.3rem', opacity: 0.8 }}>
                                {run.source === 'accuracy'
                                  ? `Stream: ${run.stream_name || run.stream_id} — ${run.rp_correct ? 'Correct' : 'Incorrect'}`
                                  : run.rp_clips > 0
                                    ? `${run.rp_correct ?? '?'}/${run.rp_clips} RP clips matched ground truth`
                                    : 'No RP clips in this run'}
                              </div>
                              <div style={{ fontSize: '0.68rem', color: '#888', marginTop: '0.2rem' }}>
                                Metric: RP clips where LLM matched ground-truth shunt type
                              </div>
                            </div>
                            {/* Token summary */}
                            <div style={{ background: '#f0f9ff', borderRadius: '10px', padding: '1rem 1.2rem', border: '1px solid #bae6fd' }}>
                              <div style={{ fontSize: '0.7rem', color: '#0369a1', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.5rem' }}>Token Usage</div>
                              {[
                                ['Prompt tokens', (run.total_prompt_tokens || 0).toLocaleString()],
                                ['Completion tokens', (run.total_completion_tokens || 0).toLocaleString()],
                                ['Total tokens (run)', (run.total_tokens || 0).toLocaleString()],
                                ['Avg tokens / RP call', (run.avg_tokens_per_rp || 0).toLocaleString()],
                              ].map(([k, v]) => (
                                <div key={k} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '0.2rem' }}>
                                  <span style={{ color: '#555' }}>{k}</span>
                                  <span style={{ fontWeight: 700, color: '#0369a1' }}>{v}</span>
                                </div>
                              ))}
                            </div>
                          </div>

                          {/* Predictions table */}
                          <h4 style={{ fontSize: '0.85rem', color: '#555', margin: '0 0 0.5rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                            Per-Clip Predictions ({run.predictions?.length || 0} RP clips processed)
                          </h4>
                          {(!run.predictions || run.predictions.length === 0) ? (
                            <p style={{ color: '#aaa', fontSize: '0.85rem' }}>No RP clips were in this stream — all clips had normal forward flow (EP).</p>
                          ) : (
                            <div style={{ overflowX: 'auto' }}>
                              {run.predictions.map((p, i) => {
                                const noShunt = /no shunt|insufficient/i.test(p.assessment);
                                return (
                                  <div key={i} style={{
                                    border: '1px solid #e5e7eb', borderRadius: '8px', marginBottom: '0.75rem',
                                    overflow: 'hidden'
                                  }}>
                                    {/* Clip header */}
                                    <div style={{
                                      background: noShunt ? '#f3f4f6' : '#fdf2f2',
                                      padding: '0.5rem 0.75rem',
                                      display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap',
                                      borderBottom: '1px solid #e5e7eb'
                                    }}>
                                      <span style={{ fontWeight: 700, fontSize: '0.78rem', color: '#444' }}>Clip #{p.seq}</span>
                                      <span style={{ fontSize: '0.72rem', background: '#fee2e2', color: '#991b1b', borderRadius: '4px', padding: '1px 6px' }}>RP · {p.vein_path}</span>
                                      <span style={{ fontSize: '0.72rem', color: '#666' }}>{p.step} · {p.leg_side}</span>
                                      <span style={{ fontSize: '0.72rem', color: '#888' }}>reflux {p.reflux_duration}s</span>
                                      <span style={{ marginLeft: 'auto', fontSize: '0.7rem', background: '#f0f9ff', color: '#0369a1', borderRadius: '4px', padding: '1px 6px' }}>
                                        {p.total_tokens} tok (↑{p.prompt_tokens} ↓{p.completion_tokens})
                                      </span>
                                    </div>
                                    {/* Prediction body */}
                                    <div style={{ padding: '0.6rem 0.75rem', fontSize: '0.8rem' }}>
                                      {/* Ground truth vs LLM assessment row */}
                                      <div style={{ display: 'flex', gap: '1.5rem', marginBottom: '0.45rem', flexWrap: 'wrap' }}>
                                        {p.ground_truth && p.ground_truth !== '—' && (
                                          <div>
                                            <span style={{ fontWeight: 600, color: '#444' }}>Ground Truth: </span>
                                            <span style={{ fontWeight: 700, color: '#059669', background: '#d1fae5', padding: '1px 8px', borderRadius: '4px' }}>
                                              {p.ground_truth}
                                            </span>
                                          </div>
                                        )}
                                        <div>
                                          <span style={{ fontWeight: 600, color: '#444' }}>LLM Assessment: </span>
                                          <span style={{
                                            fontWeight: 700,
                                            color: noShunt ? '#6b7280' : '#C01C1C',
                                            background: noShunt ? '#f3f4f6' : '#fdf2f2',
                                            padding: '1px 8px', borderRadius: '4px'
                                          }}>{p.assessment || '—'}</span>
                                        </div>
                                      </div>
                                      {p.reasoning && (
                                        <div style={{ marginBottom: '0.35rem', color: '#555' }}>
                                          <span style={{ fontWeight: 600, color: '#444' }}>Reasoning: </span>{p.reasoning}
                                        </div>
                                      )}
                                      {p.treatment && (
                                        <div style={{ color: '#555' }}>
                                          <span style={{ fontWeight: 600, color: '#444' }}>Treatment: </span>
                                          <span style={{ whiteSpace: 'pre-wrap' }}>{p.treatment}</span>
                                        </div>
                                      )}
                                      {p.description && (
                                        <div style={{ marginTop: '0.25rem', fontSize: '0.72rem', color: '#888', fontStyle: 'italic' }}>{p.description}</div>
                                      )}
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                          )}
                        </>
                      );
                    })() : (
                      <div style={{ color: '#aaa', textAlign: 'center', marginTop: '4rem' }}>
                        <div style={{ fontSize: '2rem' }}>👈</div>
                        <p style={{ marginTop: '0.5rem' }}>Select a run to view details</p>
                      </div>
                    )}
                  </div>
                </div>
              );
            })()}
          </div>
        </div>
      )}

      {/* ── Post-Assessment Report Mode ── */}
      {mode === 'report' && (
        <>
          <div className="section">
            <h2 className="section-title" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.5rem' }}>
              <span>📋 Assess Captured Clips (Post-Assessment)</span>
              <button
                className="btn btn-secondary"
                onClick={() => setMetricsOpen(true)}
                style={{ fontSize: '0.85rem', padding: '0.35rem 0.9rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}
                title={`${runs.length} run(s) recorded`}
              >
                📊 View Model Metrics
              </button>
            </h2>
            <div className="section-content">
              <p className="text-muted" style={{ marginBottom: '1rem', fontSize: '0.9rem' }}>
                Paste 15–20 EP/RP clips from the completed duplex assessment. The LLM will classify
                the shunt type using few-shot examples from the CHIVA cheatsheet and generate a
                downloadable PDF report for clinical review.
              </p>

              <div className="form-group">
                <label className="form-label">
                  Clip List (JSON array, 15–20 points)
                  <span className="text-muted"> — auto-generated on every refresh (0–3 shunts, 15–20 clips)</span>
                </label>
                <textarea
                  className="form-textarea"
                  value={reportClips}
                  onChange={(e) => setReportClips(e.target.value)}
                  style={{ minHeight: '280px', fontFamily: 'monospace', fontSize: '0.82rem' }}
                  placeholder='[{"flow":"EP","fromType":"N1","toType":"N2","posXRatio":0.30,"posYRatio":0.07,...}]' />
              </div>

              <div className="form-group">
                <label className="form-label">Patient / Assessment Info (optional)</label>
                <textarea
                  className="form-textarea"
                  value={reportPatientInfo}
                  onChange={(e) => setReportPatientInfo(e.target.value)}
                  style={{ minHeight: '100px', fontFamily: 'monospace', fontSize: '0.82rem' }} />
              </div>

              <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', alignItems: 'center' }}>
                <button
                  className="btn btn-primary"
                  onClick={handleGenerateReport}
                  disabled={reportLoading}
                  style={{ backgroundColor: '#C01C1C', borderColor: '#C01C1C' }}
                >
                  {reportLoading ? '🔄 Classifying...' : '🔬 Classify Shunt'}
                </button>
                <button
                  className="btn btn-primary"
                  onClick={handleDownloadPDF}
                  disabled={reportLoading}
                  style={{ backgroundColor: '#8B0000', borderColor: '#8B0000' }}
                >
                  📄 Download PDF Report
                </button>
                <button
                  className="btn btn-secondary"
                  onClick={() => { setReportClips(generateMockClips()); setReportResult(null); setReportError(null); } }
                  disabled={reportLoading}
                  title="Generate a fresh random dataset (0–3 shunts, 15–20 clips)"
                >
                  🔀 Regenerate Clips
                </button>
                {reportResult && (
                  <button
                    className="btn btn-secondary"
                    onClick={() => { setReportResult(null); setReportError(null); } }
                  >
                    Clear
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* Report error */}
          {reportError && (
            <div className="output-container" style={{ borderLeft: '4px solid #dc2626' }}>
              <div className="output-header"><h3>❌ Error</h3></div>
              <div className="output-content"><p className="text-error">{reportError}</p></div>
            </div>
          )}

          {/* Report results */}
          {reportResult && (
            <>
              {/* Multi-finding overview badges */}
              {reportResult.findings && reportResult.findings.length > 1 && (
                <div className="output-container" style={{ borderLeft: '4px solid #C01C1C' }}>
                  <div className="output-header">
                    <h3>🔬 Findings Overview — {reportResult.findings.length} Legs</h3>
                    <span className="output-status success">✓ LLM-Based</span>
                  </div>
                  <div className="output-content">
                    <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                      {reportResult.findings.map((f, i) => (
                        <div key={i} style={{ flex: 1, minWidth: 160, background: '#C01C1C', color: 'white', borderRadius: 8, padding: '0.9rem 1.1rem', textAlign: 'center' }}>
                          <div style={{ fontSize: '0.75rem', opacity: 0.85, textTransform: 'uppercase', letterSpacing: 1 }}>{f.leg} Leg</div>
                          <div style={{ fontSize: '1.05rem', fontWeight: 700, margin: '0.3rem 0' }}>{f.shunt_type}</div>
                          <div style={{ fontSize: '0.78rem', opacity: 0.85 }}>Confidence: {((f.confidence || 0) * 100).toFixed(0)}%</div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* Per-finding detail */}
              {(reportResult.findings || [{ ...reportResult, leg: 'Assessment', num_clips: reportResult.num_clips }]).map((f, fi) => (
                <div key={fi} className="output-container" style={{ borderLeft: '4px solid #C01C1C' }}>
                  <div className="output-header">
                    <h3>🦵 {f.leg} Leg — {f.shunt_type}</h3>
                    <span className="output-status success">Confidence: {((f.confidence || 0) * 100).toFixed(0)}%</span>
                  </div>
                  <div className="output-content">
                    {f.summary && <p style={{ marginBottom: '0.8rem', color: '#333', fontStyle: 'italic', fontSize: '0.92rem' }}>{f.summary}</p>}

                    {/*{(f.needs_elim_test || f.ask_diameter || f.ask_branching) && (*/}
                    {(f.needs_elim_test || f.ask_branching) && (
                      <div style={{ background: '#FFFBEB', border: '1px solid #F59E0B', borderRadius: 4, padding: '0.5rem 0.8rem', marginBottom: '0.8rem', fontSize: '0.85rem', color: '#92400E' }}>
                        {f.needs_elim_test && <div>⚠ Elimination test required before ligation decision</div>}
                        {/* {f.ask_diameter    && <div>ℹ Specify RP diameter at N2: Small or Large</div>} */}
                        {f.ask_branching && <div>ℹ Specify N3 branching pattern</div>}
                      </div>
                    )}

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
                      <div style={{ background: '#eff6ff', borderLeft: '3px solid #C01C1C', borderRadius: 4, padding: '0.75rem' }}>
                        <div style={{ fontWeight: 600, color: '#C01C1C', fontSize: '0.85rem', marginBottom: '0.4rem' }}>Clinical Reasoning</div>
                        {(f.reasoning || []).map((r, i) => <div key={i} style={{ fontSize: '0.82rem', marginBottom: '0.2rem' }}>• {String(r).replace(/^[•\-\s]+/, '')}</div>)}
                        {(!f.reasoning || f.reasoning.length === 0) && <div style={{ fontSize: '0.82rem', color: '#888' }}>No pattern detected.</div>}
                      </div>
                      <div style={{ background: '#fff5f5', borderLeft: '3px solid #8B0000', borderRadius: 4, padding: '0.75rem' }}>
                        <div style={{ fontWeight: 600, color: '#8B0000', fontSize: '0.85rem', marginBottom: '0.4rem' }}>Proposed Ligation</div>
                        {(() => {
                          const ligationSteps = normalizeLigationSteps(f);
                          const pointOfLigation = getPointOfLigation(f);
                          const approach = getLigationApproach(f);

                          return (
                            <>
                              {pointOfLigation && (
                                <div style={{ fontSize: '0.8rem', marginBottom: '0.35rem' }}>
                                  <span style={{ fontWeight: 600 }}>Point of ligation:</span> {pointOfLigation}
                                </div>
                              )}
                              {approach && (
                                <div style={{ fontSize: '0.8rem', marginBottom: '0.35rem' }}>
                                  <span style={{ fontWeight: 600 }}>Approach:</span> {approach}
                                </div>
                              )}
                              {ligationSteps.length > 0
                                ? ligationSteps.map((l, i) => <div key={i} style={{ fontSize: '0.82rem', marginBottom: '0.2rem', fontWeight: i === 0 ? 600 : 400 }}>• {l}</div>)
                                : <div style={{ fontSize: '0.82rem', color: '#888' }}>No ligation required.</div>}
                            </>
                          );
                        })()}
                      </div>
                    </div>

                    {String(f.chiva_approach || '').trim() && String(f.chiva_approach).trim().toLowerCase() !== 'unable to determine.' && (
                      <div style={{ marginTop: '0.75rem', background: '#f8fafc', border: '1px solid #e5e7eb', borderRadius: 4, padding: '0.75rem' }}>
                        <div style={{ fontWeight: 600, color: '#374151', fontSize: '0.85rem', marginBottom: '0.35rem' }}>CHIVA Approach</div>
                        <div style={{ fontSize: '0.82rem', color: '#555' }}>{f.chiva_approach}</div>
                      </div>
                    )}

                    {String(f.clinical_rationale || '').trim() && (
                      <div style={{ marginTop: '0.75rem', background: '#f8fafc', border: '1px solid #e5e7eb', borderRadius: 4, padding: '0.75rem' }}>
                        <div style={{ fontWeight: 600, color: '#374151', fontSize: '0.85rem', marginBottom: '0.35rem' }}>Clinical Rationale</div>
                        <div style={{ fontSize: '0.82rem', color: '#555' }}>{f.clinical_rationale}</div>
                      </div>
                    )}

                    {Array.isArray(f.additional_info_needed) && f.additional_info_needed.length > 0 && (
                      <div style={{ marginTop: '0.75rem', background: '#fffbeb', border: '1px solid #f59e0b', borderRadius: 4, padding: '0.75rem' }}>
                        <div style={{ fontWeight: 600, color: '#92400e', fontSize: '0.85rem', marginBottom: '0.35rem' }}>Additional Info Needed</div>
                        {f.additional_info_needed.map((item, i) => <div key={i} style={{ fontSize: '0.82rem', color: '#92400e' }}>• {String(item).replace(/^[•\-\s]+/, '')}</div>)}
                      </div>
                    )}

                    {Array.isArray(f.complications_contraindications) && f.complications_contraindications.length > 0 && (
                      <div style={{ marginTop: '0.75rem', background: '#fff1f2', border: '1px solid #fecdd3', borderRadius: 4, padding: '0.75rem' }}>
                        <div style={{ fontWeight: 600, color: '#9f1239', fontSize: '0.85rem', marginBottom: '0.35rem' }}>Complications / Contraindications</div>
                        {f.complications_contraindications.map((item, i) => <div key={i} style={{ fontSize: '0.82rem', color: '#9f1239' }}>• {String(item).replace(/^[•\-\s]+/, '')}</div>)}
                      </div>
                    )}

                    {String(f.followup_schedule || '').trim() && (
                      <div style={{ marginTop: '0.75rem', background: '#f0fdf4', border: '1px solid #bbf7d0', borderRadius: 4, padding: '0.75rem' }}>
                        <div style={{ fontWeight: 600, color: '#166534', fontSize: '0.85rem', marginBottom: '0.35rem' }}>Follow-Up Schedule</div>
                        <div style={{ fontSize: '0.82rem', color: '#166534' }}>{f.followup_schedule}</div>
                      </div>
                    )}
                  </div>
                </div>
              ))}

              <div style={{ textAlign: 'center', marginTop: '1rem', marginBottom: '1.5rem' }}>
                <button
                  className="btn btn-primary"
                  onClick={handleDownloadPDF}
                  style={{
                    backgroundColor: '#8B0000', borderColor: '#8B0000',
                    padding: '0.75rem 2rem', fontSize: '1rem'
                  }}
                >
                  📄 Download Full PDF Report
                </button>
              </div>
            </>
          )}

          {/* Loading */}
          {reportLoading && (
            <div className="output-container" style={{ textAlign: 'center' }}>
              <div className="spinner" style={{ marginBottom: '1rem' }}></div>
              <p>🔄 Running LLM shunt classification with few-shot examples...</p>
              <p className="text-muted mt-1">Analysing {(() => { try { return JSON.parse(reportClips).length; } catch { return '?'; } })()} clips</p>
            </div>
          )}
        </>
      )}

      {/* Error Display */}
      {error && (
        <div className="output-container" style={{ borderLeft: '4px solid #dc2626' }}>
          <div className="output-header">
            <h3>❌ Error</h3>
          </div>
          <div className="output-content">
            <p className="text-error">{error}</p>
          </div>
        </div>
      )}

      {/* Single Mode Response Display */}
      {mode === 'single' && response && !response.results && (
        <>
          <div className="output-container">
            <div className="output-header">
              <h3>🔬 Clinical Reasoning</h3>
              <span className="output-status success">✓ Complete</span>
            </div>
            <div className="output-content">
              <div style={{ display: 'grid', gap: '0.85rem' }}>
                {response.shunt_type_assessment ? (
                  <div style={{ padding: '1rem', backgroundColor: '#f0fdf4', borderLeft: '4px solid #059669', borderRadius: '0.25rem', whiteSpace: 'pre-wrap', fontFamily: 'system-ui' }}>
                    <div style={{ fontWeight: 600, marginBottom: '0.35rem' }}>Shunt Type Assessment</div>
                    {response.shunt_type_assessment}
                  </div>
                ) : null}

                {response.reasoning ? (
                  <div style={{ padding: '1rem', backgroundColor: '#eff6ff', borderLeft: '4px solid #1e40af', borderRadius: '0.25rem', whiteSpace: 'pre-wrap', fontFamily: 'system-ui' }}>
                    <div style={{ fontWeight: 600, marginBottom: '0.35rem' }}>Reasoning</div>
                    {response.reasoning}
                  </div>
                ) : null}

                {response.treatment_plan ? (
                  <div style={{ padding: '1rem', backgroundColor: '#fef3c7', borderLeft: '4px solid #d97706', borderRadius: '0.25rem', whiteSpace: 'pre-wrap', fontFamily: 'system-ui' }}>
                    <div style={{ fontWeight: 600, marginBottom: '0.35rem' }}>Treatment Plan</div>
                    {response.treatment_plan}
                  </div>
                ) : null}

                {(response.ligation_steps?.length > 0 || response.point_of_ligation || response.chiva_approach || response.clinical_rationale) && (
                  <div style={{ padding: '1rem', backgroundColor: '#fff5f5', borderLeft: '4px solid #8B0000', borderRadius: '0.25rem', whiteSpace: 'pre-wrap', fontFamily: 'system-ui' }}>
                    <div style={{ fontWeight: 600, marginBottom: '0.35rem', color: '#8B0000' }}>Ligation Details</div>
                    {response.point_of_ligation && <div><strong>Point of ligation:</strong> {response.point_of_ligation}</div>}
                    {response.chiva_approach && <div><strong>Approach:</strong> {response.chiva_approach}</div>}
                    {response.ligation_steps?.length > 0 && (
                      <div style={{ marginTop: '0.4rem' }}>
                        <strong>Ligation steps:</strong>
                        <div style={{ marginTop: '0.25rem' }}>
                          {response.ligation_steps.map((step, index) => (
                            <div key={index}>• {String(step).replace(/^[•\-\s]+/, '')}</div>
                          ))}
                        </div>
                      </div>
                    )}
                    {response.clinical_rationale && <div style={{ marginTop: '0.4rem' }}><strong>Clinical rationale:</strong> {response.clinical_rationale}</div>}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Raw Response */}
          <details style={{ marginTop: '2rem' }}>
            <summary style={{ cursor: 'pointer', fontWeight: '600', color: '#666' }}>
              Show Raw LLM Response
            </summary>
            <div style={{
              marginTop: '1rem',
              padding: '1rem',
              backgroundColor: '#f3f4f6',
              borderRadius: '0.5rem',
              whiteSpace: 'pre-wrap',
              fontFamily: 'Monaco, monospace',
              fontSize: '0.85rem',
              overflow: 'auto',
              maxHeight: '400px'
            }}>
              {response.raw_response}
            </div>
          </details>
        </>
      )}

      {/* Stream Mode Response Display - LIVE UPDATES */}
      {mode === 'stream' && response && response.processed_points && (
        <>
          <div className="output-container">
            <div className="output-header">
              <h3>📊 Stream Processing - Live Results</h3>
              <span className="output-status success">✓ Processing: {response.processed_points.length}/{response.total_points} points</span>
            </div>
            <div className="output-content">
              <p><strong>Total Data Points:</strong> {response.total_points}</p>
              <p><strong>Processed:</strong> {response.processed_points.length} points</p>
              <p><strong>Buffer Interval:</strong> {bufferInterval}s between points</p>
              {analysisTime && <p><strong>Total Time:</strong> {analysisTime}s</p>}
            </div>
          </div>

          {/* Current Latest Result (LATEST POINT) */}
          <div className="output-container">
            <div className="output-header">
              <h3>⏱️ Latest Result</h3>
              <span className="output-status success">✓ Active</span>
            </div>
            <div className="output-content">
              {response.processed_points.length > 0 && (
                <>
                  <div style={{
                    padding: '0.75rem',
                    backgroundColor: '#f0fdf4',
                    borderLeft: '4px solid #059669',
                    borderRadius: '0.25rem',
                    marginBottom: '1rem',
                    fontSize: '0.9rem'
                  }}>
                    <div><strong>Point #</strong> {response.processed_points[response.processed_points.length - 1].sequence_number}</div>
                    <div><strong>Location:</strong> {response.processed_points[response.processed_points.length - 1].location}</div>
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Shunt Type Assessment - LATEST */}
          <div className="output-container">
            <div className="output-header">
              <h3>🔍 Shunt Type Assessment (Latest)</h3>
              <span className="output-status success">✓ Complete</span>
            </div>
            <div className="output-content">
              {response.processed_points.length > 0 && response.processed_points[response.processed_points.length - 1].flow_type === 'RP' ? (
                response.current_assessment ? (
                  <div style={{
                    padding: '1rem',
                    backgroundColor: '#f0fdf4',
                    borderLeft: '4px solid #059669',
                    borderRadius: '0.25rem',
                    whiteSpace: 'pre-wrap',
                    fontFamily: 'system-ui'
                  }}>
                    {response.current_assessment}
                  </div>
                ) : (
                  <p className="text-muted">Processing...</p>
                )
              ) : (
                <p className="text-muted">✓ No abnormal flow detected - sections remain empty</p>
              )}
            </div>
          </div>

          {/* Reasoning - LATEST */}
          <div className="output-container">
            <div className="output-header">
              <h3>💭 Reasoning (Latest Point)</h3>
              <span className="output-status success">✓ Complete</span>
            </div>
            <div className="output-content">
              {response.processed_points.length > 0 && response.processed_points[response.processed_points.length - 1].flow_type === 'RP' ? (
                response.current_reasoning ? (
                  <div style={{
                    padding: '1rem',
                    backgroundColor: '#eff6ff',
                    borderLeft: '4px solid #1e40af',
                    borderRadius: '0.25rem',
                    whiteSpace: 'pre-wrap',
                    fontFamily: 'system-ui'
                  }}>
                    {response.current_reasoning}
                  </div>
                ) : (
                  <p className="text-muted">Processing...</p>
                )
              ) : (
                <p className="text-muted">✓ No abnormal flow detected - sections remain empty</p>
              )}
            </div>
          </div>

          {/* Ligation - CHIVA SURGICAL PROCEDURE */}
          <div className="output-container">
            <div className="output-header">
              <h3>⚕️ Ligation (Latest Point)</h3>
              <span className="output-status success">✓ CHIVA Surgical Strategy</span>
            </div>
            <div className="output-content">
              {response.processed_points.length > 0 && response.processed_points[response.processed_points.length - 1].flow_type === 'RP' ? (
                response.current_treatment ? (
                  <div style={{
                    padding: '1rem',
                    backgroundColor: '#fdf2f8',
                    borderLeft: '4px solid #8b5cf6',
                    borderRadius: '0.25rem',
                    whiteSpace: 'pre-wrap',
                    fontFamily: 'system-ui'
                  }}>
                    {response.current_treatment}
                  </div>
                ) : (
                  <p className="text-muted">Processing surgical ligation procedures...</p>
                )
              ) : (
                <p className="text-muted">✓ No abnormal flow detected - ligation not required</p>
              )}
            </div>
          </div>

          {/* RP Classifications Summary */}
          {response.shunt_classifications && response.shunt_classifications.length > 0 && (
            <div className="output-container" style={{ borderLeft: '4px solid #C01C1C' }}>
              <div className="output-header"><h3>🩺 RP Clip Assessments ({response.shunt_classifications.length} clips)</h3></div>
              <div className="output-content">
                {response.shunt_classifications.map((s, idx) => (
                  <div key={idx} style={{ padding: '0.5rem 0', borderBottom: '1px solid #f0f0f0', fontSize: '0.9rem' }}>
                    <strong>#{s.sequence}</strong> {s.flow} at {s.step} →&nbsp;
                    <span style={{ color: '#C01C1C', fontWeight: 600 }}>{s.assessment || '...'}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* All Processed Points Summary */}
          <details style={{ marginTop: '2rem' }}>
            <summary style={{ cursor: 'pointer', fontWeight: '600', color: '#666' }}>
              📋 Show All {response.processed_points.length} Processed Data Points
            </summary>
            <div style={{ marginTop: '1rem' }}>
              {response.processed_points.map((point, idx) => (
                <div key={idx} style={{
                  padding: '0.75rem 1rem',
                  marginBottom: '0.5rem',
                  backgroundColor: point.flow_type === 'RP' ? '#fff5f5' : '#f9fafb',
                  border: `1px solid ${point.flow_type === 'RP' ? '#fecaca' : '#e5e7eb'}`,
                  borderRadius: '0.375rem',
                  fontSize: '0.88rem'
                }}>
                  <div>
                    <strong>#{point.point_number}</strong>
                    &nbsp;<span style={{ color: point.flow_type === 'RP' ? '#dc2626' : '#059669', fontWeight: 600 }}>{point.flow_type}</span>
                    &nbsp;· {point.location} · {point.reflux_duration != null ? `${point.reflux_duration}s` : ''}
                  </div>
                  <div style={{ color: '#555', marginTop: '0.2rem' }}>{point.description}</div>
                  {point.error && <div style={{ color: '#dc2626', marginTop: '0.25rem' }}><strong>Error:</strong> {point.error}</div>}
                </div>
              ))}
            </div>
          </details>
        </>
      )}

      {mode === 'accuracy' && (
        <div className="section-content">
        <p className="text-muted" style={{ marginBottom: '1.25rem', fontSize: '0.9rem' }}>
          10 streams (3 real clinical sessions + 7 synthesised) each with a defined ground-truth shunt type.
          Classify each stream with the LLM and compare against ground truth to measure model accuracy.
          Streams 1–3 are real data; Streams 4–10 are anatomically-grounded synthetic data.
        </p>

        {/* Stream list */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginBottom: '1.5rem' }}>
          {ACCURACY_TEST_STREAMS.map((stream) => {
            const result = accStreamResults[stream.id];
            const isLoading = accStreamLoading[stream.id];
            const rpCount = stream.clips.filter(c => c.flow === 'RP').length;
            const epCount = stream.clips.filter(c => c.flow === 'EP').length;

            return (
              <div key={stream.id} style={{
                border: result
                  ? (result.isCorrect ? '2px solid #10b981' : result.error ? '2px solid #f87171' : '2px solid #f59e0b')
                  : '2px solid #e5e7eb',
                borderRadius: '10px', overflow: 'hidden',
                background: result?.isCorrect ? '#f0fdf4' : result?.error ? '#fef2f2' : result ? '#fffbeb' : '#fff',
              }}>
                {/* Stream header row */}
                <div style={{ padding: '0.75rem 1rem', display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
                  {/* Name + source badge */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', minWidth: '100px' }}>
                    <span style={{ fontWeight: 700, fontSize: '0.95rem', color: '#1f2937' }}>{stream.name}</span>
                    <span style={{
                      fontSize: '0.68rem', padding: '1px 6px', borderRadius: '999px', fontWeight: 600,
                      background: stream.source === 'real' ? '#dbeafe' : '#f3e8ff',
                      color: stream.source === 'real' ? '#1e40af' : '#6b21a8'
                    }}>
                      {stream.source === 'real' ? '🏥 Real' : '🔬 Synthetic'}
                    </span>
                  </div>
                  {/* Description */}
                  <span style={{ fontSize: '0.8rem', color: '#6b7280', flex: 1 }}>{stream.description}</span>
                  {/* Clip stats */}
                  <div style={{ display: 'flex', gap: '0.4rem', fontSize: '0.75rem' }}>
                    <span style={{ background: '#f3f4f6', borderRadius: '4px', padding: '2px 6px', color: '#374151' }}>
                      {stream.clips.length} clips
                    </span>
                    <span style={{ background: '#fee2e2', borderRadius: '4px', padding: '2px 6px', color: '#991b1b' }}>
                      {rpCount} RP
                    </span>
                    <span style={{ background: '#d1fae5', borderRadius: '4px', padding: '2px 6px', color: '#065f46' }}>
                      {epCount} EP
                    </span>
                  </div>
                  {/* Ground truth */}
                  <div style={{ fontSize: '0.78rem', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                    <span style={{ color: '#888' }}>GT:</span>
                    <span style={{ fontWeight: 700, color: '#059669', background: '#d1fae5', borderRadius: '4px', padding: '1px 8px' }}>
                      {stream.groundTruth}
                    </span>
                  </div>
                  {/* Classify button */}
                  <button
                    className="btn btn-primary"
                    onClick={() => handleAccuracyClassify(stream)}
                    disabled={isLoading}
                    style={{ fontSize: '0.82rem', padding: '0.35rem 0.9rem', whiteSpace: 'nowrap', background: '#C01C1C', border: 'none' }}
                  >
                    {isLoading ? '⏳ Classifying…' : result ? '🔄 Re-Classify' : '▶ Classify'}
                  </button>
                  {/* Expand toggle */}
                  {result && !result.error && (
                    <button
                      onClick={() => setAccExpandedStream(accExpandedStream === stream.id ? null : stream.id)}
                      style={{ background: 'none', border: '1px solid #d1d5db', borderRadius: '6px', cursor: 'pointer', fontSize: '0.78rem', padding: '0.3rem 0.6rem', color: '#555' }}
                    >
                      {accExpandedStream === stream.id ? '▲ Hide' : '▼ Details'}
                    </button>
                  )}
                </div>

                {/* Result row */}
                {result && (
                  <div style={{ borderTop: '1px solid #e5e7eb', padding: '0.6rem 1rem', display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap', background: 'rgba(0,0,0,0.02)' }}>
                    {result.error ? (
                      <span style={{ color: '#dc2626', fontSize: '0.82rem' }}>❌ Error: {result.error}</span>
                    ) : (
                      <>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                          <span style={{ fontSize: '0.78rem', color: '#888' }}>LLM:</span>
                          <span style={{ fontWeight: 700, fontSize: '0.85rem', color: '#C01C1C', background: '#fdf2f2', borderRadius: '4px', padding: '1px 8px' }}>
                            {result.llmLabel}
                          </span>
                        </div>
                        <span style={{ fontWeight: 800, fontSize: '1rem', color: result.isCorrect ? '#059669' : '#dc2626' }}>
                          {result.isCorrect ? '✅ Correct' : '❌ Incorrect'}
                        </span>
                        {result.confidence != null && (
                          <span style={{ fontSize: '0.78rem', color: '#6b7280' }}>
                            Confidence: {(result.confidence * 100).toFixed(0)}%
                          </span>
                        )}
                        {result.total_tokens > 0 && (
                          <span style={{ fontSize: '0.75rem', color: '#059669', fontFamily: 'monospace' }}>
                            📊 {result.total_tokens.toLocaleString()} tokens
                          </span>
                        )}
                        <span style={{ fontSize: '0.75rem', color: '#9ca3af', marginLeft: 'auto' }}>
                          {new Date(result.timestamp).toLocaleTimeString()} · {result.duration_sec}s
                        </span>
                      </>
                    )}
                  </div>
                )}

                {/* Expanded detail */}
                {result && !result.error && accExpandedStream === stream.id && (
                  <div style={{ borderTop: '1px solid #e5e7eb', padding: '0.75rem 1rem', background: '#fafafa' }}>
                    {result.findings?.map((f, i) => (
                      <div key={i} style={{ marginBottom: '0.5rem', fontSize: '0.82rem' }}>
                        <span style={{ fontWeight: 600, color: '#374151' }}>{f.leg || 'Finding'}: </span>
                        <span style={{ color: '#C01C1C', fontWeight: 700 }}>{f.shunt_type}</span>
                        {f.confidence != null && <span style={{ color: '#6b7280', marginLeft: '0.5rem' }}>({(f.confidence * 100).toFixed(0)}% confidence)</span>}
                        {f.reasoning?.length > 0 && (
                          <div style={{ marginTop: '0.25rem', color: '#555', paddingLeft: '1rem', borderLeft: '2px solid #e5e7eb' }}>
                            {f.reasoning.map((r, j) => <div key={j}>• {r}</div>)}
                          </div>
                        )}
                        {(normalizeLigationSteps(f).length > 0 || getPointOfLigation(f) || getLigationApproach(f) || f.clinical_rationale) && (
                          <div style={{ marginTop: '0.35rem', paddingLeft: '1rem', borderLeft: '2px solid #fca5a5' }}>
                            {getPointOfLigation(f) && (
                              <div style={{ color: '#7f1d1d' }}><span style={{ fontWeight: 600 }}>Point of ligation:</span> {getPointOfLigation(f)}</div>
                            )}
                            {getLigationApproach(f) && (
                              <div style={{ color: '#7f1d1d' }}><span style={{ fontWeight: 600 }}>Approach:</span> {getLigationApproach(f)}</div>
                            )}
                            {normalizeLigationSteps(f).length > 0 && (
                              <div style={{ color: '#7f1d1d' }}>
                                <span style={{ fontWeight: 600 }}>Ligation steps:</span>
                                <div style={{ marginTop: '0.2rem' }}>
                                  {normalizeLigationSteps(f).map((step, j) => <div key={j}>• {step}</div>)}
                                </div>
                              </div>
                            )}
                            {String(f.clinical_rationale || '').trim() && (
                              <div style={{ color: '#7f1d1d' }}><span style={{ fontWeight: 600 }}>Clinical rationale:</span> {f.clinical_rationale}</div>
                            )}
                          </div>
                        )}
                      </div>
                    ))}
                    {(result.ligation_steps?.length > 0 || result.ligation?.length > 0) && (
                      <div style={{ marginTop: '0.5rem', fontSize: '0.82rem' }}>
                        <span style={{ fontWeight: 600, color: '#374151' }}>Proposed Ligation: </span>
                        <span style={{ color: '#555' }}>{(result.ligation_steps || result.ligation).join(', ')}</span>
                      </div>
                    )}
                    {String(result.point_of_ligation || '').trim() && (
                      <div style={{ marginTop: '0.5rem', fontSize: '0.82rem' }}>
                        <span style={{ fontWeight: 600, color: '#374151' }}>Point of Ligation: </span>
                        <span style={{ color: '#555' }}>{result.point_of_ligation}</span>
                      </div>
                    )}
                    {String(result.chiva_approach || '').trim() && (
                      <div style={{ marginTop: '0.5rem', fontSize: '0.82rem' }}>
                        <span style={{ fontWeight: 600, color: '#374151' }}>Approach: </span>
                        <span style={{ color: '#555' }}>{result.chiva_approach}</span>
                      </div>
                    )}
                    {String(result.clinical_rationale || '').trim() && (
                      <div style={{ marginTop: '0.5rem', fontSize: '0.82rem' }}>
                        <span style={{ fontWeight: 600, color: '#374151' }}>Clinical Rationale: </span>
                        <span style={{ color: '#555' }}>{result.clinical_rationale}</span>
                      </div>
                    )}
                    {result.total_tokens > 0 && (
                      <div style={{ marginTop: '0.5rem', fontSize: '0.78rem', color: '#059669', fontFamily: 'monospace', background: '#f0fdf4', padding: '0.4rem 0.6rem', borderRadius: '4px' }}>
                        📊 Tokens: {result.total_prompt_tokens?.toLocaleString() || 0} prompt + {result.total_completion_tokens?.toLocaleString() || 0} completion = {result.total_tokens?.toLocaleString() || 0} total
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Overall Accuracy Button */}
        {(() => {
          const classified = ACCURACY_TEST_STREAMS.filter(s => accStreamResults[s.id] && !accStreamResults[s.id].error);
          const correct = classified.filter(s => accStreamResults[s.id].isCorrect).length;
          const total = classified.length;
          const overallPct = total > 0 ? ((correct / total) * 100).toFixed(1) : null;

          return (
            <div style={{ textAlign: 'center', marginTop: '0.5rem' }}>
              <div style={{
                display: 'inline-block', padding: '1.5rem 2.5rem',
                border: '2px solid #C01C1C', borderRadius: '12px',
                background: overallPct !== null ? (parseFloat(overallPct) >= 70 ? '#f0fdf4' : parseFloat(overallPct) >= 40 ? '#fffbeb' : '#fef2f2') : '#f9fafb',
                minWidth: '320px',
              }}>
                <div style={{ fontSize: '0.75rem', color: '#888', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.4rem' }}>
                  Overall LLM Accuracy — {total}/{ACCURACY_TEST_STREAMS.length} streams classified
                </div>
                {overallPct !== null ? (
                  <>
                    <div style={{ fontSize: '3rem', fontWeight: 900, color: parseFloat(overallPct) >= 70 ? '#059669' : parseFloat(overallPct) >= 40 ? '#d97706' : '#dc2626', lineHeight: 1 }}>
                      {overallPct}%
                    </div>
                    <div style={{ fontSize: '0.85rem', color: '#555', marginTop: '0.4rem' }}>
                      {correct} correct out of {total} classified streams
                    </div>
                    <div style={{ fontSize: '0.75rem', color: '#888', marginTop: '0.3rem' }}>
                      (Uses most recent classification run per stream)
                    </div>
                  </>
                ) : (
                  <div style={{ color: '#aaa', fontSize: '0.9rem' }}>Classify at least one stream to see overall accuracy</div>
                )}
                <button
                  className="btn btn-primary"
                  style={{ marginTop: '1rem', background: '#C01C1C', border: 'none', fontSize: '0.9rem', padding: '0.5rem 1.5rem' }}
                  onClick={() => {
                    // Trigger classify on all unclassified streams
                    ACCURACY_TEST_STREAMS.forEach(s => {
                      if (!accStreamResults[s.id] && !accStreamLoading[s.id]) {
                        handleAccuracyClassify(s);
                      }
                    });
                  } }
                >
                  🚀 Get Overall Accuracy (Classify All Remaining)
                </button>
              </div>
            </div>
          );
        })()}
        </div>
      )}
    </div></>
  );
};

export default ClinicalReasoning;