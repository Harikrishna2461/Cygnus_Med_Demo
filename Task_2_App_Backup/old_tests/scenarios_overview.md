# CHIVA Streaming Guidance — Test Scenarios Overview

Run one:   `python tests/run_stream_scenario.py type1`
Run all:   `python tests/run_stream_scenario.py all`

---

## Scenario Summary

| ID      | Shunt Type  | Clip Sequence                                          | Key Tests                                      |
|---------|-------------|--------------------------------------------------------|------------------------------------------------|
| type1   | Type 1      | EP N1→N2 (SFJ) + RP N2→N1                             | Basic navigation; complete fires correctly     |
| type2a  | Type 2A     | EP N1→N2 (SPJ, posY=0.47 posterior) + RP N2→N1 (SSV)  | Posterior/SPJ navigation; complete at SPJ circuit |
| type3   | Type 3      | EP N1→N2 + RP N2→N1 + EP N2→N3 + RP N3→N1 → No Reflux| maneuver fires; No Reflux → complete           |
| type1p2 | Type 1+2    | EP N1→N2 + RP N2→N1 + EP N2→N3 + RP N3→N1 → Reflux   | maneuver fires; Reflux → complete              |
| type4   | Type 4      | EP N1→N3 (calf perf) + RP N2→N1 (trunk)               | Perforator-direct path; no SFJ entry           |
| type6   | Type 6      | EP N1→N3 + RP N3→N1 (no trunk clips at all)            | Pure N3 circuit; no trunk navigation needed    |

---

## What Each Scenario Specifically Validates

### type1 — Baseline Type 1
- At SFJ with no clips: guidance references SFJ anatomy
- After EP N1→N2: guidance directs distally to GSV trunk
- At UPPER_THIGH: guidance asks for trunk reflux assessment
- After RP N2→N1: guidance directs to Hunterian/tributaries
- At POPLITEAL: guidance checks SPJ competence
- **Critical**: After EP+RP, no EP N2→N3 → `action="complete"` fires
- **Critical**: `action="maneuver"` never fires (no EP N2→N3 in list)

### type2a — SPJ-Based (Type 2A)
- SFJ visited but competent (no clip) → guidance still progresses
- At POPLITEAL posterior with no clips: guidance assesses SPJ
- After EP N1→N2 at SPJ (posY=0.47): guidance directs SSV distally (not GSV)
- **Critical**: EP at posY=0.47 + RP at posY=0.62 → `action="complete"` fires
  (verifies the model doesn't require SFJ posY for complete)

### type3 — Tributary Loop (No Reflux)
- After EP N2→N3 alone (no RP N3→N1 yet): `action="maneuver"` must NOT fire
- After RP N3→N1 added: `action="maneuver"` MUST fire (all 3 clips + no elimTest)
- **Critical**: After `elimination_test="No Reflux"` on EP N2→N3 → `action="complete"`

### type1p2 — Combined Shunt (Reflux)
- Same setup as type3 (same clip sequence triggers maneuver)
- **Critical**: After `elimination_test="Reflux"` on EP N2→N3 → `action="complete"`
- (Distinguishes from Type 3 via the Reflux result alone)

### type4 — Perforator Entry
- SFJ and SPJ both competent (no clips from either)
- EP N1→N3 confirmed at calf: guidance looks for RP
- **Critical**: EP N1→N3 + RP N2→N1 → `action="complete"`

### type6 — Pure Perforator Circuit
- SFJ and SPJ both competent (no clips)
- EP N1→N3 confirmed at calf lateral
- **Critical**: EP N1→N3 + RP N3→N1 (NO RP N2→N1) → `action="complete"`
  (verifies model doesn't require trunk clips for Type 6)

---

## Pass/Fail Criteria by Feature

| Feature                              | Tested By          |
|--------------------------------------|--------------------|
| SFJ navigation at Q1                 | type1, type2a      |
| SPJ/posterior navigation             | type2a             |
| Distal GSV trunk tracing after EP    | type1, type3       |
| Tributary escape detection (Q3)      | type3, type1p2     |
| Maneuver trigger (3 clips + no elim) | type3, type1p2     |
| No premature maneuver (EP N2→N3 only)| type3              |
| No Reflux → Type 3 complete          | type3              |
| Reflux → Type 1+2 complete           | type1p2            |
| Type 1 complete (EP+RP, no N2→N3)    | type1              |
| SPJ-circuit complete (posY 0.47)     | type2a             |
| Perforator-direct complete           | type4              |
| Pure N3 circuit complete (no trunk)  | type6              |
