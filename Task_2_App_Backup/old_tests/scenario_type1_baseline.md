# CHIVA Type 1 Shunt — Surgeon Scenario Baseline

## Purpose
Ground-truth walkthrough of a Type 1 CHIVA assessment. Used to test whether the
streaming guidance engine (Task-2) navigates the surgeon correctly. Each step
documents what the surgeon does, what they find, what event to emit, and what
the AI guidance response must contain to be considered correct.

---

## Patient Profile
- Age/sex: 65F, right leg
- Presentation: Symptomatic varicosities medial thigh and calf, CEAP C3-C4
- Pre-exam clinical test: Tourniquet (Brodie-Trendelenburg) positive — varicosities
  fill from above, collapse when SFJ compressed by finger
- Impression going in: SFJ incompetence with GSV reflux

---

## Type 1 Shunt Definition (ground truth)
- **EP N1→N2 at SFJ**: deep system (CFV) refluxes into GSV at groin
- **RP N2→N1 along GSV trunk**: retrograde flow down the saphenous trunk
- **No EP N2→N3**: no escape from trunk into tributaries at any level
- **SPJ competent**: no SSV contribution
- **Ligation target**: flush SFJ high ligation ± trunk stripping

Minimum clip set to trigger `action: "complete"`:
`EP N1→N2 (posY 0.04–0.09)` + `RP N2→N1 (any posY)` with no EP N2→N3 present

---

## Timestamp Scenario

---

### T=00:00 — Patient Positioning

**What surgeon does:**
Patient standing, 30-degree reverse Trendelenburg on examination table. Right leg
non-weight-bearing (slight knee bend, calf relaxed). Duplex machine in colour/PW
Doppler mode. Acoustic coupling gel applied to medial thigh and groin.

**Clinical intent:** Allow venous distension so reflux is more easily provoked.

**Socket event (stream_start):**
```json
{ "session_id": "test_type1" }
```

**Expected server response:**
```json
{ "session_id": "test_type1" }
```
Event: `session_ready`

---

### T=00:30 — Probe at SFJ

**What surgeon does:**
Probe placed transversely in groin crease. Mickey Mouse sign identified:
- Centre: common femoral vein (CFV) — compressible, non-pulsatile
- Medial "ear": great saphenous vein (GSV) — 8 mm diameter (dilated)
- Lateral "ear": femoral artery — pulsatile

Switch to longitudinal. Sample volume placed in GSV lumen immediately distal
to terminal valve (on the superficial/saphenous side of the junction). No
spontaneous colour signal at rest.

**Socket event (probe_move):**
```json
{
  "session_id": "test_type1",
  "region": "SFJ",
  "pos_y_ratio": 0.06,
  "surface": "anterior-medial",
  "leg": "right"
}
```

**Expected AI guidance (action: "move"):**
Must instruct surgeon to assess reflux at this junction using Valsalva and
squeeze maneuvers. Acceptable phrasings:
- "Scan SFJ transversely — confirm Mickey Mouse sign then assess with Valsalva"
- "At SFJ — apply Valsalva maneuver to provoke reflux"
- Any instruction directing attention to the SFJ anatomy

**Pass if:** guidance mentions SFJ, groin, or femoral junction. Action = "move".
**Fail if:** guidance says to move away from SFJ before confirming EP.

---

### T=01:00 — Valsalva Maneuver

**What surgeon finds:**
Patient strains (Valsalva) for 3 seconds. Doppler: audible reflux signal >0.8 sec
in GSV lumen. Pre-terminal valve incompetent — reflux crosses junction.

**Surgeon interpretation:** Positive Valsalva → sustained reflux across terminal
valve → deep blood entering GSV. One of two required provocations positive.

**Socket event (probe_move — same position, no change):**
```json
{
  "session_id": "test_type1",
  "region": "SFJ",
  "pos_y_ratio": 0.06,
  "surface": "anterior-medial",
  "leg": "right"
}
```
*(Re-emitted because probe has not moved; guidance should still direct toward
confirming both maneuvers.)*

---

### T=01:30 — Thigh Squeeze (Augmentation)

**What surgeon finds:**
Surgeon firmly squeezes distal thigh, then releases. Doppler: reflux on release
lasting >1.0 sec → anterograde venous pump fills GSV; on release, blood refluxes
back in an EP pattern. Both Valsalva AND squeeze positive → EP N1→N2 confirmed.

**Surgeon decision:** SFJ incompetent. EP is at SFJ. Clip ready to mark.

---

### T=02:00 — MARK CLIP: EP N1→N2 at SFJ

**What surgeon does:**
Records the finding using the UI clip form.

**Socket event (clip_mark):**
```json
{
  "session_id": "test_type1",
  "flow": "EP",
  "from_type": "N1",
  "to_type": "N2",
  "pos_y_ratio": 0.06,
  "leg": "right",
  "region": "SFJ",
  "surface": "anterior-medial",
  "elimination_test": ""
}
```

**Expected AI guidance after clip_mark (action: "move"):**
EP N1→N2 at SFJ is now confirmed. Q1 answered. Q2 is now open: does GSV trunk
carry reflux distally? Must direct probe DISTALLY along medial thigh.

Acceptable phrasings:
- "Move distally along medial thigh to trace GSV trunk"
- "Scan medial thigh distally — confirm trunk reflux below SFJ"

**Pass if:** guidance says to move distally toward thigh / GSV trunk.
**Fail if:** guidance stays at SFJ or moves to wrong surface.

---

### T=02:30 — Move to Upper Thigh

**What surgeon does:**
Probe moved distally along medial thigh. Transverse view first to confirm GSV
inside saphenous eye (fascial envelope visible as bright triangle containing vessel).
GSV 7.5 mm, clearly within saphenous compartment — confirmed N2, not AASV.

**Socket event (probe_move):**
```json
{
  "session_id": "test_type1",
  "region": "UPPER_THIGH",
  "pos_y_ratio": 0.18,
  "surface": "medial",
  "leg": "right"
}
```

**Expected AI guidance (action: "move"):**
Probe is now in thigh GSV zone with one EP confirmed. Q2 open. Must direct
surgeon to assess for retrograde flow (RP N2→N1) here.

Acceptable phrasings:
- "Confirm GSV trunk reflux — apply distal squeeze to provoke retrograde flow"
- "Check trunk Doppler for retrograde flow in saphenous eye"
- "Assess for retrograde flow along medial thigh GSV"

**Pass if:** guidance references trunk reflux assessment or squeeze provocation
at this level. Action = "move".

---

### T=03:00 — Doppler at Upper Thigh

**What surgeon finds:**
Foot/calf squeeze applied, then released. Colour Doppler: brief forward
(anterograde) signal on squeeze; on release, sustained retrograde signal >1.5 sec
filling from proximal to distal (downward in standing patient). RP N2→N1 confirmed
in GSV trunk. No tributaries visible branching from GSV at this level.

**Socket event (probe_move — same position):**
```json
{
  "session_id": "test_type1",
  "region": "UPPER_THIGH",
  "pos_y_ratio": 0.18,
  "surface": "medial",
  "leg": "right"
}
```

---

### T=03:30 — Brief AASV Check

**What surgeon does:**
Rotate probe ~10-15 degrees anterior-lateral at posY ~0.12. AASV lies parallel
and anterior to GSV.

**What surgeon finds:**
AASV small caliber (~2 mm), no colour Doppler signal on provocation. Not involved.
Return to medial surface.

**Clinical significance:** Rules out AASV as an additional EP (would complicate
Type 1 into a Type 1+2 if AASV fed via a separate SFJ tributary).

---

### T=04:00 — MARK CLIP: RP N2→N1 at Upper Thigh

**Socket event (clip_mark):**
```json
{
  "session_id": "test_type1",
  "flow": "RP",
  "from_type": "N2",
  "to_type": "N1",
  "pos_y_ratio": 0.18,
  "leg": "right",
  "region": "UPPER_THIGH",
  "surface": "medial",
  "elimination_test": ""
}
```

**Expected AI guidance after clip_mark (action: "move"):**
Q2 is now partially answered (RP confirmed in upper thigh). Must trace distally
to determine: (a) how far down does the reflux extend, and (b) is there any
EP N2→N3 escape point? Direct probe to Hunterian zone.

Acceptable phrasings:
- "Move distally to mid-thigh Hunterian zone — check for escape tributaries"
- "Continue distally along medial thigh — look for perforators or tributaries"
- "Track GSV distally toward Hunterian region"

**Pass if:** guidance directs distally toward Hunterian or mid-thigh.

---

### T=04:30 — Move to Hunterian Zone

**What surgeon does:**
Probe moved to medial mid-thigh, Hunterian perforator zone.

**Socket event (probe_move):**
```json
{
  "session_id": "test_type1",
  "region": "HUNTERIAN",
  "pos_y_ratio": 0.28,
  "surface": "medial",
  "leg": "right"
}
```

**Expected AI guidance (action: "move"):**
Now in Hunterian zone. EP N1→N2 confirmed at SFJ, RP N2→N1 confirmed in
upper thigh. Q3 is open: does blood escape into tributaries here? Check for
perforators and tributaries branching from GSV at this level.

Acceptable phrasings:
- "Assess for perforators or tributaries at Hunterian level"
- "Check medial mid-thigh for escape tributaries from GSV trunk"
- "Scan Hunterian zone for N2→N3 escape points"

**Pass if:** guidance references checking for tributaries or escape points. Does
NOT direct surgeon back toward SFJ or to posterior surface.

---

### T=05:30 — Perforator Assessment at Hunterian

**What surgeon finds:**
One small perforator visible at posY ~0.29, diameter ~2.3 mm. Doppler on Paranà
maneuver: flow direction is INWARD during diastole (toward deep system). This is
a RE-ENTRY perforator (RP pattern, N2→N1 directional), not an escape EP.
Diameter <3.5 mm, outward flow <500 ms — below pathological threshold.

**Clinical interpretation:** In Type 1, the retrograde GSV trunk may "force"
blood into Hunterian perforators as a re-entry point. This is expected and does
NOT represent an additional EP or escape point. The perforator here is a
consequence of the trunk reflux, not an independent entry.

**What surgeon does NOT mark:**
No clip for this perforator (it's a re-entry RP type and surgeon is not adding
redundant RP clips; the upper thigh RP already establishes trunk reflux is present).

---

### T=06:00 — Check for Tributaries at Hunterian

**What surgeon finds:**
Rotate to scan slightly anterior and posterior while tracking GSV. No tributaries
visible branching from GSV. No colour Doppler signal in any adjacent superficial
structures. GSV reflux pattern continues unchanged.

**Critical finding:** No EP N2→N3 at Hunterian level → no escape here.

---

### T=06:30 — Move to Distal Thigh / Upper Popliteal Approach

**What surgeon does:**
Continue distally along medial thigh, cross-referencing visible varicosities
with duplex to ensure none are fed by a tributary escape from the GSV trunk.

**Socket event (probe_move):**
```json
{
  "session_id": "test_type1",
  "region": "MID_THIGH",
  "pos_y_ratio": 0.36,
  "surface": "medial",
  "leg": "right"
}
```

**Expected AI guidance (action: "move"):**
EP N1→N2 at SFJ + RP N2→N1 in upper thigh → two clips confirmed, no EP N2→N3
found yet. Still need to confirm SPJ is competent to rule out any SSV contribution.
Should direct to posterior/popliteal zone or continue distal thigh scan.

Acceptable phrasings:
- "Scan posteriorly toward popliteal fossa to assess SPJ competence"
- "Continue distal checking for escape tributaries before assessing SPJ"
- Any instruction progressing distally or toward popliteal

**Pass if:** guidance does NOT fire `action: "complete"` here (two clips present
but EP N2→N3 absence has not been confirmed distally, circuit is not yet verified).

---

### T=07:00 — Patient Turns: Move to Popliteal/SPJ

**What surgeon does:**
Patient repositions to expose posterior knee. Probe placed in popliteal fossa.

**Socket event (probe_move):**
```json
{
  "session_id": "test_type1",
  "region": "POPLITEAL",
  "pos_y_ratio": 0.47,
  "surface": "posterior",
  "leg": "right"
}
```

**Expected AI guidance (action: "move"):**
Popliteal/SPJ zone. With EP N1→N2 at SFJ already confirmed, the SPJ check
rules out a second EP via the SSV. Must assess SPJ competence.

Acceptable phrasings:
- "Assess SPJ at popliteal fossa — apply Paranà and compression to test competence"
- "Scan posteriorly at knee level for SSV junction competence"
- "Confirm SPJ with Paranà maneuver at popliteal fossa"

**Pass if:** guidance references SPJ or popliteal assessment.

---

### T=07:30 — SPJ Competence Test

**What surgeon finds:**
SSV visible entering popliteal vein. Paranà maneuver applied: brief antegrade
forward signal, NO reflux on release. Thigh compression applied: NO retrograde
signal in SSV. Both tests negative → SPJ terminal valve COMPETENT.

**Clinical interpretation:** SPJ is competent. No SSV circuit. This rules out:
- Type 2A (SSV EP + GSV reflux)
- Any combined SSV contribution to the varicosities

**What surgeon does NOT mark:** No clip for SPJ (competent junction → not an EP).

---

### T=08:00 — Giacomini Vein Check

**What surgeon does:**
From popliteal fossa, scan posteriorly and proximally in the posterior thigh for
the Giacomini vein (connects SSV to GSV posteriorly).

**What surgeon finds:**
No Giacomini vein identified. Not visible on B-mode, no Doppler signal. This is
a common Type 1 finding — without SSV reflux there is nothing to drive a Giacomini.

---

### T=08:30 — Confirm No Mid-Thigh Escape (Final Check)

**What surgeon does:**
Returns to medial surface, scans posY 0.30–0.40 looking for any small tributaries.

**Socket event (probe_move):**
```json
{
  "session_id": "test_type1",
  "region": "MID_THIGH",
  "pos_y_ratio": 0.35,
  "surface": "medial",
  "leg": "right"
}
```

**What surgeon finds:**
No tributaries with EP N2→N3 signal at any level from 0.15 to 0.40.
GSV reflux throughout thigh confirmed as RP N2→N1 only — no escape point found.

---

### T=09:00 — Classification Summary

**Surgeon's ground-truth reasoning:**
| Finding                  | Result      | Implication                        |
|--------------------------|-------------|------------------------------------|
| EP at SFJ                | CONFIRMED   | N1→N2 entry point identified       |
| GSV trunk reflux         | CONFIRMED   | RP N2→N1 throughout upper/mid thigh|
| Trunk escape EP N2→N3    | NOT FOUND   | No tributary circuit                |
| SPJ competence           | CONFIRMED   | No SSV contribution                |
| Giacomini vein           | NOT PRESENT | No posterior thigh circuit          |
| AASV                     | NOT INVOLVED| No secondary SFJ tributary circuit  |

**Shunt Type: TYPE 1**

Circuit: `CFV → (EP N1→N2 at SFJ) → GSV trunk → (RP N2→N1 downward) → varicosities`

The circuit is contained entirely within the GSV trunk. No escape into tributaries.
Varicosities fill directly from the retrograde GSV trunk, not from a separate N3 escape.

**Ligation target:**
Single point: flush SFJ high ligation (preserve femoral vein). Confirm residual
GSV reflux extent before deciding whether trunk stripping is warranted.

---

### T=09:30 — Final Clip Confirmation + AI "Complete" Check

After confirming no EP N2→N3 exists, the surgeon formally confirms the Type 1
diagnosis. At this point the AI SHOULD fire `action: "complete"` if it has been
following along, since:
- `EP N1→N2` is present (posY=0.06)
- `RP N2→N1` is present (posY=0.18)
- No `EP N2→N3` is in the clip list
- That matches the Type 1 minimum set from the system prompt

**Test: send a probe_move at posY=0.35 after both clips are confirmed:**

```json
{
  "session_id": "test_type1",
  "region": "MID_THIGH",
  "pos_y_ratio": 0.35,
  "surface": "medial",
  "leg": "right"
}
```

Clips in session at this point:
1. `EP N1→N2  posY=0.06  right leg`
2. `RP N2→N1  posY=0.18  right leg`

**Expected AI response:** `action: "complete"` with guidance = "Circuit mapped — sufficient findings for classification".

**Important nuance:** The system prompt's Type 1 complete condition is
`EP N1→N2 + RP N2→N1 (no EP N2→N3 present)`. The model must infer absence
of EP N2→N3 from the clip list. This is the most critical pass/fail test
in this scenario — it verifies the sufficiency logic works.

**Pass if:** `action` = `"complete"` and guidance includes "Circuit" or "classification".
**Fail if:** `action` = `"move"` (model keeps navigating instead of concluding).

---

## Test Event Sequence Summary

| Step | Event       | Key Fields                                         | Expected Action |
|------|-------------|-----------------------------------------------------|-----------------|
| 1    | stream_start| session_id=test_type1                              | session_ready   |
| 2    | probe_move  | SFJ, posY=0.06, anterior-medial, right             | move (→ assess SFJ) |
| 3    | probe_move  | SFJ, posY=0.06 (re-emit, no move)                  | move (→ confirm both maneuvers) |
| 4    | clip_mark   | EP N1→N2, posY=0.06, SFJ, right                   | move (→ trace distally) |
| 5    | probe_move  | UPPER_THIGH, posY=0.18, medial, right              | move (→ assess trunk reflux) |
| 6    | probe_move  | UPPER_THIGH, posY=0.18 (re-emit)                   | move (→ confirm RP) |
| 7    | clip_mark   | RP N2→N1, posY=0.18, UPPER_THIGH, right            | move (→ Hunterian) |
| 8    | probe_move  | HUNTERIAN, posY=0.28, medial, right                | move (→ check tributaries) |
| 9    | probe_move  | MID_THIGH, posY=0.36, medial, right                | move (NOT complete yet) |
| 10   | probe_move  | POPLITEAL, posY=0.47, posterior, right             | move (→ SPJ check) |
| 11   | probe_move  | MID_THIGH, posY=0.35, medial, right                | **complete** ← key test |

---

## Pass/Fail Criteria Summary

| #  | Criterion                                                    | Why it Matters                      |
|----|--------------------------------------------------------------|-------------------------------------|
| P1 | At SFJ with no clips: guidance references SFJ anatomy        | Q1 must direct toward EP search     |
| P2 | After EP N1→N2: guidance directs distally along GSV          | Q2 open after EP confirmed          |
| P3 | At UPPER_THIGH: guidance asks for trunk reflux assessment    | Q2 requires confirming RP           |
| P4 | After RP N2→N1: guidance directs toward Hunterian/tributaries| Q3 open after trunk reflux confirmed|
| P5 | At POPLITEAL with no EP N2→N3: guidance checks SPJ           | SPJ competence rules out SSV circuit|
| P6 | After EP+RP confirmed, no EP N2→N3, posY=0.35: action=complete| Type 1 circuit recognition works  |
| P7 | No premature "complete" before RP N2→N1 is confirmed         | Incomplete circuit ≠ Type 1         |
| P8 | No "maneuver" action unless EP N2→N3 exists in clip list     | Elimination test only for Type 3/1+2|

---

## How to Run This Test

1. Start the backend: `cd Task_2_App/backend && python app.py`
2. Open `stream.html` in browser
3. Start a session (stream_start)
4. For each step in the sequence above:
   - Move the probe icon on the leg diagram to the specified posY
   - Note the guidance text that appears in the right panel
   - Check the action (look for maneuver/complete color changes)
   - Compare against expected guidance
5. Mark clips when the scenario calls for it
6. At step 11 (posY=0.35 with EP+RP clips), verify `action: "complete"` fires

**For scripted replay**, adapt `tests/run_scenario.py` to emit these
socket events using a Socket.IO client (e.g. `python-socketio` library)
and capture each `guidance_update` event for comparison.